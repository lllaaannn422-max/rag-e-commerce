# -*- coding: utf-8 -*-
"""
RAG 评测公共库：检索 harness + 指标计算。

设计要点：
- 直接复用项目底层服务（本地 Qwen3-Embedding、Milvus、BGE Rerank、DeepSeek），
  但自行封装检索逻辑，以便对「多路改写 / Rerank」做消融开关控制；
- 金标准判定：检索 chunk 的 file_name 与 golden.doc 一致，且 chunk_text 包含任一 fragment；
- 所有指标基于真实检索结果计算，不估算、不编造。
"""
import concurrent.futures
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import config.config as config
from utils.logger import logger


# ============================================================
# 独立的 Embedding 封装（避免项目单例冲突，支持指定模型路径）
# ============================================================
class _DirectEmbed:
    """基于 sentence-transformers 的直接封装，API 与项目 EmbeddingService 兼容。"""

    def __init__(self, model_path: str):
        import torch
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"[eval] 加载 Embedding 模型: {model_path} (device={device})")
        self._model = SentenceTransformer(model_path, device=device)

    def encode(self, texts, is_query: bool = False, **kwargs) -> np.ndarray:
        embeddings = self._model.encode(
            texts,
            prompt_name="query" if is_query else None,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> List[float]:
        return self.encode(query, is_query=True).tolist()


# ============================================================
# 检索 Harness（与生产 pipeline 同源，但支持消融开关）
# ============================================================
class RetrievalHarness:
    def __init__(self, embed_model_path: Optional[str] = None):
        from pymilvus import Collection, connections

        model_path = embed_model_path or getattr(config, "LOCAL_EMBED_MODEL_PATH", None)
        if not model_path:
            raise RuntimeError("未找到 Embedding 模型路径，请检查 config.LOCAL_EMBED_MODEL_PATH")
        if not Path(model_path).exists():
            raise RuntimeError(f"Embedding 模型路径不存在: {model_path}")

        self.embed = _DirectEmbed(model_path)
        db_path = Path(config.MILVUS_DB_PATH).resolve()
        logger.info(f"[eval] 连接 Milvus: {db_path}")
        try:
            connections.connect(alias="eval_conn", uri=db_path.as_posix())
            self.collection = Collection(config.COLLECTION_NAME, using="eval_conn")
            self.collection.load()
        except Exception as e:
            raise RuntimeError(
                f"无法打开 Milvus 数据文件 {db_path}（{e}）。\n"
                f"Milvus Lite 同一时刻只允许一个进程访问数据文件：\n"
                f"  1) 请先停止 FastAPI 服务（api_server.py / uvicorn）以及所有占用该文件的进程；\n"
                f"  2) 如服务带 --reload 启动，评测过程中不要修改项目文件，避免触发自动重启抢占锁；\n"
                f"  3) 确认没有其他 python 进程持有 data/milvus.db 的锁后再重试。"
            ) from e

        from services.deepseek_service import DeepSeekService
        from services.rerank_service import RerankService

        self.deepseek = DeepSeekService()
        self.rerank = RerankService()
        self.rerank_available = True  # 首次调用失败后置 False（服务未启动时降级）

    # ---------- 底层检索 ----------
    def _search(self, query_vec: np.ndarray, top_k: int, expr: Optional[str] = None) -> List[Any]:
        # FLAT 索引无聚类，nprobe 无效，params 留空即可
        search_params = {"metric_type": "COSINE", "params": {}}
        results = self.collection.search(
            data=[query_vec.tolist()],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["chunk_text", "file_name", "file_suffix", "source_bucket", "publish_time"],
        )
        return results[0]

    # ---------- 向量相似度去重（同生产 deduplicate_candidates） ----------
    def _dedup(self, candidates: List[Dict[str, Any]], threshold: float = 0.95, max_candidates: int = 200) -> List[Dict[str, Any]]:
        sorted_candidates = sorted(candidates, key=lambda x: x.get("total_score", 0), reverse=True)[:max_candidates]
        texts = [c["chunk_text"] for c in sorted_candidates]
        vectors = self.embed.encode(texts, is_query=False)
        kept = []
        for i, vec_i in enumerate(vectors):
            is_dup = False
            for j in kept:
                sim = float(np.dot(vec_i, vectors[j]) / (np.linalg.norm(vec_i) * np.linalg.norm(vectors[j])))
                if sim >= threshold:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(i)
        return [sorted_candidates[i] for i in kept]

    # ---------- 主检索入口（支持消融） ----------
    def retrieve(
        self,
        query: str,
        use_rewrite: bool = True,
        use_rerank: bool = True,
        recall_k: int = 50,
        rerank_top_n: int = 10,
        bucket: Optional[str] = None,
        use_dedup: bool = True,
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """返回 (排序后的候选列表, rerank 是否可用)。候选字段与生产一致：chunk_text/file_name/source_bucket/total_score。"""
        # 1. 多路改写（可选）
        queries = [query]
        if use_rewrite:
            try:
                queries = self.deepseek.rewrite_queries(query)  # 已包含原问题
            except Exception as e:
                logger.warning(f"[eval] 查询改写失败，回退单查询: {e}")

        # 2. 预计算各子查询向量（复用，避免重复编码）
        query_vecs = {}
        for q in queries:
            try:
                query_vecs[q] = self.embed.encode(q, is_query=True)
            except Exception as e:
                logger.error(f"[eval] 查询向量化失败 ['{q}']: {e}")
        queries = [q for q in queries if q in query_vecs]
        if not queries:
            return [], self.rerank_available

        expr = f'source_bucket == "{bucket}"' if bucket else None

        # 3. 并发向量检索 + 加权融合（原问题 1.0 / 改写 0.8，同生产）
        agg_dict: Dict[str, Dict[str, Any]] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(queries), 8)) as executor:
            future_map = {
                executor.submit(self._search, query_vecs[q], recall_k, expr): (idx, q)
                for idx, q in enumerate(queries)
            }
            results_list = []
            for future in concurrent.futures.as_completed(future_map):
                idx, q = future_map[future]
                try:
                    results_list.append((idx, future.result()))
                except Exception as e:
                    logger.error(f"[eval] 向量检索失败 ['{q}']: {e}")
                    results_list.append((idx, []))
        results_list.sort(key=lambda x: x[0])

        for idx, hits in results_list:
            weight = 1.0 if idx == 0 else 0.8
            for hit in hits:
                chunk_text = hit.entity.get("chunk_text")
                if not chunk_text:
                    continue
                # Milvus Lite COSINE 返回的 distance 即余弦相似度（越大越相似，降序返回）
                score = hit.distance * weight
                if chunk_text in agg_dict:
                    agg_dict[chunk_text]["total_score"] += score
                else:
                    agg_dict[chunk_text] = {
                        "chunk_text": chunk_text,
                        "file_name": hit.entity.get("file_name"),
                        "file_suffix": hit.entity.get("file_suffix"),
                        "source_bucket": hit.entity.get("source_bucket"),
                        "publish_time": hit.entity.get("publish_time", ""),
                        "total_score": score,
                    }

        candidates = list(agg_dict.values())
        if not candidates:
            return [], self.rerank_available

        # 4. 相似度去重（可选）
        if use_dedup and len(candidates) > 1:
            candidates = self._dedup(candidates)

        candidates.sort(key=lambda x: x["total_score"], reverse=True)

        # 5. Rerank 精排（可选；threshold=0 以保留 top_n 用于指标计算）
        if use_rerank:
            actual_top_n = min(rerank_top_n, len(candidates))
            doc_texts = [c["chunk_text"] for c in candidates]
            try:
                sorted_indices = self.rerank.rerank(query, doc_texts, top_n=actual_top_n, threshold=0.0)
                candidates = [candidates[i] for i in sorted_indices]
            except Exception as e:
                logger.warning(f"[eval] Rerank 不可用（{e}），降级为原始得分排序")
                self.rerank_available = False
                candidates = candidates[:rerank_top_n]
        else:
            candidates = candidates[:rerank_top_n]

        return candidates, self.rerank_available


# ============================================================
# 金标准匹配与指标
# ============================================================
def is_relevant(doc: Dict[str, Any], golden: Dict[str, Any]) -> bool:
    if doc.get("file_name") != golden.get("doc"):
        return False
    text = doc.get("chunk_text", "") or ""
    return any(frag in text for frag in golden.get("fragments", []))


def _ndcg_at_k(relevance: List[int], k: int) -> float:
    rel = relevance[:k]
    dcg = sum(r / math.log2(i + 1) for i, r in enumerate(rel, 1))
    ideal = sorted(rel, reverse=True)
    idcg = sum(r / math.log2(i + 1) for i, r in enumerate(ideal, 1))
    return dcg / idcg if idcg > 0 else 0.0


def _fragment_recall(ranked: List[Dict[str, Any]], golden: Dict[str, Any], k: int) -> float:
    found = set()
    for doc in ranked[:k]:
        if doc.get("file_name") != golden.get("doc"):
            continue
        text = doc.get("chunk_text", "") or ""
        for frag in golden.get("fragments", []):
            if frag in text:
                found.add(frag)
    total = len(golden.get("fragments", []))
    return (len(found) / total) if total else 0.0


def compute_metrics(ranked: List[Dict[str, Any]], golden: Dict[str, Any], ks=(1, 3, 5, 10)) -> Dict[str, float]:
    """单条 query 的检索指标。"""
    rel = [1 if is_relevant(d, golden) else 0 for d in ranked]
    out: Dict[str, float] = {}
    for k in ks:
        top = rel[:k]
        out[f"hit@{k}"] = 1.0 if any(top) else 0.0
        out[f"precision@{k}"] = sum(top) / k
        out[f"frag_recall@{k}"] = _fragment_recall(ranked, golden, k)
    mrr = 0.0
    for i, r in enumerate(rel, 1):
        if r:
            mrr = 1.0 / i
            break
    out["mrr"] = mrr
    out["ndcg@5"] = _ndcg_at_k(rel, 5)
    return out


def aggregate_metrics(per_query: List[Dict[str, float]]) -> Dict[str, float]:
    """对多条 query 的指标取平均。"""
    if not per_query:
        return {}
    keys = per_query[0].keys()
    return {k: round(sum(q[k] for q in per_query) / len(per_query), 4) for k in keys}


def load_golden(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"]


# ============================================================
# 消融配置
# ============================================================
ABLATION_CONFIGS = [
    ("baseline", dict(use_rewrite=False, use_rerank=False)),
    ("rewrite_only", dict(use_rewrite=True, use_rerank=False)),
    ("rerank_only", dict(use_rewrite=False, use_rerank=True)),
    ("full", dict(use_rewrite=True, use_rerank=True)),
]
