# rag_query_engine.py
import re
import numpy as np
from typing import List, Dict, Any, Optional
import concurrent.futures
import config.config as config
from utils.logger import logger
from services.embedding_local_service import EmbeddingService as LocalEmbeddingService
from services.milvus_service import MilvusService
from services.rerank_service import RerankService
from services.deepseek_service import DeepSeekService
from services.minio_service import MinioService
from services.memory_service import MemoryService
from services.web_search_service import WebSearchService


class RAGQueryEngine:
    def __init__(self):
        self.embed_service = LocalEmbeddingService()
        self.milvus_service = MilvusService()
        self.rerank_service = RerankService()
        self.deepseek_service = DeepSeekService()
        self.minio_service = MinioService()
        self.memory_service = MemoryService()
        self.web_search_service = WebSearchService()

    def parse_filter_buckets(self, filter_expr: Optional[str]) -> Optional[List[str]]:
        if not filter_expr:
            return None
        match = re.search(r'source_bucket\s+(?:==|in)\s+\[?(["\w,\s]+)\]?', filter_expr)
        if match:
            parts = re.findall(r'"([^"]+)"', match.group(1))
            if parts:
                return parts
        return None

    def deduplicate_candidates(self, candidates: List[Dict[str, Any]], threshold: float = 0.95, max_candidates: int = 200) -> List[Dict[str, Any]]:
        """向量相似度去重，保留高分候选"""
        if not candidates:
            return []
        sorted_candidates = sorted(candidates, key=lambda x: x.get('total_score', 0), reverse=True)[:max_candidates]
        texts = [c['chunk_text'] for c in sorted_candidates]
        vectors = self.embed_service.encode(texts, is_query=False)
        kept = []
        for i, vec_i in enumerate(vectors):
            is_dup = False
            for j in kept:
                sim = np.dot(vec_i, vectors[j]) / (np.linalg.norm(vec_i) * np.linalg.norm(vectors[j]))
                if sim >= threshold:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(i)
        return [sorted_candidates[i] for i in kept]

    def search_with_rewrite(self, query: str, filter_expr: Optional[str] = None, intent: Optional[str] = None,
                             recall_k: int = 30, rerank_top_n: int = 10, step: int = 50, max_recall_k: int = 100,
                             history: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, Any]]:
        # 1.生成多路改写query（本地/web 两条路径共用）；携带对话历史时做指代消解
        try:
            queries = self.deepseek_service.rewrite_queries(query, history=history)
        except Exception as e:
            # 多次重试后仍为空时降级仅用原问句，保证改写失败不中断整条检索链路
            logger.warning(f"问句改写失败（重试耗尽），降级仅使用原问句: {e}")
            queries = [query]
        logger.info(f"多问句引擎拆分列表: {queries}")
        _ = self.parse_filter_buckets(filter_expr)

        # 2.news 意图：优先联网搜索（Tavily）；失败或无结果时不 return，自然回退到下方本地检索
        if intent == "news" and config.WEB_SEARCH_ENABLED:
            web_docs = self._search_web(query, queries, rerank_top_n)
            if web_docs:
                return web_docs
            logger.warning("联网搜索无有效结果，回退本地 rag-news 向量检索")

        agg_dict = {}
        candidates = []  # 修复存量 bug：首轮检索全空时 break 跳过赋值，candidates 需预先初始化
        current_recall_k = recall_k

        while current_recall_k <= max_recall_k:
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(queries), 8)) as executor:
                future_to_query = {
                    executor.submit(self.milvus_service.search_knowledge, q, current_recall_k, filter_expr): (idx, q)
                    for idx, q in enumerate(queries)
                }
                results_list = []
                for future in concurrent.futures.as_completed(future_to_query):
                    idx, q = future_to_query[future]
                    try:
                        res = future.result()
                        results_list.append((idx, res))
                    except Exception as e:
                        logger.error(f"并发向量检索 Query ['{q}'] 失败: {e}")
                        results_list.append((idx, None))
            results_list.sort(key=lambda x: x[0])
            for idx, results in results_list:
                if not results or len(results) == 0:
                    continue
                weight = 1.0 if idx == 0 else 0.8
                for hit in results[0]:
                    chunk_text = hit.entity.get('chunk_text')
                    if not chunk_text:
                        continue
                    score = hit.distance * weight  # Milvus Lite COSINE: distance 即余弦相似度（越大越相似）
                    if chunk_text in agg_dict:
                        agg_dict[chunk_text]['total_score'] += score
                    else:
                        agg_dict[chunk_text] = {
                            "chunk_text": chunk_text,
                            "file_name": hit.entity.get('file_name'),
                            "file_suffix": hit.entity.get('file_suffix'),
                            "source_bucket": hit.entity.get('source_bucket'),
                            "publish_time": hit.entity.get('publish_time', ''),
                            "total_score": score
                        }
            if not agg_dict:
                break
            candidates = list(agg_dict.values())
            if len(candidates) > 1:
                candidates = self.deduplicate_candidates(candidates)
            candidates.sort(key=lambda x: x['total_score'], reverse=True)
            if len(candidates) >= rerank_top_n:
                break
            current_recall_k += step

        if not candidates:
            return []

        # Rerank重排序
        actual_top_n = min(rerank_top_n, len(candidates))
        doc_texts = [c["chunk_text"] for c in candidates]
        # 原问句可能是代词指代句（如“它的电池容量是多少？”），BGE 对原句打分可能全部低于
        # 阈值导致结果被清空；此时依次用改写后的明确问句（含指代消解）重新打分
        rerank_queries = [query] + [q for q in queries if q != query]
        sorted_indices = []
        for rq in rerank_queries:
            try:
                sorted_indices = self.rerank_service.rerank(rq, doc_texts, top_n=actual_top_n)
            except Exception as e:
                logger.error(f"Rerank 降级使用原始得分排序: {e}")
                sorted_indices = list(range(actual_top_n))
                break
            if sorted_indices:
                break
        if not sorted_indices:
            # 所有问句打分均低于阈值：视为无相关内容，交由调用方返回空结果话术
            return []
        return [candidates[idx] for idx in sorted_indices]

    def _search_web(self, query: str, queries: List[str], rerank_top_n: int) -> List[Dict[str, Any]]:
        """news 意图联网搜索：多路改写并发搜索 → 按 URL 聚合去重 → Rerank 重排

        失败或空结果时返回 []，由调用方 fall-through 回退本地检索。
        """
        try:
            search_queries = queries[:3]  # 截断改写数量，控制 Tavily credit 消耗
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(search_queries), 8)) as executor:
                future_to_q = {executor.submit(self.web_search_service.search, q): q for q in search_queries}
                agg_dict = {}
                for future in concurrent.futures.as_completed(future_to_q):
                    q = future_to_q[future]
                    try:
                        items = future.result()
                    except Exception as e:
                        logger.error(f"联网搜索子查询 ['{q}'] 失败: {e}")
                        continue
                    for item in items:
                        url = item.get("url", "")
                        if not url:
                            continue
                        if url in agg_dict:
                            # 同一 URL 被多路查询命中：得分累加，保留得分最高那条的内容
                            agg_dict[url]["total_score"] += item["total_score"]
                            if item["total_score"] > agg_dict[url]["_best_score"]:
                                agg_dict[url]["_best_score"] = item["total_score"]
                                agg_dict[url]["chunk_text"] = item["chunk_text"]
                                agg_dict[url]["file_name"] = item["file_name"]
                        else:
                            agg_dict[url] = dict(item)
                            agg_dict[url]["_best_score"] = item["total_score"]

            candidates = list(agg_dict.values())
            for c in candidates:
                c.pop("_best_score", None)
            if not candidates:
                return []
            candidates.sort(key=lambda x: x["total_score"], reverse=True)

            # Rerank 重排：web 摘要较短、BGE 打分偏低，阈值降至 0.3；空结果时降级按 Tavily 得分排序
            actual_top_n = min(rerank_top_n, len(candidates))
            doc_texts = [c["chunk_text"] for c in candidates]
            sorted_indices = []
            try:
                sorted_indices = self.rerank_service.rerank(query, doc_texts, top_n=actual_top_n, threshold=0.3)
            except Exception as e:
                logger.error(f"联网结果 Rerank 异常，降级使用 Tavily 得分排序: {e}")
            if not sorted_indices:
                sorted_indices = list(range(actual_top_n))
            return [candidates[idx] for idx in sorted_indices]
        except Exception as e:
            logger.error(f"联网新闻搜索失败，回退本地检索: {e}")
            return []
