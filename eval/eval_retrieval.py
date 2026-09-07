# -*- coding: utf-8 -*-
"""
检索层评测：对金标准集计算 Hit@K / Precision@K / FragRecall@K / MRR / nDCG@K，
并支持 4 组消融对比（baseline / rewrite_only / rerank_only / full）。

用法示例：
  python eval/eval_retrieval.py                     # 跑全部 4 组消融
  python eval/eval_retrieval.py --config full       # 只跑 full
  python eval/eval_retrieval.py --limit 5           # 只跑前 5 题
  python eval/eval_retrieval.py --embed-path D:\\models\\Qwen3-Embedding-0.6B --config full
                                                    # 用基础模型跑 full（LoRA 微调前后对比）
  python eval/eval_retrieval.py --json-out eval/results_retrieval.json
"""
import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.logger import logger
from eval.eval_lib import (
    ABLATION_CONFIGS,
    RetrievalHarness,
    aggregate_metrics,
    compute_metrics,
    load_golden,
)


def kb_overview(harness: RetrievalHarness) -> dict:
    """知识库概览：每个文档的 chunk 数 / bucket。"""
    from collections import Counter

    rows = harness.collection.query(
        expr="id >= 0", output_fields=["file_name", "source_bucket"], limit=50000
    )
    counter = Counter((r["file_name"], r["source_bucket"]) for r in rows)
    return {f"{fname} ({bucket})": cnt for (fname, bucket), cnt in sorted(counter.items())}


def run_retrieval_eval(golden, limit=None, recall_k=50, rerank_top_n=10, bucket=None,
                       configs=None, embed_path=None, use_dedup=True, only_available_docs=False):
    harness = RetrievalHarness(embed_model_path=embed_path)
    overview = kb_overview(harness)

    # overview key 形如 "file (bucket)"，解析出已入库文件名集合
    available_files = set()
    for key in overview:
        available_files.add(key.split(" (")[0])

    if only_available_docs:
        before = len(golden)
        # 未入库文档统计基于本次实际评测的 golden 集（修复：原先硬编码 golden_set.json）
        missing = sorted({g["doc"] for g in golden if g["doc"] not in available_files})
        golden = [g for g in golden if g["doc"] in available_files]
        logger.info(f"知识库已入库文档: {sorted(available_files)}")
        logger.info(f"评测题目过滤: {before} -> {len(golden)}（未入库文档对应题目跳过）")
        if missing:
            logger.warning(f"以下文档尚未入库，对应题目未评测: {missing}")

    if limit:
        golden = golden[:limit]

    configs = configs or ABLATION_CONFIGS
    all_results = {}
    rerank_ok = True
    for name, opts in configs:
        logger.info(f"===== 消融配置: {name} ({opts}) =====")
        per_query = []
        details = []
        for g in golden:
            try:
                ranked, rerank_available = harness.retrieve(
                    g["query"],
                    recall_k=recall_k,
                    rerank_top_n=rerank_top_n,
                    bucket=bucket,
                    use_dedup=use_dedup,
                    **opts,
                )
                rerank_ok = rerank_ok and rerank_available
                m = compute_metrics(ranked, g)
                top_files = [d.get("file_name") for d in ranked[:5]]
                error = None
            except Exception as e:
                logger.error(f"[eval] 题目 {g['id']} 检索失败: {e}")
                m = {k: 0.0 for k in
                     ["hit@1", "precision@1", "frag_recall@1", "hit@3", "precision@3", "frag_recall@3",
                      "hit@5", "precision@5", "frag_recall@5", "hit@10", "precision@10", "frag_recall@10",
                      "mrr", "ndcg@5"]}
                top_files = []
                error = str(e)
            per_query.append(m)
            details.append({
                "id": g["id"],
                "query": g["query"],
                "doc": g["doc"],
                "top_files": top_files,
                "metrics": m,
                "error": error,
            })
        agg = aggregate_metrics(per_query)
        all_results[name] = {"config": opts, "aggregate": agg, "per_query": details}
        logger.info(f"[{name}] 聚合指标: {json.dumps(agg, ensure_ascii=False)}")

    all_results["kb_overview"] = overview
    all_results["rerank_available"] = rerank_ok
    all_results["golden_count"] = len(golden)
    return all_results


def print_table(all_results: dict):
    """打印消融对比表。"""
    rows = [k for k in all_results if k in [c[0] for c in ABLATION_CONFIGS]]
    metrics = ["hit@1", "hit@3", "hit@5", "mrr", "ndcg@5", "frag_recall@10"]
    header = "配置".ljust(14) + "".join(m.ljust(12) for m in metrics)
    print("\n" + "=" * len(header))
    print("检索层消融对比（数值为多条题目均值，0-1）")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for name in rows:
        agg = all_results[name]["aggregate"]
        line = name.ljust(14)
        for m in metrics:
            line += f"{agg.get(m, 0):.4f}".ljust(12)
        print(line)
    print("-" * len(header))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="检索层评测（Hit@K/MRR/nDCG@K + 消融）")
    parser.add_argument("--golden", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_set.json"))
    parser.add_argument("--limit", type=int, default=None, help="只评测前 N 条题目")
    parser.add_argument("--recall-k", type=int, default=50)
    parser.add_argument("--rerank-top-n", type=int, default=10)
    parser.add_argument("--bucket", default=None, help="按 bucket 过滤（如 rag-fiction）")
    parser.add_argument("--config", choices=[c[0] for c in ABLATION_CONFIGS] + ["all"], default="all")
    parser.add_argument("--embed-path", default=None, help="指定 Embedding 模型路径（用于微调前后对比）")
    parser.add_argument("--no-dedup", action="store_true", help="关闭向量相似度去重")
    parser.add_argument("--all-docs", action="store_true", help="不过滤未入库文档（默认自动过滤）")
    parser.add_argument("--json-out", default=None, help="结果 JSON 输出路径")
    args = parser.parse_args()

    golden = load_golden(args.golden)
    configs = ABLATION_CONFIGS if args.config == "all" else [c for c in ABLATION_CONFIGS if c[0] == args.config]

    results = run_retrieval_eval(
        golden,
        limit=args.limit,
        recall_k=args.recall_k,
        rerank_top_n=args.rerank_top_n,
        bucket=args.bucket,
        configs=configs,
        embed_path=args.embed_path,
        use_dedup=not args.no_dedup,
        only_available_docs=not args.all_docs,
    )
    print_table(results)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已写入: {args.json_out}")
