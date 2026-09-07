# -*- coding: utf-8 -*-
"""
RAG 系统一键评测：检索层（消融对比）+ 生成层（LLM-as-Judge），输出 Markdown 报告。

用法：
  python eval/run_eval.py                          # 全量评测
  python eval/run_eval.py --smoke                  # 冒烟：只评 2 条题目（快）
  python eval/run_eval.py --limit 10
  python eval/run_eval.py --skip-generation        # 只做检索层
  python eval/run_eval.py --compare-base D:\\models\\Qwen3-Embedding-0.6B
                                                   # 追加「基础模型 vs LoRA 微调」检索对比（耗时约 2 倍）
输出：eval/results.json + eval/report.md
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL_DIR)

from eval.eval_lib import ABLATION_CONFIGS, load_golden  # noqa: E402
from eval.eval_retrieval import run_retrieval_eval, print_table  # noqa: E402
from eval.eval_generation import run_generation_eval  # noqa: E402


def render_report(retrieval: dict, generation: dict, base_comparison: dict = None) -> str:
    lines = []
    lines.append("# RAG 系统评测报告")
    lines.append(f"\n生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"\n评测题目数（已入库文档对应）：检索 {retrieval.get('golden_count')} 条 / 生成 {generation.get('golden_count', 'N/A')} 条")
    lines.append(f"Rerank 服务可用：{'是' if retrieval.get('rerank_available', True) else '否（已降级为原始得分排序）'}")

    lines.append("\n## 一、知识库概览")
    lines.append("\n| 文档 (bucket) | Chunk 数 |")
    lines.append("|---|---|")
    for key, cnt in retrieval.get("kb_overview", {}).items():
        lines.append(f"| {key} | {cnt} |")

    lines.append("\n## 二、检索层指标（消融对比，均值 0-1）")
    lines.append("\n| 配置 | Hit@1 | Hit@3 | Hit@5 | MRR | nDCG@5 | FragRecall@10 |")
    lines.append("|---|---|---|---|---|---|---|")
    for name, _ in ABLATION_CONFIGS:
        agg = retrieval.get(name, {}).get("aggregate", {})
        lines.append(
            f"| {name} | {agg.get('hit@1', 0):.4f} | {agg.get('hit@3', 0):.4f} | "
            f"{agg.get('hit@5', 0):.4f} | {agg.get('mrr', 0):.4f} | "
            f"{agg.get('ndcg@5', 0):.4f} | {agg.get('frag_recall@10', 0):.4f} |"
        )

    if base_comparison:
        lines.append("\n### Embedding 微调前后对比（full 配置）")
        lines.append("\n| 模型 | Hit@1 | Hit@3 | MRR | nDCG@5 |")
        lines.append("|---|---|---|---|---|")
        for label, res in (("LoRA 微调后（当前部署）", retrieval.get("full", {}).get("aggregate", {})),
                           ("基础模型", base_comparison.get("aggregate", {}))):
            lines.append(
                f"| {label} | {res.get('hit@1', 0):.4f} | {res.get('hit@3', 0):.4f} | "
                f"{res.get('mrr', 0):.4f} | {res.get('ndcg@5', 0):.4f} |"
            )

    lines.append("\n## 三、生成层评分（LLM-as-Judge，1-5 越高越好）")
    agg = generation.get("aggregate", {})
    lines.append(f"\n- Faithfulness（忠实度）：**{agg.get('faithfulness', 'N/A')}** / 5（{agg.get('judged_count', 0)} 条）")
    lines.append(f"- Answer Relevancy（相关性）：**{agg.get('answer_relevancy', 'N/A')}** / 5")
    lines.append(f"- CS Quality（客服质量）：**{agg.get('cs_quality', 'N/A')}** / 5")
    lines.append("\n### 逐题明细")
    lines.append("\n| 题目 | Faithfulness | Relevancy | CS质量 | 模型答案 |")
    lines.append("|---|---|---|---|---|")
    for q in generation.get("per_query", []):
        ans = (q.get("answer") or "").replace("\n", " ")[:60]
        lines.append(f"| {q['id']} {q['query']} | {q.get('faithfulness')} | {q.get('answer_relevancy')} | {q.get('cs_quality')} | {ans} |")

    lines.append("\n## 四、指标说明")
    lines.append("""
- **Hit@K**：前 K 条检索结果中是否出现与金标准匹配的片段（片段判定：file_name 与 golden.doc 一致，且 chunk_text 包含 golden.fragments 中任一关键词）。
- **FragRecall@K**：金标准答案关键词被前 K 条结果覆盖的比例（近似召回率）。
- **MRR / nDCG@5**：排序质量；nDCG 采用二值相关性。
- **Faithfulness / Answer Relevancy / CS Quality**：由 DeepSeek（judge）按 1-5 打分，分数为多条题目的均值；CS Quality 衡量客服规范（共情/语气/不夸大政策/信息不足引导人工）。
- 消融配置：baseline（单查询+无精排）、rewrite_only（+多路改写）、rerank_only（+BGE Rerank）、full（改写+精排）。
- 评测结果仅反映当前知识库数据与配置，数字随数据/模型/服务状态变化。
""")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="RAG 系统一键评测")
    parser.add_argument("--golden", default=os.path.join(EVAL_DIR, "golden_set.json"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="冒烟：只评 2 条题目")
    parser.add_argument("--skip-retrieval", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--recall-k", type=int, default=50)
    parser.add_argument("--rerank-top-n", type=int, default=10)
    parser.add_argument("--judge-model", default="deepseek-chat")
    parser.add_argument("--compare-base", default=None, help="基础 Embedding 模型路径（追加微调前后对比）")
    parser.add_argument("--json-out", default=os.path.join(EVAL_DIR, "results.json"))
    parser.add_argument("--report-out", default=os.path.join(EVAL_DIR, "report.md"))
    args = parser.parse_args()

    golden = load_golden(args.golden)
    if args.smoke:
        args.limit = 2

    retrieval = {}
    generation = {"aggregate": {}, "per_query": [], "golden_count": 0}
    base_comparison = None

    if not args.skip_retrieval:
        print(">>> 检索层评测（4 组消融）...")
        retrieval = run_retrieval_eval(
            golden, limit=args.limit, recall_k=args.recall_k, rerank_top_n=args.rerank_top_n,
            only_available_docs=True,
        )
        print_table(retrieval)

    if not args.skip_generation:
        print(">>> 生成层评测（DeepSeek 生成 + Judge 打分）...")
        gen_golden = golden[:args.limit] if args.limit else golden
        generation = run_generation_eval(gen_golden, judge_model=args.judge_model,
                                         recall_k=args.recall_k, rerank_top_n=args.rerank_top_n)

    if args.compare_base:
        print(f">>> 基础模型检索对比（{args.compare_base}）...")
        tmp = os.path.join(tempfile.gettempdir(), "eval_base_retrieval.json")
        cmd = [
            sys.executable, os.path.join(EVAL_DIR, "eval_retrieval.py"),
            "--golden", args.golden, "--config", "full",
            "--embed-path", args.compare_base, "--json-out", tmp,
        ]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
        with open(tmp, "r", encoding="utf-8") as f:
            base_comparison = json.load(f).get("full", {})

    if not retrieval and not generation.get("per_query"):
        print("没有可用的评测结果，请检查参数。")
        return

    report = render_report(retrieval, generation, base_comparison)
    with open(args.report_out, "w", encoding="utf-8") as f:
        f.write(report)

    payload = {"retrieval": retrieval, "generation": generation, "base_comparison": base_comparison}
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n报告已生成: {args.report_out}")
    print(f"结果 JSON: {args.json_out}")


if __name__ == "__main__":
    main()
