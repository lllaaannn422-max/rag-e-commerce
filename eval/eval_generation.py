# -*- coding: utf-8 -*-
"""
生成层评测：对金标准集跑「完整检索 -> DeepSeek 生成答案 -> LLM-as-Judge 打分」，
Judge 对每条答案打三个维度（1-5 分）：
  - faithfulness（忠实度）：答案是否完全基于检索上下文，有无编造/与文档矛盾；
  - answer_relevancy（答案相关性）：答案是否直接、完整地回答用户问题；
  - cs_quality（客服质量）：是否符合电商客服规范（共情/语气/不夸大政策/信息不足引导人工）。

用法示例：
  python eval/eval_generation.py                 # 评测全部已入库文档对应题目
  python eval/eval_generation.py --limit 5
  python eval/eval_generation.py --judge-model deepseek-chat --top-n 3
"""
import argparse
import json
import os
import re
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config.config as config
from utils.logger import logger
from eval.eval_lib import RetrievalHarness, load_golden

JUDGE_PROMPT = """你是 RAG 问答系统的质量评审员。请从三个维度对「模型生成的答案」打分，均为 1-5 整数：

1. faithfulness（忠实度）：答案中的每一条信息是否都能在下面的「文档片段」中找到依据？有没有编造、臆测或与文档矛盾的内容？
   5=完全基于文档且无任何编造；3=大部分有依据但有少量超出文档的信息；1=大量编造或与文档明显矛盾。
2. answer_relevancy（答案相关性）：答案是否直接、完整地回答了用户问题？有没有答非所问或回避问题？
   5=直接完整回答；3=部分回答或略有偏离；1=基本不相关。
3. cs_quality（客服质量）：是否符合电商客服规范？是否先共情或直接给结论、语气口语化友好？对政策类问题是否严格按文档而不夸大承诺？信息不足时是否明确说明不编造并引导人工？
   5=完全符合客服规范；3=基本符合但有明显不足；1=不符合（如编造政策、语气生硬）。

用户问题：{query}

提供的文档片段：
{context}

模型生成的答案：
{answer}

只输出一个 JSON 对象，不要输出其他内容，格式：
{{"faithfulness": 整数1-5, "answer_relevancy": 整数1-5, "cs_quality": 整数1-5, "reason": "一句话评价"}}"""


def _call_llm(client, model, messages, temperature=0.2, max_tokens=500) -> str:
    resp = client.chat.completions.create(
        model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
    )
    return resp.choices[0].message.content.strip()


def judge_answer(query: str, context_docs: list, answer: str, judge_model: str = "deepseek-chat") -> dict:
    """DeepSeek 作为裁判，对答案的忠实度与相关性打分（严格 JSON 输出）。"""
    from openai import OpenAI

    client = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    context_text = "\n\n".join(
        f"[文档{i}] {d.get('chunk_text', '')[:500]}" for i, d in enumerate(context_docs, 1)
    )
    prompt = JUDGE_PROMPT.format(query=query, context=context_text, answer=answer)
    content = _call_llm(
        client, judge_model,
        [{"role": "system", "content": "你是严格的评测员，只输出 JSON。"},
         {"role": "user", "content": prompt}],
    )
    m = re.search(r"\{.*\}", content, re.DOTALL)
    if not m:
        raise ValueError(f"Judge 未返回有效 JSON: {content[:200]}")
    data = json.loads(m.group())
    return {
        "faithfulness": int(data.get("faithfulness", 0)),
        "answer_relevancy": int(data.get("answer_relevancy", 0)),
        "cs_quality": int(data.get("cs_quality", 0)),
        "reason": str(data.get("reason", "")),
    }


def run_generation_eval(golden, top_n=3, judge_model="deepseek-chat", recall_k=50, rerank_top_n=10):
    harness = RetrievalHarness()

    # 已入库文档过滤
    from eval.eval_retrieval import kb_overview
    overview = kb_overview(harness)
    available = {k.split(" (")[0] for k in overview}
    before = len(golden)
    golden = [g for g in golden if g["doc"] in available]
    logger.info(f"生成评测题目: {before} -> {len(golden)}（未入库文档跳过）")

    per_query = []
    for g in golden:
        logger.info(f"评测题目: {g['id']} {g['query']}")
        try:
            ranked, _ = harness.retrieve(
                g["query"], use_rewrite=True, use_rerank=True,
                recall_k=recall_k, rerank_top_n=rerank_top_n, use_dedup=True,
            )
            context = ranked[:top_n]
            if not context:
                per_query.append({"id": g["id"], "query": g["query"], "answer": "(未检索到上下文)",
                                  "faithfulness": 0, "answer_relevancy": 0, "cs_quality": 0, "error": "no_context"})
                continue

            answer = harness.deepseek.generate_answer(g["query"], context)
            if not (answer or "").strip():  # 偶发空返回，重试一次
                answer = harness.deepseek.generate_answer(g["query"], context)
            try:
                judge = judge_answer(g["query"], context, answer, judge_model=judge_model)
            except Exception as e:
                logger.error(f"Judge 打分失败: {e}")
                judge = {"faithfulness": None, "answer_relevancy": None, "cs_quality": None, "reason": f"judge_error: {e}"}

            per_query.append({
                "id": g["id"],
                "query": g["query"],
                "answer": answer,
                "faithfulness": judge["faithfulness"],
                "answer_relevancy": judge["answer_relevancy"],
                "cs_quality": judge["cs_quality"],
                "reason": judge["reason"],
            })
            logger.info(f"  faithfulness={judge['faithfulness']} answer_relevancy={judge['answer_relevancy']} cs_quality={judge['cs_quality']}")
        except Exception as e:
            logger.error(f"题目 {g['id']} 生成评测失败: {e}")
            per_query.append({"id": g["id"], "query": g["query"], "answer": f"(评测异常: {e})",
                              "faithfulness": None, "answer_relevancy": None, "cs_quality": None, "error": str(e)})

    scores = [q for q in per_query if q.get("faithfulness") is not None]
    aggregate = {}
    if scores:
        aggregate = {
            "faithfulness": round(sum(q["faithfulness"] for q in scores) / len(scores), 2),
            "answer_relevancy": round(sum(q["answer_relevancy"] for q in scores) / len(scores), 2),
            "cs_quality": round(sum(q["cs_quality"] for q in scores) / len(scores), 2),
            "judged_count": len(scores),
        }
    return {"aggregate": aggregate, "per_query": per_query, "golden_count": len(golden)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成层评测（LLM-as-Judge 打分）")
    parser.add_argument("--golden", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_set.json"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-n", type=int, default=3, help="送入生成器的上下文条数")
    parser.add_argument("--judge-model", default="deepseek-chat", help="Judge 使用的大模型")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    golden = load_golden(args.golden)
    if args.limit:
        golden = golden[:args.limit]
    results = run_generation_eval(golden, top_n=args.top_n, judge_model=args.judge_model)
    print("\n===== 生成层评分（1-5，越高越好） =====")
    print(json.dumps(results["aggregate"], ensure_ascii=False, indent=2))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"结果已写入: {args.json_out}")
