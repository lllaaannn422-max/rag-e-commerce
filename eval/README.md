# RAG 系统评测工具

针对本项目（FastAPI + Milvus + Qwen3-Embedding + BGE Rerank + DeepSeek）的检索层与生成层评测脚本。
所有指标基于真实检索结果计算，金标准题目已在 `docs/` 原文中人工核对（见 `golden_set.json` 的 fragments）。

## 目录结构

| 文件 | 作用 |
|---|---|
| `golden_set.json` | 金标准评测集：17 道小说问答题目（题目/参考答案/来源文档/命中关键词） |
| `ingest_docs.py` | 把 `docs/` 文档切片向量化入库（默认跳过 MinIO，`--with-minio` 开启） |
| `eval_lib.py` | 检索 harness（含消融开关）+ 指标计算公共库 |
| `eval_retrieval.py` | 检索层评测：Hit@K / Precision@K / FragRecall@K / MRR / nDCG@K + 4 组消融 |
| `eval_generation.py` | 生成层评测：DeepSeek 生成答案 + LLM-as-Judge 打分（Faithfulness / Answer Relevancy） |
| `run_eval.py` | 一键评测：检索 + 生成 + 输出 `report.md` / `results.json` |
| `probe_db.py` | 查看知识库实际入库的文档与 bucket 分布 |

## 快速开始

```bash
# 1.（可选）先把全部小说入库（当前知识库只有《笑傲江湖》）
python eval/ingest_docs.py

# 2. 一键评测（检索 4 组消融 + 生成层打分）
python eval/run_eval.py

# 冒烟测试（只评 2 题，验证脚本可用）
python eval/run_eval.py --smoke

# 只做检索层 / 只做生成层
python eval/run_eval.py --skip-generation
python eval/run_eval.py --skip-retrieval
```

## 进阶用法

```bash
# Embedding 微调前后对比（基础模型路径按 train_lora.py 中的实际路径填写）
python eval/run_eval.py --compare-base D:\\models\\Qwen3-Embedding-0.6B

# 自定义检索参数
python eval/run_eval.py --recall-k 100 --rerank-top-n 20

# 换 Judge 模型（默认 deepseek-chat，更省）
python eval/run_eval.py --judge-model deepseek-chat
```

## 指标定义

- **Hit@K**：前 K 条结果是否包含金标准片段（`file_name == golden.doc` 且 `chunk_text` 含任一 fragment）。
- **Precision@K**：前 K 条中相关片段占比。
- **FragRecall@K**：金标准答案关键词被前 K 条覆盖的比例（近似召回率）。
- **MRR / nDCG@5**：排序质量指标（nDCG 用二值相关性）。
- **Faithfulness / Answer Relevancy**：DeepSeek 裁判按 1-5 打分后的均值（1 最差 / 5 最好）。

## 消融配置

| 配置 | 多路改写 | BGE Rerank | 目的 |
|---|---|---|---|
| `baseline` | ✗ | ✗ | 单查询直接向量检索 |
| `rewrite_only` | ✓ | ✗ | 多路改写带来的召回提升 |
| `rerank_only` | ✗ | ✓ | Rerank 带来的排序提升 |
| `full` | ✓ | ✓ | 生产完整链路 |

## 注意事项

1. **依赖服务**：DeepSeek API（改写/生成/Judge）、Xinference BGE Rerank（`localhost:9997`，未启动时自动降级并记录 `rerank_available=false`）、本地 Milvus 数据文件。
2. **已入库文档过滤**：脚本自动跳过知识库中不存在文档对应的题目，并在报告中列出缺失文档；先跑 `ingest_docs.py` 可补齐。
3. **Rerank 阈值**：评测时 Rerank 阈值设为 0 以保留 top-N 供指标计算；生产配置（0.7/0.3）不变。
4. **数字真实性**：评测输出的是你当前环境跑出的真实数值，可直接用于简历；换数据/模型/服务后需重跑。
5. 首次运行会加载 0.6B Embedding 模型（约 1-2 分钟），属正常现象。
