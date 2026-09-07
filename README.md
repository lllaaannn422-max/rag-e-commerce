# RAG 智能客服问答系统（意图路由 + 多轮记忆 + 重排序）

基于 **FastAPI + Milvus + Redis + Qwen3-Embedding + BGE-Reranker + DeepSeek** 的电商客服 RAG 问答系统：
覆盖「文档摄入 → 切片向量化 → 意图识别 → 多路召回 → 精排 → 生成」全链路，支持售前 / 售后 / 活动 / 对比 / 订单查询 / 转人工 / 闲聊等多类意图的分流处理，并内置 **Redis 多轮对话记忆（含指代消解）**。

> 项目由通用知识库 RAG 演化而来：早期面向小说语料问答（金庸小说全文 1.2 万+ Chunk），现聚焦电商客服场景（产品手册 / FAQ / 售后政策 / 活动规则 4 类知识域 + 模拟订单库）。版权语料与模型权重**未随仓库分发**，见文末说明。

---

## 功能特性

- **9 类意图识别与路由**：LLM JSON 结构化输出分类（pre_sales / after_sales / promotion / comparison / order_status / escalation / chitchat / clarification / other），confidence < 0.7 自动降级关键词规则兜底。
- **4 条处理路径**：
  | 路径 | 触发意图 | 处理方式 |
  |---|---|---|
  | `rag` | 售前 / 售后 / 活动 / 对比等 | 意图 → Milvus bucket 过滤 → 多路召回 → Rerank → 生成 |
  | `order` | 订单查询 | 正则快路径 → LLM 抽取兜底 → mock 订单库精确匹配；未命中返回固定话术，**杜绝编造物流信息** |
  | `escalation` | 投诉 / 转人工 | 共情安抚话术 + `needs_handoff=true` 标记人工介入 |
  | `chitchat` | 闲聊 | 简短问候并引导业务话题 |
- **检索质量优化**（自建金标准评测验证）：LLM 多路查询改写 → ThreadPoolExecutor 并发召回 → 加权融合（原题 1.0 / 改写 0.8）→ 余弦去重（0.95）→ BGE-Reranker 交叉编码精排 + 相关性阈值过滤，Rerank 异常自动回退向量得分排序。
- **Redis 多轮对话记忆**：改写阶段结合对话历史做**指代消解**（"他师父是谁？" → "令狐冲的师父是谁？"这类代词补全为可独立检索的问句）；消息限长 / 条数裁剪 / TTL 滑动过期治理；Redis 故障时**静默降级为无状态单轮**、恢复后懒重连自动启用（30s 冷却期防黑洞端口拖慢请求）。
- **Embedding 本地化 + 领域微调**：Qwen3-Embedding-0.6B（1024 维、query 指令前缀）本地部署；基于 PEFT/LoRA（r=8 / alpha=16 / MultipleNegativesRankingLoss）在领域语料上微调后合并权重部署。
- **自建评测体系**（`eval/`）：金标准评测集 + Hit@K / MRR / nDCG@K + 4 组消融对比 + LLM-as-Judge 生成质量打分，脚本支持任意场景金标准集扩展。
- **高可用设计**：指数退避重试装饰器覆盖全部外部调用；每类分支均有降级兜底，单点故障不中断主链路。

---

## 系统架构

```
                         ┌──────────────────────────────────────────┐
  用户 ── HTTP ──► FastAPI (api_server.py)                          │
                         │ 1. intent 识别 (DeepSeek, 置信度<0.7降级规则)│
                         │ 2. 按意图分流：                            │
                         │    rag / order / escalation / chitchat   │
                         ├──────────────────────────────────────────┤
  rag 路径:                                                          │
   ├─ 多路查询改写 (DeepSeek, 携带对话历史做指代消解)                   │
   ├─ 并发向量召回 (ThreadPoolExecutor + Milvus, source_bucket 过滤)  │
   ├─ 加权融合 → 余弦去重 → BGE-Reranker 精排                        │
   └─ 答案生成 (DeepSeek, 结合历史 + 上下文 + 引用 sources)            │
                                                                     │
  基础设施:                                                          │
   ├─ Milvus(Lite/Standalone) 向量库     IVF_FLAT + COSINE, 1024 维  │
   ├─ Redis 多轮对话记忆 (List+JSON, TTL, 可降级无状态)               │
   ├─ MinIO 原始文档归档                                             │
   ├─ Xinference: bge-reranker-v2-m3 (可选)                         │
   └─ DeepSeek API (意图/改写/生成)                                  │
                         └──────────────────────────────────────────┘
```

组件与技术栈：Python 3.10 · FastAPI + Pydantic v2 · pymilvus / Milvus Lite · redis-py · sentence-transformers · Qwen3-Embedding-0.6B · BGE-Reranker-v2-M3 · DeepSeek · MinIO · Docker Compose · Gunicorn。

---

## 目录结构

```
rag_project/
├── api_server.py                 # FastAPI 入口：/api/v1/chat、/api/v1/documents/ingest、/api/v1/conversations/{id}
├── config/config.py               # 集中配置（密钥一律从 .env 读取）
├── services/
│   ├── embedding_local_service.py # Qwen3-Embedding 本地加载（单例）
│   ├── milvus_service.py          # Milvus 集合/索引/插入/检索封装
│   ├── rerank_service.py          # BGE Rerank（Xinference HTTP）
│   ├── deepseek_service.py        # 意图识别 / 多路改写 / 答案生成 / 订单抽取 / 话术分支
│   ├── memory_service.py          # Redis 多轮记忆（限长/裁剪/TTL/降级/懒重连）
│   ├── order_service.py           # mock 订单查询（正则快路径 + lookup）
│   ├── minio_service.py           # 对象存储
│   └── web_search_service.py      # Tavily 联网搜索（客服场景默认关闭）
├── pipeline/
│   ├── document_processor.py      # 解析(pdf/txt/md) + 切片（滑窗 350/60，FAQ 问答对切块）
│   └── rag_query_engine.py        # 检索链路编排（改写/并发召回/融合/去重/精排）
├── docs/ecommerce/                # 演示语料：产品手册/FAQ/售后政策/活动规则（md）
├── data/mock_orders.json          # 模拟订单库（order_status 路径演示）
├── eval/                          # 评测工具链（脚本入库，产物不入库）
├── test/                          # 单元/端到端/防护测试
├── docker-compose.yml             # 容器化部署（MinIO/Etcd/Milvus/Redis/rag-app）
└── .env.example                   # 环境变量模板（复制为 .env 填真实密钥）
```

---

## 快速开始（本地）

### 1. 环境准备

- Python 3.10+（Windows/Linux 均可）
- Redis（可选）：`localhost:6379`；**不可用时系统自动降级为无状态单轮**，不影响运行
- 本地 Embedding 模型：`Qwen3-Embedding-0.6B`（模型权重 >4GB，未入库）

```bash
# 依赖
pip install -r requirements.txt

# 密钥：复制模板并填入
cp .env.example .env        # Windows: copy .env.example .env
```

### 2. 准备 Embedding / Rerank 模型

仓库未分发模型权重（体积原因，见 `.gitignore`）：

1. **Embedding**：从 [HuggingFace Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) 下载 sentence-transformers 格式模型，放置到
   `Qwen3-Embedding-0.6B/qwen3-embedding-lora-merged/`（与 `config.LOCAL_EMBED_MODEL_PATH` 对应）。
   本地曾基于 PEFT/LoRA + MultipleNegativesRankingLoss 做领域微调后合并部署（r=8 / alpha=16 / 3 epochs，微调与合并脚本保存在本地工作区，未入库）。
2. **Rerank**（可选，缺失时自动降级）：Xinference 部署 `bge-reranker-v2-m3` 并监听 `localhost:9997`；无鉴权则 `.env` 中 `RERANK_API_KEY` 留空。

### 3. 摄入语料并启动

```bash
# 摄入 docs/ecommerce 语料：按一级子目录自动映射 bucket（product→rag-product、faq→rag-faq、policy→rag-policy、promotion→rag-promotion），
# faq 子目录自动走问答对切块；幂等跳过已入库文件；默认不经 MinIO
python eval/ingest_docs.py

# 启动 API（开发模式）
python api_server.py
# 生产：gunicorn api_server:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

> 注：`docker-compose.yml` 提供 MinIO / Etcd / Milvus Standalone / Redis / rag-app 的容器化编排示例；镜像内不含本地模型，Embedding/Rerank 请通过 `host.docker.internal` 指向宿主机的 Xinference 服务（环境变量已预留 `EMBED_URL` / `RERANK_URL`）。

### 4. 调用示例

```bash
# 单轮问答（售前）
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "星曜X1 Pro 的电池容量是多少？"}'

# 多轮问答（携带 conversation_id 触发 Redis 记忆与指代消解）
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "RedmiK80 支持快充吗？", "conversation_id": "session-001"}'
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "那它的价格是多少？", "conversation_id": "session-001"}'

# 订单查询（mock 订单库）
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "我的订单 YX20260815001 到哪了？"}'

# 转人工 / 投诉
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "我要投诉，给我转人工！"}'

# 清除会话历史
curl -X DELETE http://127.0.0.1:8000/api/v1/conversations/session-001
```

响应含 `answer`（回答）、`sources`（引用来源：片段/文档/bucket）、`intent`、`route`（rag/order/escalation/chitchat）、`needs_handoff`、`order_id` 等字段。

---

## 多轮记忆设计要点

- 存储：Redis List + JSON 消息，`RPUSH + LTRIM + EXPIRE` 经 pipeline 原子执行；默认保留最近 10 轮、单条截断 1000 字符、TTL 2 天滑动过期。
- 指代消解：改写提示词注入最近对话历史，要求补全代词/缺省对象，生成**脱离上下文可独立检索**的改写问句再并发召回；生成阶段同样注入历史理解指代。
- 韧性：Redis 不可用时读取返回空、写入静默跳过（无状态单轮）；连接失败进入 30s 冷却期避免黑洞端口拖慢请求；Redis 恢复后懒重连自动启用记忆。
- 测试：`test/test_memory_service.py` 覆盖正常 / 降级 / 恢复三态；`test/test_e2e_memory_api.py` 走真实 API 验证两轮对话的指代消解与落库；另有改写空结果防护测试（mock，不发真实请求）。

---

## 评测（eval/）

自建金标准评测工具链，支持任意场景金标准集：

```bash
# 一键评测：检索层 4 组消融（baseline / rewrite_only / rerank_only / full）+ 生成层 LLM-as-Judge
python eval/run_eval.py --golden eval/golden_set_ecommerce.json

# 仅检索层 / 冒烟
python eval/run_eval.py --golden eval/golden_set_ecommerce.json --skip-generation
python eval/run_eval.py --golden eval/golden_set_ecommerce.json --smoke
```

- 检索指标：Hit@K / Precision@K / FragRecall@K / MRR / nDCG@K（金标准判定：命中片段 = `file_name` 一致且含金标准关键词）。
- 生成指标：Faithfulness（忠实度，防幻觉）与 Answer Relevancy（相关性），由 DeepSeek 裁判按 1-5 打分。
- 脚本会自动跳过知识库中未入库文档对应的题目并提示；`--compare-base` 可对比 Embedding 微调前后效果。
- `eval/` 下的报告/结果 JSON 为运行产物（已 gitignore），脚本与金标准集入库。

历史评测报告（本地运行产物，未入库）中小说场景 17 题全量数据为：完整链路 Hit@1=0.82 / Hit@5=0.94 / MRR=0.85，BGE Rerank 使 Hit@1 由 0.24 提升至 0.82（+59pp）；LLM-as-Judge 生成 Faithfulness 5.0/5。

---

## 配置与密钥

所有密钥仅从环境变量 / `.env` 读取（见 `config/config.py` 与 `.env.example`），仓库内**无任何明文密钥**：

| 变量 | 必填 | 说明 |
|---|---|---|
| `DEEPSEEK_API_KEY` | ✅ | DeepSeek API Key |
| `RERANK_API_KEY` | 视部署 | Xinference Rerank 鉴权，格式 `Bearer xf-xxx`，无鉴权留空 |
| `TAVILY_API_KEY` | 否 | 联网搜索（客服场景 `WEB_SEARCH_ENABLED=False` 时无需） |
| `REDIS_HOST/PORT/DB/PASSWORD` | 否 | 多轮记忆；不可用时自动降级无状态 |

其他可调项（切片大小 350/60、TTL、会话轮数上限、重试次数、各意图召回/精排规格）集中在 `config/config.py`，均有中文注释。

---

## 测试

```bash
# 记忆模块三态（正常路径需本机 Redis；无 Redis 时仅降级用例可跑）
python test/test_memory_service.py

# 改写空结果防护（mock，不发真实请求）
python test/test_rewrite_guard.py

# 端到端（需 API 服务运行在 8000 且 Redis/DeepSeek 可用）
python test/test_e2e_memory_api.py
```

---

## 免责与合规说明

- **版权语料**：早期场景的金庸小说全文（`docs_archive/`、`raw_corpus/`）为个人学习研究用途，**未随本仓库分发**；仓库内仅保留自编的电商演示语料（`docs/ecommerce/`）。
- **模型权重**：`Qwen3-Embedding-0.6B/`（>4GB）未入库，请按上文指引自行下载。
- **mock 订单**：`data/mock_orders.json` 为演示用模拟数据，生产环境请将 `OrderService` 替换为真实订单中心 API。
