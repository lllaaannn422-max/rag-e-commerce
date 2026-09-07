# config.py
import os
from pathlib import Path

# 从项目根目录 .env 加载本地密钥（.env 已被 .gitignore 排除，不随仓库分发）
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Windows 下 tokenizers（Rust/rayon 多线程）与 torch/Milvus 原生库混跑会偶发段错误
# （复现：Milvus search 后紧跟批量 encode 崩溃，~1/3 概率）；关闭 tokenizer 并行后稳定
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ===== Milvus（Milvus Lite 本地文件，随项目位置自动推导） =====
MILVUS_DB_PATH = str(Path(__file__).resolve().parent.parent / "data" / "milvus.db")
COLLECTION_NAME = "knowledge_base"

# ===== MinIO（本机 D:\MinIO 服务，默认账号 minioadmin/minioadmin） =====
MINIO_ENDPOINT = "localhost:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"

# ===== Embedding 在线测试test=====
# EMBED_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
# EMBED_DIM = 512  #测试维度降低，仅用于功能测试
# BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索："


# ===== Embedding 生产环境配置：当前项目文件夹内本地部署的 Qwen3-Embedding-0.6B =====
EMBED_MODEL_NAME = "Qwen3-Embedding-0.6B"
EMBED_DIM = 1024  # Qwen3-Embedding-0.6B 的输出维度为 1024

# 本地 sentence-transformers 格式模型路径（项目文件夹内，LoRA 合并后的最终模型）
LOCAL_EMBED_MODEL_PATH = str(
    Path(__file__).resolve().parent.parent / "Qwen3-Embedding-0.6B" / "qwen3-embedding-lora-merged"
)

# 已改为本地加载，不再通过 Xinference 调用 Embedding（如需切回远程，取消下面注释）
# EMBED_URL = "http://localhost:9997/v1/embeddings"


# ===== Chunking =====
CHUNK_SIZE = 350
CHUNK_OVERLAP = 60

# ===== Documents（项目内 docs 文件夹） =====
DOC_ROOT_DIR = str(Path(__file__).resolve().parent.parent / "docs")
SUPPORT_SUFFIX = {".pdf", ".txt", ".md"}

# ===== Intent routing（电商客服意图 → 检索 bucket） =====
INTENT_BUCKET_MAP = {
    "pre_sales":    ["rag-product", "rag-faq"],
    "after_sales":  ["rag-policy", "rag-faq"],
    "promotion":    ["rag-promotion"],
    "comparison":   ["rag-product", "rag-promotion"],
    "order_status": [],          # 走订单查询分支（mock 订单库），不检索知识库
    "escalation":   [],          # 走转人工安抚分支，不检索知识库
    "chitchat":     [],          # 走闲聊分支，不检索知识库
    "clarification": ["rag-product", "rag-faq", "rag-policy", "rag-promotion"],
    "other":        ["rag-product", "rag-faq", "rag-policy", "rag-promotion"],
}
TIME_DECAY_INTENTS = []   # 客服场景无时效衰减意图（新闻链路已关闭）

# ===== Intent prompt（电商客服意图分类） =====
INTENT_PROMPT_TEMPLATE = """你是电商客服系统的查询意图分类器。请根据以下判定规则，将用户消息归入唯一的类别。

    【判定规则】
    1. pre_sales：咨询商品参数、配置、价格、库存、功能等售前问题（尚未购买）。
    2. after_sales：咨询退换货、保修、物流配送、运费、发票、维修等售后政策问题。
    3. promotion：咨询优惠活动、满减、折扣券、赠品、活动时间等营销规则。
    4. comparison：明确要求对比两个或多个商品、活动或政策的差异、优劣。
    5. order_status：询问自己订单的状态、物流到哪了、什么时候发货/到货，或提供订单号查询。
    6. escalation：用户表达投诉、强烈不满、要求人工/电话客服、要求赔偿或升级处理。
    7. chitchat：问候、寒暄、闲聊、自我介绍等与业务无关的对话。
    8. clarification：问题指代不清、缺少关键信息（如商品型号），或过于宽泛。
    9. other：不属于以上任何一类。

    【参考示例】
    问题：星曜X1 Pro 的电池容量是多少？
    答案：{{"intent": "pre_sales", "confidence": 0.96, "description": "询问商品参数"}}

    问题：7天无理由退货需要满足什么条件？
    答案：{{"intent": "after_sales", "confidence": 0.95, "description": "询问退换货政策"}}

    问题：开学季活动满3000减300可以和店铺券叠加吗？
    答案：{{"intent": "promotion", "confidence": 0.94, "description": "询问促销规则"}}

    问题：星曜X1 Pro 和 X1 的屏幕有什么区别？
    答案：{{"intent": "comparison", "confidence": 0.97, "description": "对比两款商品"}}

    问题：我的订单 YX20260815001 到哪了？
    答案：{{"intent": "order_status", "confidence": 0.98, "description": "查询订单物流"}}

    问题：你们太差了，我要投诉，给我转人工！
    答案：{{"intent": "escalation", "confidence": 0.97, "description": "强烈不满要求人工"}}

    问题：你好呀
    答案：{{"intent": "chitchat", "confidence": 0.98, "description": "问候寒暄"}}

    问题：这个好用吗？
    答案：{{"intent": "clarification", "confidence": 0.88, "description": "缺少商品指代"}}

    【现在请分类】
    问题：{query}
    答案："""

# ===== 客服意图召回/精排规格（recall_k, rerank_top_n） =====
INTENT_RECALL_CONFIG = {
    "pre_sales":      (30, 10),
    "after_sales":    (30, 10),
    "promotion":      (20, 5),
    "comparison":     (80, 15),
    "clarification":  (30, 10),
    "other":          (30, 10),
}

# ===== 电商文档子目录 → bucket 映射（ingest_docs.py 与 API 摄入共用） =====
ECOMMERCE_SUBDIR_BUCKET_MAP = {
    "product":   "rag-product",
    "faq":       "rag-faq",
    "policy":    "rag-policy",
    "promotion": "rag-promotion",
}
ECOMMERCE_ROOT_DIR = str(Path(__file__).resolve().parent.parent / "docs" / "ecommerce")

# ===== 模拟订单数据（order_status 意图查询） =====
MOCK_ORDERS_PATH = str(Path(__file__).resolve().parent.parent / "data" / "mock_orders.json")




# ===== Rerank：Xinference 本地部署的 BGE Reranker（密钥从环境变量读取） =====
RERANK_URL = "http://localhost:9997/v1/rerank"
RERANK_API_KEY = os.getenv("RERANK_API_KEY", "")  # 格式如 "Bearer xf-xxxx"；无鉴权可留空
RERANK_MODEL_NAME = "bge-reranker-v2-m3"

# ===== DeepSeek API（密钥从环境变量读取，必填） =====
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

# ===== Web Search（Tavily 联网搜索；客服场景默认关闭） =====
WEB_SEARCH_ENABLED = False
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")  # 留空则跳过联网直接走本地检索
TAVILY_SEARCH_URL = "https://api.tavily.com/search"
WEB_SEARCH_MAX_RESULTS = 5        # 注意：Tavily basic 模式上限为 5，>5 需切 advanced
WEB_SEARCH_TOPIC = "news"         # Tavily topic 参数：news 偏向新闻文章（带发布日期），general 为通用

# ===== Redis（多轮对话记忆，可选依赖：Redis 不可用时自动降级为无状态单轮） =====
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "2"))        # 独立 DB，避免与本机其他项目的 Redis 数据冲突
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")  # 本地开发默认无密码
REDIS_KEY_PREFIX = "rag:chat"                     # key 格式: rag:chat:{conversation_id}
REDIS_HISTORY_TTL = int(os.getenv("REDIS_HISTORY_TTL", str(2 * 24 * 3600)))   # 会话保留 2 天（滑动过期）
REDIS_MAX_HISTORY_MESSAGES = int(os.getenv("REDIS_MAX_HISTORY_MESSAGES", "20"))  # 最多保留 10 轮（user+assistant 各一条）
REDIS_MAX_MESSAGE_CHARS = 1000                    # 单条消息入库前截断长度（防止超长回答撑爆改写提示词）
REDIS_CONNECT_TIMEOUT = 0.5   # 秒，建连超时（启动探测/懒重连）
REDIS_SOCKET_TIMEOUT = 2.0    # 秒，单条命令超时
REDIS_RECONNECT_COOLDOWN = 30 # 秒，连接失败后的冷却期：期间不再重试，避免黑洞端口拖慢每个请求（实测本机连接空端口约 15s）

# ===== Retry settings =====
MAX_RETRIES = 3
RETRY_DELAY = 1  # 秒
WHOOSH_MAX_RETRIES = 5  # Whoosh 补偿重试最大次数
