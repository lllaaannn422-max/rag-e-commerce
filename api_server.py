# api_server_v2.py
'''
电商客服 RAG：意图路由（售前/售后/活动/对比/订单/转人工/闲聊）+ 向量检索 + rerank + MinIO + Milvus
订单查询走 mock 订单库（OrderService），投诉/转人工与闲聊走专用话术分支，其余意图走本地知识库检索。
'''
import os
import sys
# 将项目根目录添加到 python 搜索路径中
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import asyncio
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from services.minio_service import MinioService
from services.milvus_service import MilvusService
from pipeline.document_processor import DocumentProcessor
# 导入项目配置
import config.config as config
print("config file:", config.__file__)  # 调试用，可删除
from utils.logger import logger
from pipeline.rag_query_engine import RAGQueryEngine
from pipeline.document_processor import DocumentProcessor
from services.order_service import OrderService

# 全局引擎单例句柄
rag_engine: Optional[RAGQueryEngine] = None
doc_processor: Optional[DocumentProcessor] = None
order_service: Optional[OrderService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global rag_engine, doc_processor, order_service
    logger.info("正在初始化 RAG 检索引擎与文档处理器...")
    rag_engine = RAGQueryEngine()
    doc_processor = DocumentProcessor()
    order_service = OrderService()
    logger.info("电商客服 RAG 服务初始化完成，随时准备接受请求！")
    yield
    logger.info("服务关闭中，释放资源...")


app = FastAPI(
    title="RAG Knowledge Base API",
    description="精简版：FastAPI + Milvus + LocalEmbedding + Rerank + DeepSeek",
    version="2.0.0",
    lifespan=lifespan
)

# ===== Schemas (适配 Pydantic V2 规范) =====
class QueryRequest(BaseModel):
    query: str = Field(..., description="用户输入的查询问题", json_schema_extra={"example": "对比Python和Java在AI开发中的优缺点。"})
    top_k: Optional[int] = Field(None, description="可选：自定义召回数量", json_schema_extra={"example": 20})
    conversation_id: Optional[str] = Field(None, description="可选：多轮对话会话ID，不传则为无状态单轮问答")


class QueryResponse(BaseModel):
    query: str
    intent: str
    intent_description: str
    answer: str
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="引用的参考文档来源")
    conversation_id: Optional[str] = Field(None, description="本次请求使用的会话ID")
    route: Optional[str] = Field(None, description="处理路径: rag / order / escalation / chitchat")
    needs_handoff: bool = Field(False, description="是否需要转人工")
    order_id: Optional[str] = Field(None, description="命中的订单号（仅 order 路径）")


class IngestRequest(BaseModel):
    file_path: str = Field(..., description="服务器本地或挂载目录下的文档绝对路径", json_schema_extra={"example": "D:/zuoye/rag_project/docs/test.pdf"})
    bucket_name: str = Field("rag-product", description="目标知识库 Bucket 名称（rag-product/rag-faq/rag-policy/rag-promotion）", json_schema_extra={"example": "rag-product"})


class StandardResponse(BaseModel):
    code: int = 200
    message: str


# ===== Endpoints =====
@app.get("/health", response_model=StandardResponse, tags=["Health"])
async def health_check():
    return StandardResponse(code=200, message="Service Healthy")


@app.post("/api/v1/chat", response_model=QueryResponse, tags=["RAG Engine"])
async def rag_chat(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Query 不能为空")
    try:
        logger.info(f"收到 Query 请求: {request.query} | top_k: {request.top_k} | conversation_id: {request.conversation_id}")
        response_data = await asyncio.to_thread(_process_query_pipeline, request.query, request.top_k, request.conversation_id)
        return response_data
    except Exception as e:
        logger.error(f"处理 Query 请求异常: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"内部处理失败: {str(e)}")


@app.delete("/api/v1/conversations/{conversation_id}", response_model=StandardResponse, tags=["Conversation Memory"])
async def clear_conversation(conversation_id: str):
    """清除指定会话的多轮对话历史。Redis 不可用时返回 200 并提示（会话本就处于无状态模式）。"""
    # Redis 为阻塞调用，必须放入线程池，避免 Redis 故障时卡死事件循环
    cleared = await asyncio.to_thread(rag_engine.memory_service.clear, conversation_id)
    if cleared:
        logger.info(f"对话历史已清除: {conversation_id}")
        return StandardResponse(code=200, message=f"对话历史已清除: {conversation_id}")
    logger.warning(f"Redis 不可用或清除失败（无状态模式），无需清除: {conversation_id}")
    return StandardResponse(code=200, message=f"Redis 不可用或清除失败（当前为无状态模式）: {conversation_id}")


@app.post("/api/v1/documents/ingest", response_model=StandardResponse, tags=["Document Processing"])
async def ingest_document(request: IngestRequest):
    try:
        await asyncio.to_thread(doc_processor.process_and_index_file, request.file_path, request.bucket_name)
        return StandardResponse(code=200, message=f"文件 [{request.file_path}] 处理完成并成功构建索引！")
    except Exception as e:
        logger.error(f"文档摄入失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


def _process_query_pipeline(query: str, custom_top_k: Optional[int] = None,
                            conversation_id: Optional[str] = None) -> QueryResponse:
    logger.info(f"2------------开始处理 Query: {query} | top_k: {custom_top_k} | conversation_id: {conversation_id}")
    # 多轮对话历史：Redis 不可用时 get_history 返回 []，与单轮无状态行为一致
    history = rag_engine.memory_service.get_history(conversation_id) if conversation_id else []
    intent_info = rag_engine.deepseek_service.recognize_intent(query)
    intent = intent_info.get("intent", "other")
    intent_desc = intent_info.get("description", "")
    logger.info(f"【并发线程处理】Query: {query} | 意图: {intent}")

    route, needs_handoff, order_id = "rag", False, None
    answer, sources = None, []

    if intent == "order_status":
        # 订单查询分支：正则快路径 → LLM 抽取兜底 → mock 订单库查找（不检索知识库）
        route = "order"
        oid = order_service.extract_order_id_fast(query)
        if not oid:
            try:
                oid = (rag_engine.deepseek_service.extract_order_info(query).get("order_id") or "").strip() or None
            except Exception as e:
                logger.warning(f"LLM 订单号抽取失败，按未提供订单号处理: {e}")
        order = order_service.lookup(oid) if oid else None
        if order:
            order_id = order["order_id"]
            answer = rag_engine.deepseek_service.generate_order_answer(query, order, history=history)
        else:
            # 未知/无订单号：固定模板回复，不调 LLM，杜绝编造物流信息
            answer = (f"抱歉，我没有查到您提供的订单号「{oid or ''}」对应的订单信息。"
                      "请您核对订单号是否正确；也可以在「我的订单」页面查看物流动态，"
                      "或回复“转人工”联系人工客服为您查询。")
    elif intent == "escalation":
        # 投诉/转人工分支：共情安抚话术，标记需要人工介入
        route, needs_handoff = "escalation", True
        answer = rag_engine.deepseek_service.generate_escalation_reply(query, history=history)
    elif intent == "chitchat":
        # 闲聊分支：简短问候并引导业务话题
        route = "chitchat"
        answer = rag_engine.deepseek_service.generate_chitchat_reply(query, history=history)
    else:
        # RAG 检索分支（pre_sales / after_sales / promotion / comparison / clarification / other）
        target_buckets = config.INTENT_BUCKET_MAP.get(intent, config.INTENT_BUCKET_MAP["other"])
        if len(target_buckets) == 1:
            filter_expr = f'source_bucket == "{target_buckets[0]}"'
        else:
            bucket_str = ', '.join([f'"{b}"' for b in target_buckets])
            filter_expr = f'source_bucket in [{bucket_str}]'

        if custom_top_k:
            recall_k, rerank_top_n = custom_top_k * 4, custom_top_k
        else:
            recall_k, rerank_top_n = config.INTENT_RECALL_CONFIG.get(intent, (30, 10))

        final_docs = rag_engine.search_with_rewrite(
            query, filter_expr=filter_expr, intent=intent,
            recall_k=recall_k, rerank_top_n=rerank_top_n, history=history
        )
        if not final_docs:
            answer = ("抱歉，我没有在知识库中找到与您问题相关的信息。"
                      "您可以换个问法再试试，或回复“转人工”联系人工客服（工作时间 9:00-21:00）。")
        else:
            answer = rag_engine.deepseek_service.generate_answer(query, final_docs, history=history)
            sources = [
                {
                    "text": (doc.get("chunk_text") or "")[:150] + "...",
                    "doc_name": doc.get("doc_name") or doc.get("file_name") or "",
                    "bucket": doc.get("source_bucket") or doc.get("bucket") or "",
                    "url": doc.get("url") or ""   # 本地文档为空串，联网结果为新闻原文链接
                }
                for doc in final_docs
            ]
    if conversation_id:
        # 仅记录成功完成的轮次；Redis 不可用时 add_message 内部静默降级
        rag_engine.memory_service.add_message(conversation_id, "user", query)
        rag_engine.memory_service.add_message(conversation_id, "assistant", answer)
    return QueryResponse(
        query=query,
        intent=intent,
        intent_description=intent_desc,
        answer=answer,
        sources=sources,
        conversation_id=conversation_id,
        route=route,
        needs_handoff=needs_handoff,
        order_id=order_id
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
