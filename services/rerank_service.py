import requests
from typing import List
import config.config as config
from utils.logger import logger
from utils.decorators import retry_on_exception

class RerankService:
    def __init__(self):
        self.session = requests.Session()
        headers = {"Content-Type": "application/json"}
        if config.RERANK_API_KEY:
            headers["Authorization"] = config.RERANK_API_KEY
        self.session.headers.update(headers)

    @retry_on_exception(max_retries=config.MAX_RETRIES, delay=config.RETRY_DELAY)
    def rerank(self, query: str, documents: List[str], top_n: int = 10, threshold: float = 0.7) -> List[int]:
        if not documents:
            return []

        # 从 config 获取模型名，若未定义则使用默认值
        model_name = getattr(config, "RERANK_MODEL_NAME", "bge-reranker-v2-m3")

        payload = {
            "model": model_name,
            "query": query,
            "documents": documents
        }
        if top_n:
            payload["top_n"] = top_n   # Xinference Rerank API 使用 top_n

        response = self.session.post(config.RERANK_URL, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        # Xinference 返回格式：{"results": [{"index": 0, "relevance_score": 0.9}, ...]}
        results = data.get("results", data.get("data", []))
        sorted_results = sorted(results, key=lambda x: x.get("relevance_score", x.get("score", 0)), reverse=True)
        filtered = [item for item in sorted_results if item.get("relevance_score", item.get("score", 0)) > threshold]
        top_items = filtered[:top_n]
        return [item["index"] for item in top_items]