# services/web_search_service.py
import requests
from typing import List, Dict, Any, Optional
import config.config as config
from utils.logger import logger
from utils.decorators import retry_on_exception


class WebSearchService:
    """Tavily 联网搜索服务（news 意图触发），结果结构与本地向量检索候选一致"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    # 仅网络类错误重试；key 未配置等配置类错误直接抛出（ValueError 不在 exceptions 内，不重试）
    @retry_on_exception(max_retries=config.MAX_RETRIES, delay=config.RETRY_DELAY,
                        exceptions=(requests.exceptions.RequestException,))
    def search(self, query: str, max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        调用 Tavily Search API 执行联网搜索。

        :param query: 搜索语句
        :param max_results: 返回结果数，钳制在 basic 模式上限 5 以内
        :return: 候选文档列表 [{"chunk_text", "file_name", "source_bucket", "url", "publish_time", "total_score"}, ...]
        """
        if not config.TAVILY_API_KEY or "tvly-" not in config.TAVILY_API_KEY:
            logger.warning("TAVILY_API_KEY 未配置，跳过联网搜索（请在 config.py 中填入 key）")
            raise ValueError("TAVILY_API_KEY 未配置")

        # Tavily basic 模式 max_results 上限为 5
        result_num = min(max_results or config.WEB_SEARCH_MAX_RESULTS, 5)

        payload = {
            "api_key": config.TAVILY_API_KEY,
            "query": query,
            "search_depth": "basic",
            "max_results": result_num,
            "topic": getattr(config, "WEB_SEARCH_TOPIC", "news"),   # news 偏向新闻文章，结果带发布日期
        }
        response = self.session.post(config.TAVILY_SEARCH_URL, json=payload, timeout=(5, 30))
        response.raise_for_status()
        results = response.json().get("results", [])

        docs = []
        for r in results:
            content = (r.get("content") or "").strip()
            if not content:
                continue
            docs.append({
                "chunk_text": content,
                "file_name": r.get("title", ""),
                "source_bucket": "web_news",
                "url": r.get("url", ""),
                "publish_time": r.get("published_date", ""),
                "total_score": r.get("score", 0.0) or 0.0,
            })
        return docs


if __name__ == "__main__":
    service = WebSearchService()
    items = service.search("今天有什么重要的科技新闻", 3)
    print(f"获取 {len(items)} 条结果")
    for it in items:
        print("-", it["file_name"], "|", it["url"], "| score:", it["total_score"])
