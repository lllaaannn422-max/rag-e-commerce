# order_service.py
import json
import re
from typing import Dict, List, Optional
import config.config as config
from utils.logger import logger


class OrderService:
    """模拟订单查询服务：从本地 JSON 懒加载 mock 订单，支持正则快路径抽取订单号。

    真实生产环境中此处应替换为订单中心 API/数据库查询。
    """

    ORDER_RE = re.compile(r"(?:订单号?|单号)\s*[:：]?\s*([A-Za-z0-9]{6,20})")

    def __init__(self, orders_path: Optional[str] = None):
        self.orders_path = orders_path or config.MOCK_ORDERS_PATH
        self._orders: Optional[List[Dict]] = None

    def _load(self) -> List[Dict]:
        if self._orders is None:
            try:
                with open(self.orders_path, "r", encoding="utf-8") as f:
                    self._orders = json.load(f).get("orders", [])
                logger.info(f"订单数据加载完成: {len(self._orders)} 条（{self.orders_path}）")
            except Exception as e:
                logger.error(f"订单数据加载失败，按空订单库处理: {e}")
                self._orders = []
        return self._orders

    def lookup(self, order_id: str) -> Optional[Dict]:
        """按订单号精确查询（大小写不敏感）；不存在返回 None"""
        if not order_id:
            return None
        oid = order_id.strip().upper()
        for order in self._load():
            if order.get("order_id", "").upper() == oid:
                return order
        return None

    def extract_order_id_fast(self, query: str) -> Optional[str]:
        """正则快路径：命中「订单号/单号 + 编号」模式直接返回，未命中返回 None（由 LLM 兜底抽取）"""
        m = self.ORDER_RE.search(query or "")
        return m.group(1) if m else None

    def build_order_context(self, order: Dict) -> str:
        """将订单 dict 压缩为提示词上下文（中文键名便于模型理解）"""
        return json.dumps(order, ensure_ascii=False, indent=2)
