# test_fallback_intent.py
"""fallback_rule_based 电商客服关键词规则测试（无网络，校验检查顺序）"""
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.deepseek_service import DeepSeekService


class FallbackIntentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.svc = DeepSeekService()

    def _intent(self, query):
        return self.svc.fallback_rule_based(query)["intent"]

    def test_01_escalation_priority(self):
        self.assertEqual(self._intent("我要投诉，给我转人工"), "escalation")
        self.assertEqual(self._intent("你们态度太差了"), "escalation")

    def test_02_order_before_after_sales(self):
        # 「退货订单」同时含售后与订单关键词，应命中订单查询
        self.assertEqual(self._intent("我的退货订单到哪了"), "order_status")
        self.assertEqual(self._intent("物流到哪了"), "order_status")

    def test_03_after_sales(self):
        self.assertEqual(self._intent("我要退货"), "after_sales")
        self.assertEqual(self._intent("保修多久"), "after_sales")
        self.assertEqual(self._intent("运费谁出"), "after_sales")

    def test_04_promotion(self):
        self.assertEqual(self._intent("满减活动怎么参加"), "promotion")
        self.assertEqual(self._intent("优惠券能用吗"), "promotion")

    def test_05_comparison(self):
        self.assertEqual(self._intent("X1 Pro 和 X1 哪个好"), "comparison")

    def test_06_pre_sales(self):
        self.assertEqual(self._intent("手机电池多大"), "pre_sales")
        self.assertEqual(self._intent("多少钱"), "pre_sales")

    def test_07_chitchat(self):
        self.assertEqual(self._intent("你好"), "chitchat")
        self.assertEqual(self._intent("谢谢"), "chitchat")

    def test_08_other_fallback(self):
        self.assertEqual(self._intent("qwertyuiop"), "other")


if __name__ == "__main__":
    unittest.main(verbosity=2)
