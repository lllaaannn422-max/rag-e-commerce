# test_order_service.py
"""OrderService 单元测试：正则抽取 / 订单查找 / LLM 抽取（mock，不发真实请求）"""
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.order_service import OrderService
from services.deepseek_service import DeepSeekService


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class FakeCompletions:
    def __init__(self, contents):
        self.contents = list(contents)
        self.calls = 0

    def create(self, **kwargs):
        idx = min(self.calls, len(self.contents) - 1)
        self.calls += 1
        return FakeResponse(self.contents[idx])


class OrderServiceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.svc = OrderService()  # 默认 data/mock_orders.json

    def test_01_extract_fast_path(self):
        self.assertEqual(self.svc.extract_order_id_fast("我的订单 YX20260815001 到哪了"), "YX20260815001")
        self.assertEqual(self.svc.extract_order_id_fast("帮我查一下订单号：YX20260802008"), "YX20260802008")
        self.assertIsNone(self.svc.extract_order_id_fast("你们发货要多久啊"))

    def test_02_lookup_known_order(self):
        order = self.svc.lookup("YX20260815001")
        self.assertIsNotNone(order)
        self.assertEqual(order["status"], "配送中")
        self.assertIn("shipping", order)
        self.assertTrue(order["shipping"]["timeline"], "配送中订单应有物流时间线")

    def test_03_lookup_case_insensitive(self):
        order = self.svc.lookup("yx20260815001")
        self.assertIsNotNone(order)
        self.assertEqual(order["order_id"], "YX20260815001")

    def test_04_lookup_unknown_order(self):
        self.assertIsNone(self.svc.lookup("NOPE00000001"))
        self.assertIsNone(self.svc.lookup(""))
        self.assertIsNone(self.svc.lookup(None))

    def test_05_build_context(self):
        order = self.svc.lookup("YX20260728003")
        ctx = self.svc.build_order_context(order)
        self.assertIn("退款中", ctx)
        self.assertIn("YX20260728003", ctx)


class ExtractOrderInfoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.svc = DeepSeekService()
        cls._orig_create = cls.svc.client.chat.completions.create

    @classmethod
    def tearDownClass(cls):
        cls.svc.client.chat.completions.create = cls._orig_create

    def test_01_valid_json_parsed(self):
        fake = FakeCompletions(['{"order_id": "YX20260815001", "confidence": 0.95, "note": "命中"}'])
        self.svc.client.chat.completions.create = fake.create
        info = self.svc.extract_order_info("帮我查订单 YX20260815001")
        self.assertEqual(info["order_id"], "YX20260815001")
        self.assertGreater(info["confidence"], 0.9)

    def test_02_empty_order_id(self):
        fake = FakeCompletions(['{"order_id": "", "confidence": 0.1, "note": "未提供"}'])
        self.svc.client.chat.completions.create = fake.create
        info = self.svc.extract_order_info("你们发货快吗")
        self.assertEqual(info["order_id"], "")

    def test_03_malformed_output_raises(self):
        # 非 JSON 输出 → ValueError（retry_on_exception 重试后仍失败则抛出）
        fake = FakeCompletions(["这不是JSON", "也不是JSON", "还不是JSON"])
        self.svc.client.chat.completions.create = fake.create
        with self.assertRaises(ValueError):
            self.svc.extract_order_info("查订单")


if __name__ == "__main__":
    unittest.main(verbosity=2)
