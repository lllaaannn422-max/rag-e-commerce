# test_rewrite_guard.py
"""rewrite_queries 空结果防护测试（mock DeepSeek 客户端，不发真实请求）

验证：
  1. LLM 返回空内容 → 抛 ValueError 触发 retry_on_exception 重试，重试后成功返回
  2. 连续空内容（重试耗尽）→ 最终抛出 ValueError（由引擎侧捕获降级为原问句）

注意：重试间隔为 config.RETRY_DELAY（装饰器绑定在 import 时），故用例耗时约 5~10s。
运行：python test/test_rewrite_guard.py
"""
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    """按顺序返回 contents；超出后复用最后一个"""

    def __init__(self, contents):
        self.contents = list(contents)
        self.calls = 0

    def create(self, **kwargs):
        idx = min(self.calls, len(self.contents) - 1)
        self.calls += 1
        return FakeResponse(self.contents[idx])


class RewriteGuardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.svc = DeepSeekService()
        cls._orig_create = cls.svc.client.chat.completions.create

    @classmethod
    def tearDownClass(cls):
        cls.svc.client.chat.completions.create = cls._orig_create

    def test_01_empty_then_valid_retries_and_succeeds(self):
        """首次返回空 → 抛异常触发重试 → 第二次成功"""
        fake = FakeCompletions(["", "令狐冲的师父是谁？"])
        self.svc.client.chat.completions.create = fake.create
        rewrites = self.svc.rewrite_queries("他师父是谁？", num_rewrites=2)
        self.assertEqual(fake.calls, 2, "空结果应触发重试（共调用 2 次）")
        self.assertIn("令狐冲的师父是谁？", rewrites)
        self.assertIn("他师父是谁？", rewrites)  # 原句仍在列表首位

    def test_02_always_empty_raises_after_retries(self):
        """连续空内容 → 重试耗尽后抛出 ValueError（供引擎侧捕获降级）"""
        fake = FakeCompletions(["", "", ""])
        self.svc.client.chat.completions.create = fake.create
        with self.assertRaises(ValueError):
            self.svc.rewrite_queries("他师父是谁？")
        self.assertEqual(fake.calls, 3, "应重试 MAX_RETRIES=3 次后抛出")


if __name__ == "__main__":
    unittest.main(verbosity=2)