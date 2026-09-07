# test_memory_service.py
"""MemoryService（Redis 多轮对话记忆）单元测试

覆盖：
  正常路径：增/查往返、顺序、单条截断、条数上限、TTL、坏数据跳过、清空、会话隔离
  降级路径：Redis 不可用时静默无状态化 + 冷却期 + 恢复后懒重连

运行方式：
    python test/test_memory_service.py            # 直接运行
    pytest test/test_memory_service.py -v         # pytest 兼容
前置条件：本机 localhost:6379 有 Redis（正常路径用例）；无 Redis 时仅降级用例可跑。
"""
import os
import sys
import time
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import redis
import config.config as config
from services.memory_service import MemoryService

DEAD_PORT = 6399  # 本机无服务监听的端口，用于降级测试


def _now_cid(prefix: str) -> str:
    return f"{prefix}-{int(time.time() * 1000)}"


class MemoryNormalTest(unittest.TestCase):
    """正常路径：真实 Redis (localhost:6379, DB=2)"""

    @classmethod
    def setUpClass(cls):
        cls.svc = MemoryService()
        if cls.svc._client is None:
            raise unittest.SkipTest("前置条件不满足：localhost:6379 Redis 不可用，跳过正常路径用例")
        cls.cid = _now_cid("memtest-normal")

    def tearDown(self):
        self.svc.clear(self.cid)

    def test_01_add_get_roundtrip_and_order(self):
        """增/查往返：内容与角色正确，顺序为旧→新"""
        self.svc.add_message(self.cid, "user", "令狐冲是谁？")
        self.svc.add_message(self.cid, "assistant", "令狐冲是《笑傲江湖》主角。")
        self.svc.add_message(self.cid, "user", "他师父是谁？")
        history = self.svc.get_history(self.cid)
        self.assertEqual(
            history,
            [
                {"role": "user", "content": "令狐冲是谁？"},
                {"role": "assistant", "content": "令狐冲是《笑傲江湖》主角。"},
                {"role": "user", "content": "他师父是谁？"},
            ],
        )

    def test_02_message_char_truncation(self):
        """单条消息超长时截断到 REDIS_MAX_MESSAGE_CHARS"""
        long_msg = "长" * (config.REDIS_MAX_MESSAGE_CHARS + 500)
        self.svc.add_message(self.cid, "user", long_msg)
        history = self.svc.get_history(self.cid)
        self.assertEqual(len(history), 1)
        self.assertEqual(len(history[0]["content"]), config.REDIS_MAX_MESSAGE_CHARS)

    def test_03_history_count_cap(self):
        """条数上限：超出 REDIS_MAX_HISTORY_MESSAGES 时保留最新 N 条"""
        total = config.REDIS_MAX_HISTORY_MESSAGES + 5
        for i in range(total):
            self.svc.add_message(self.cid, "user" if i % 2 == 0 else "assistant", f"消息{i:03d}")
        history = self.svc.get_history(self.cid)
        self.assertEqual(len(history), config.REDIS_MAX_HISTORY_MESSAGES)
        # 最早 5 条被裁剪，第一条应为 消息005
        self.assertEqual(history[0]["content"], f"消息{total - config.REDIS_MAX_HISTORY_MESSAGES:03d}")
        self.assertEqual(history[-1]["content"], f"消息{total - 1:03d}")

    def test_04_ttl(self):
        """写入后 key 设置了 TTL（滑动过期，约 REDIS_HISTORY_TTL）"""
        self.svc.add_message(self.cid, "user", "测试TTL")
        ttl = self.svc._client.ttl(self.svc._key(self.cid))
        self.assertGreater(ttl, 0)
        self.assertLessEqual(ttl, config.REDIS_HISTORY_TTL)

    def test_05_corrupt_entries_skipped(self):
        """非法 JSON / 非法角色条目被跳过，不抛异常"""
        key = self.svc._key(self.cid)
        raw = redis.Redis(host=config.REDIS_HOST, port=config.REDIS_PORT,
                          db=config.REDIS_DB, decode_responses=True)
        raw.rpush(key, "not-a-json{{", '{"role": "system", "content": "非法角色"}', '{"content": "缺角色"}')
        self.svc.add_message(self.cid, "user", "合法消息")
        history = self.svc.get_history(self.cid)
        self.assertEqual(history, [{"role": "user", "content": "合法消息"}])

    def test_06_clear_and_isolation(self):
        """clear 幂等；不同 conversation_id 互不影响"""
        other = _now_cid("memtest-other")
        self.svc.add_message(self.cid, "user", "A会话")
        self.svc.add_message(other, "user", "B会话")
        self.assertTrue(self.svc.clear(self.cid))
        self.assertTrue(self.svc.clear(self.cid))  # 幂等：再清一次仍返回 True
        self.assertEqual(self.svc.get_history(self.cid), [])
        self.assertEqual(self.svc.get_history(other), [{"role": "user", "content": "B会话"}])
        self.svc.clear(other)


class MemoryDegradedTest(unittest.TestCase):
    """降级路径：Redis 不可用（死端口）时静默无状态化 + 冷却期"""

    @classmethod
    def setUpClass(cls):
        cls._saved = {k: getattr(config, k) for k in
                      ("REDIS_PORT", "REDIS_CONNECT_TIMEOUT", "REDIS_SOCKET_TIMEOUT", "REDIS_RECONNECT_COOLDOWN")}
        config.REDIS_PORT = DEAD_PORT
        config.REDIS_CONNECT_TIMEOUT = 0.2
        config.REDIS_SOCKET_TIMEOUT = 0.2
        config.REDIS_RECONNECT_COOLDOWN = 1.5  # 缩短冷却便于测试
        cls.svc = MemoryService()  # 构造即探测失败
        cls.cid = _now_cid("memtest-dead")

    @classmethod
    def tearDownClass(cls):
        for k, v in cls._saved.items():
            setattr(config, k, v)

    def test_01_get_history_empty(self):
        self.assertEqual(self.svc.get_history(self.cid), [])

    def test_02_add_message_silent(self):
        self.svc.add_message(self.cid, "user", "写入不应抛异常")

    def test_03_clear_false(self):
        self.assertFalse(self.svc.clear(self.cid))

    def test_04_cooldown_skips_reconnect(self):
        """冷却期内不再重试（调用瞬时返回）；冷却过后重新探测并更新失败时间戳"""
        fail_ts_1 = self.svc._last_fail_ts
        self.assertGreater(fail_ts_1, 0)
        t0 = time.time()
        self.svc.get_history(self.cid)  # 冷却期内
        self.assertLess(time.time() - t0, 0.1, "冷却期内不应发起连接重试")
        self.assertEqual(self.svc._last_fail_ts, fail_ts_1)
        time.sleep(config.REDIS_RECONNECT_COOLDOWN + 0.3)
        self.svc.get_history(self.cid)  # 冷却过期，重新探测（仍失败）
        self.assertGreater(self.svc._last_fail_ts, fail_ts_1, "冷却过期后应重新探测并更新失败时间戳")


class MemoryReconnectTest(unittest.TestCase):
    """Redis 恢复后懒重连：自动重新启用多轮记忆"""

    @classmethod
    def setUpClass(cls):
        cls._saved = {k: getattr(config, k) for k in
                      ("REDIS_PORT", "REDIS_CONNECT_TIMEOUT", "REDIS_SOCKET_TIMEOUT", "REDIS_RECONNECT_COOLDOWN")}
        config.REDIS_PORT = DEAD_PORT
        config.REDIS_CONNECT_TIMEOUT = 0.2
        config.REDIS_SOCKET_TIMEOUT = 0.2
        config.REDIS_RECONNECT_COOLDOWN = 1.0
        cls.svc = MemoryService()
        cls.cid = _now_cid("memtest-reconnect")

    @classmethod
    def tearDownClass(cls):
        for k, v in cls._saved.items():
            setattr(config, k, v)
        try:
            cls.svc.clear(cls.cid)
        except Exception:
            pass

    def test_lazy_reconnect_after_redis_back(self):
        # 1. Redis 不可用：写入静默丢弃
        self.svc.add_message(self.cid, "user", "恢复前写入")
        self.assertEqual(self.svc.get_history(self.cid), [])
        # 2. 等待冷却过期，Redis 恢复
        time.sleep(config.REDIS_RECONNECT_COOLDOWN + 0.3)
        config.REDIS_PORT = self._saved["REDIS_PORT"]  # 模拟 Redis 恢复上线
        self.svc.add_message(self.cid, "user", "恢复后写入")
        history = self.svc.get_history(self.cid)
        self.assertIsNotNone(self.svc._client)
        # 3. 恢复后的消息可读（恢复前的消息本就未落库，符合"降级期间丢轮次"的设计）
        self.assertEqual(history, [{"role": "user", "content": "恢复后写入"}])


if __name__ == "__main__":
    unittest.main(verbosity=2)