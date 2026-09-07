# memory_service.py
import json
import time
from typing import List, Dict, Optional
import redis
import config.config as config
from utils.logger import logger


class MemoryService:
    """Redis 多轮对话记忆服务。

    可选能力：Redis 不可用时自动降级为无状态单轮（get_history 返回空、add_message 静默跳过），
    与项目现有的 rerank/web_search 降级哲学保持一致。
    redis-py 客户端自身线程安全（内部连接池），且 add_message 用 pipeline 保证
    RPUSH + LTRIM + EXPIRE 原子执行，可安全被 to_thread 多线程共享。
    """

    def __init__(self):
        self._client: Optional[redis.Redis] = None
        self._last_fail_ts: float = 0.0  # 最近一次连接失败时间戳，用于冷却期跳过重试
        self._get_client()  # 启动即探测，失败仅告警不影响服务

    # ---------- 连接管理（懒重连：Redis 恢复后自动重新启用记忆） ----------
    def _get_client(self) -> Optional[redis.Redis]:
        if self._client is not None:
            return self._client
        # 冷却期：连接失败后短时间内不再重试（黑洞端口单次连接可能耗时十几秒，避免拖慢每个请求）
        if time.time() - self._last_fail_ts < config.REDIS_RECONNECT_COOLDOWN:
            return None
        try:
            client = redis.Redis(
                host=config.REDIS_HOST,
                port=config.REDIS_PORT,
                db=config.REDIS_DB,
                password=config.REDIS_PASSWORD or None,
                decode_responses=True,
                socket_connect_timeout=config.REDIS_CONNECT_TIMEOUT,
                socket_timeout=config.REDIS_SOCKET_TIMEOUT,
            )
            client.ping()
            self._client = client
            logger.info(f"Redis 连接成功 ({config.REDIS_HOST}:{config.REDIS_PORT}, DB={config.REDIS_DB})，多轮对话记忆已启用")
        except Exception as e:
            self._client = None
            self._last_fail_ts = time.time()
            logger.warning(f"Redis 暂不可用（多轮记忆降级为无状态，{config.REDIS_RECONNECT_COOLDOWN}s 内不再重试）: {e}")
        return self._client

    def _key(self, conversation_id: str) -> str:
        return f"{config.REDIS_KEY_PREFIX}:{conversation_id}"

    # ---------- 对外接口（全部容错，Redis 故障时静默降级） ----------
    def add_message(self, conversation_id: str, role: str, content: str) -> None:
        """追加一条消息（user/assistant），自动截断、限长、刷新 TTL。失败仅告警不抛出。"""
        client = self._get_client()
        if client is None:
            return
        try:
            content = (content or "")[:config.REDIS_MAX_MESSAGE_CHARS]
            payload = json.dumps({"role": role, "content": content}, ensure_ascii=False)
            # pipeline 保证追加/裁剪/过期三者原子执行
            pipe = client.pipeline(transaction=True)
            pipe.rpush(self._key(conversation_id), payload)
            pipe.ltrim(self._key(conversation_id), -config.REDIS_MAX_HISTORY_MESSAGES, -1)
            pipe.expire(self._key(conversation_id), config.REDIS_HISTORY_TTL)
            pipe.execute()
        except Exception as e:
            logger.warning(f"保存对话历史失败（已降级为无状态），conversation_id={conversation_id}: {e}")

    def get_history(self, conversation_id: str) -> List[Dict[str, str]]:
        """读取会话历史（旧到新）。Redis 故障时返回 []，与单轮无状态行为一致。"""
        client = self._get_client()
        if client is None:
            return []
        try:
            items = client.lrange(self._key(conversation_id), 0, -1)
        except Exception as e:
            logger.warning(f"读取对话历史失败（已降级为无状态），conversation_id={conversation_id}: {e}")
            return []
        history = []
        for item in items:
            try:
                msg = json.loads(item)
                if msg.get("role") in ("user", "assistant") and msg.get("content"):
                    history.append({"role": msg["role"], "content": msg["content"]})
            except Exception:
                continue  # 跳过损坏条目
        return history

    def clear(self, conversation_id: str) -> bool:
        """删除会话全部历史。Redis 故障时返回 False（由上层降级处理）。"""
        client = self._get_client()
        if client is None:
            return False
        try:
            client.delete(self._key(conversation_id))
            return True
        except Exception as e:
            logger.warning(f"清除对话历史失败，conversation_id={conversation_id}: {e}")
            return False
