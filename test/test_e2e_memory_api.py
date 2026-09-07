# test_e2e_memory_api.py
"""记忆模块端到端测试（通过运行中的 API 服务，端口 8000）

流程：
  1. 清理测试会话（含此前误写入的会话）
  2. 第一轮「星曜X1 Pro 的屏幕是多大尺寸的？」→ 断言 200 + Redis 落库 2 条
  3. 第二轮「它的电池容量是多少？」→ 断言 200 + Redis 落库 4 条
     + 从日志抓取改写列表，判定指代消解是否成功
  4. DELETE 会话 → 断言 Redis key 已删除

前置条件：API 服务已启动（uvicorn api_server:app --port 8000），Redis/DeepSeek 可用。
运行：python test/test_e2e_memory_api.py
"""
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import redis
import config.config as config

API = "http://127.0.0.1:8000"
LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "rag_system.log")

results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" | {detail}" if detail else ""))


def main():
    r = redis.Redis(host=config.REDIS_HOST, port=config.REDIS_PORT, db=config.REDIS_DB, decode_responses=True)
    key = lambda cid: f"{config.REDIS_KEY_PREFIX}:{cid}"

    # 0. 清理历史残留会话
    for stale in ("memtest-e2e-1788070384",):
        try:
            requests.delete(f"{API}/api/v1/conversations/{stale}", timeout=10)
            r.delete(key(stale))
        except Exception:
            pass

    cid = f"memtest-e2e-{int(time.time())}"
    print(f"conversation_id: {cid}")

    # 1. 第一轮
    t0 = time.time()
    resp = requests.post(f"{API}/api/v1/chat",
                         json={"query": "星曜X1 Pro 的屏幕是多大尺寸的？", "conversation_id": cid}, timeout=200)
    dt1 = time.time() - t0
    ok = resp.status_code == 200
    body = resp.json() if ok else resp.text
    check("round1 HTTP 200", ok, f"{dt1:.0f}s")
    if ok:
        print(f"  intent={body.get('intent')} | answer[:120]={body.get('answer','')[:120]}")
        check("round1 有回答", bool(body.get("answer")))
    else:
        print(f"  response: {str(body)[:300]}")

    # 2. 第一轮 Redis 落库验证
    if ok:
        msgs = r.lrange(key(cid), 0, -1)
        check("round1 落库 2 条", len(msgs) == 2, f"LLEN={len(msgs)}")
        if len(msgs) == 2:
            import json as _json
            roles = [_json.loads(m).get("role") for m in msgs]
            check("round1 角色顺序 user→assistant", roles == ["user", "assistant"], str(roles))

    # 3. 第二轮：指代消解
    log_size_before = os.path.getsize(LOG_FILE)
    t0 = time.time()
    resp = requests.post(f"{API}/api/v1/chat",
                         json={"query": "它的电池容量是多少？", "conversation_id": cid}, timeout=200)
    dt2 = time.time() - t0
    ok2 = resp.status_code == 200
    body2 = resp.json() if ok2 else resp.text
    check("round2 HTTP 200", ok2, f"{dt2:.0f}s")
    if ok2:
        print(f"  intent={body2.get('intent')} | answer[:120]={body2.get('answer','')[:120]}")
        check("round2 有回答", bool(body2.get("answer")))
        msgs = r.lrange(key(cid), 0, -1)
        check("round2 落库 4 条", len(msgs) == 4, f"LLEN={len(msgs)}")
        # 从日志抓取第二轮改写列表，判定消解
        with open(LOG_FILE, "rb") as f:
            f.seek(log_size_before)
            tail = f.read().decode("utf-8", errors="replace")
        import re
        m = re.search(r"多问句引擎拆分列表: (\[[^\n]*\]).*?它的电池容量是多少？", tail, re.DOTALL)
        # 日志顺序：先"拆分列表"后可能还有其他请求；直接用最后一条含"电池容量"上下文的拆分列表
        lines = tail.splitlines()
        rewrite_line = ""
        for i, ln in enumerate(lines):
            if "它的电池容量是多少？" in ln and "收到 Query 请求" in ln:
                for j in range(i, min(i + 15, len(lines))):
                    if "多问句引擎拆分列表" in lines[j]:
                        rewrite_line = lines[j]
                        break
        if rewrite_line:
            print(f"  LOG: {rewrite_line}")
            resolved = "星曜X1 Pro" in rewrite_line
            check("round2 指代消解成功（改写含“星曜X1 Pro”）", resolved)
        else:
            check("round2 从日志提取改写列表", False, "未找到改写日志行")

    # 4. DELETE 清除会话
    resp = requests.delete(f"{API}/api/v1/conversations/{cid}", timeout=10)
    check("DELETE HTTP 200", resp.status_code == 200, resp.json().get("message", "")[:60])
    exists = r.exists(key(cid))
    check("DELETE 后 Redis key 已删除", exists == 0, f"EXISTS={exists}")

    # 汇总
    failed = [n for n, o, _ in results if not o]
    print(f"\n=== 汇总: {len(results) - len(failed)}/{len(results)} 通过 ===")
    if failed:
        print("失败项:", ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()