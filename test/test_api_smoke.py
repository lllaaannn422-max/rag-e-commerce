# test_api_smoke.py
"""电商客服 API 冒烟测试（通过运行中的 API 服务，端口 8000）

覆盖 11 例：health / 售前 / 售后 / 活动 / 对比 / 订单命中 / 订单未命中 / 转人工 /
寒暄 / 多轮指代 / 空结果不编造。

前置条件：API 服务已启动（python api_server.py 或 uvicorn api_server:app --port 8000）。
运行：python test/test_api_smoke.py
"""
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests

API = "http://127.0.0.1:8000"

results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" | {detail}" if detail else ""))


def chat(query, conversation_id=None, timeout=200):
    payload = {"query": query}
    if conversation_id:
        payload["conversation_id"] = conversation_id
    resp = requests.post(f"{API}/api/v1/chat", json=payload, timeout=timeout)
    return resp.status_code, (resp.json() if resp.status_code == 200 else resp.text)


def main():
    # 1. health
    resp = requests.get(f"{API}/health", timeout=10)
    ok = resp.status_code == 200 and "Healthy" in resp.json().get("message", "")
    check("health 200+Healthy", ok, f"{resp.status_code}")

    # 2. 售前：电池容量
    code, body = chat("星曜X1 Pro 的电池容量是多少？")
    check("pre_sales 电池容量", code == 200 and body.get("intent") == "pre_sales"
          and body.get("route") == "rag" and "5000" in body.get("answer", ""),
          f"intent={body.get('intent') if isinstance(body, dict) else '-'} "
          f"answer[:60]={body.get('answer','')[:60] if isinstance(body, dict) else body}")

    # 3. 售后：无理由退货运费（生成措辞可能同义改写，命中任一表述即可）
    code, body = chat("7天无理由退货的运费由谁承担？")
    ans3 = body.get("answer", "") if isinstance(body, dict) else ""
    ok3 = (code == 200 and body.get("intent") == "after_sales"
           and any(w in ans3 for w in ("用户", "买家", "自行承担", "自己承担")))
    check("after_sales 退货运费", ok3,
          f"intent={body.get('intent') if isinstance(body, dict) else '-'} "
          f"answer[:60]={ans3[:60]}")

    # 4. 活动：满减
    code, body = chat("开学季活动订单满3000元能减多少？")
    check("promotion 满3000减300", code == 200 and body.get("intent") == "promotion"
          and "300" in body.get("answer", ""),
          f"intent={body.get('intent') if isinstance(body, dict) else '-'}")

    # 5. 对比：屏幕区别
    code, body = chat("星曜X1 Pro 和星曜X1 的屏幕有什么区别？")
    check("comparison 屏幕对比", code == 200 and body.get("intent") == "comparison"
          and ("6.7" in body.get("answer", "") or "AMOLED" in body.get("answer", "")),
          f"intent={body.get('intent') if isinstance(body, dict) else '-'} "
          f"answer[:60]={body.get('answer','')[:60] if isinstance(body, dict) else body}")

    # 6. 订单命中：配送中
    code, body = chat("帮我查一下订单 YX20260815001 到哪了？")
    ans6 = body.get("answer", "") if isinstance(body, dict) else ""
    ok6 = (code == 200 and body.get("route") == "order"
           and body.get("order_id") == "YX20260815001"
           and any(w in ans6 for w in ("配送", "运输", "派送", "送达")))
    check("order 命中 YX20260815001", ok6,
          f"route={body.get('route') if isinstance(body, dict) else '-'} "
          f"order_id={body.get('order_id') if isinstance(body, dict) else '-'} "
          f"answer[:60]={ans6[:60]}")

    # 7. 订单未命中：固定模板不编造
    code, body = chat("帮我查一下订单 XYZ999999 的物流")
    check("order 未命中固定模板", code == 200 and body.get("route") == "order"
          and "没有查到" in body.get("answer", "") and "转人工" in body.get("answer", ""),
          f"answer[:80]={body.get('answer','')[:80] if isinstance(body, dict) else body}")

    # 8. 转人工
    code, body = chat("我要投诉，你们服务态度太差了，给我转人工！")
    check("escalation 转人工", code == 200 and body.get("route") == "escalation"
          and body.get("needs_handoff") is True and len(body.get("answer", "")) > 10,
          f"route={body.get('route') if isinstance(body, dict) else '-'} "
          f"handoff={body.get('needs_handoff') if isinstance(body, dict) else '-'}")

    # 9. 寒暄
    code, body = chat("你好呀，在吗？")
    check("chitchat 寒暄", code == 200 and body.get("route") == "chitchat"
          and bool(body.get("answer")),
          f"route={body.get('route') if isinstance(body, dict) else '-'}")

    # 10. 多轮指代
    cid = f"smoke-e2e-{int(time.time())}"
    code, body1 = chat("星曜X1 Pro 的屏幕是多大尺寸的？", conversation_id=cid)
    code2, body2 = chat("它的电池容量是多少？", conversation_id=cid)
    ok = (code == 200 and code2 == 200
          and "6.7" in body1.get("answer", "")
          and "5000" in body2.get("answer", ""))
    check("multi-turn 指代消解", ok,
          f"r1[:40]={body1.get('answer','')[:40] if isinstance(body1, dict) else '-'} | "
          f"r2[:40]={body2.get('answer','')[:40] if isinstance(body2, dict) else '-'}")
    try:
        requests.delete(f"{API}/api/v1/conversations/{cid}", timeout=10)
    except Exception:
        pass

    # 11. 空结果不编造（知识库无笔记本电脑资料）
    code, body = chat("你们商城卖笔记本电脑吗？")
    check("empty-result 不编造", code == 200
          and "没有在知识库中找到" in body.get("answer", ""),
          f"intent={body.get('intent') if isinstance(body, dict) else '-'} "
          f"answer[:60]={body.get('answer','')[:60] if isinstance(body, dict) else body}")

    # 汇总
    failed = [n for n, o, _ in results if not o]
    print(f"\n=== 汇总: {len(results) - len(failed)}/{len(results)} 通过 ===")
    if failed:
        print("失败项:", ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
