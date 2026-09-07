# test_rewrite_stability.py
"""复现「他师父是谁？」指代消解随机性：
同一对话历史下多次调用 rewrite_queries，统计 空结果 / 未消解 / 消解成功 的比例。

背景：日志显示 demo-001 第二轮“他师父是谁？”改写曾返回空结果（拆分列表仅剩原句），
疑似 deepseek 改写随机性。本脚本定量复现。

运行：python test/test_rewrite_stability.py [轮数，默认5]
"""
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.deepseek_service import DeepSeekService

# 模拟 demo-001 第一轮落库的历史（用户已 DELETE，无法从 Redis 取原文；此处用等价内容）
HISTORY = [
    {"role": "user", "content": "令狐冲是谁？"},
    {"role": "assistant", "content": "令狐冲是金庸武侠小说《笑傲江湖》的男主角，华山派大弟子，师从华山派掌门岳不群，后随风清扬学得独孤九剑。"},
]
QUERY = "他师父是谁？"
# 兼容 unittest discover（argv[1] 为 "discover" 等非数字时用默认值）
try:
    RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
except (ValueError, IndexError):
    RUNS = 5


def main():
    svc = DeepSeekService()
    stats = {"empty": 0, "resolved": 0, "unresolved": 0, "error": 0}
    for i in range(RUNS):
        t0 = time.time()
        try:
            rewrites = svc.rewrite_queries(QUERY, history=HISTORY)
        except Exception as e:
            stats["error"] += 1
            print(f"[{i+1}] 异常（重试3次仍失败）: {type(e).__name__}: {e}")
            continue
        dt = time.time() - t0
        extra = [q for q in rewrites if q != QUERY]   # 除原句外的真正改写句
        resolved = any("令狐冲" in q for q in rewrites)
        if not extra:
            stats["empty"] += 1
        elif resolved:
            stats["resolved"] += 1
        else:
            stats["unresolved"] += 1
        print(f"[{i+1}] {dt:.1f}s | 改写数={len(extra)} | 消解={'Y' if resolved else 'N'} | {rewrites}")
    print(f"\n汇总（共{RUNS}次）: 空结果={stats['empty']}  消解成功={stats['resolved']}  未消解={stats['unresolved']}  异常={stats['error']}")


if __name__ == "__main__":
    main()
