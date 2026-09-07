# test_faq_chunker.py
"""DocumentProcessor 问答对切块单元测试（无网络）"""
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.document_processor import DocumentProcessor as DP


FAQ_TEXT = """# 测试 FAQ

## Q：手机支持 5G 吗？

A：支持，需运营商 5G 套餐。

## Q：保修期多久？

A：整机保修 1 年。
"""

NO_QA_TEXT = """这是普通商品文档。
没有任何问答标记。
段落连续书写。
"""


class FaqChunkerTest(unittest.TestCase):
    def test_01_two_pairs_two_chunks(self):
        chunks = DP.qa_chunk(FAQ_TEXT)
        self.assertEqual(len(chunks), 2)
        self.assertIn("Q：手机支持 5G 吗？", chunks[0])
        self.assertIn("需运营商 5G 套餐", chunks[0])
        self.assertIn("Q：保修期多久？", chunks[1])
        self.assertIn("整机保修 1 年", chunks[1])

    def test_02_heading_prefix_variant(self):
        # 半角冒号 Q: 也会被识别，且输出统一规范为全角 Q：
        chunks = DP.qa_chunk("### Q：屏幕多大？\nA：6.1 英寸。\nQ: 多少钱？\nA: 2399 元。")
        self.assertEqual(len(chunks), 2)
        self.assertIn("Q：屏幕多大？", chunks[0])
        self.assertIn("Q：多少钱？", chunks[1])

    def test_03_long_answer_split_with_continuation_prefix(self):
        long_a = "很长的答案内容。" * 600  # 4800+ 字符 > 4096，必须切分
        chunks = DP.qa_chunk(f"Q：为什么这么长？\nA：{long_a}")
        self.assertGreaterEqual(len(chunks), 2)
        self.assertIn("（续）", chunks[1])
        self.assertIn("Q：为什么这么长？", chunks[1])

    def test_04_no_qa_marker_falls_back_to_sliding(self):
        self.assertEqual(len(DP.qa_chunk(NO_QA_TEXT)), len(DP.sliding_chunk(NO_QA_TEXT)))

    def test_05_chunk_text_dispatch(self):
        self.assertEqual(len(DP.chunk_text(FAQ_TEXT, "qa")), 2)
        self.assertEqual(len(DP.chunk_text(FAQ_TEXT, "sliding")), len(DP.sliding_chunk(FAQ_TEXT)))
        # 默认模式为 sliding
        self.assertEqual(len(DP.chunk_text(FAQ_TEXT)), len(DP.sliding_chunk(FAQ_TEXT)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
