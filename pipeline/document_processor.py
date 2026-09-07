# document_processor.py
import os
import re
import pdfplumber
from datetime import datetime
from typing import List
import config.config as config
from utils.logger import logger
from services.milvus_service import MilvusService
from services.minio_service import MinioService


class DocumentProcessor:
    def __init__(self):
        self.milvus_service = MilvusService()
        self.minio_service = MinioService()

    @staticmethod
    def extract_file_text(file_path: str) -> str:
        suffix = os.path.splitext(file_path)[1].lower()
        text = ""
        if suffix == ".pdf":
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_txt = page.extract_text()
                    if page_txt:
                        text += page_txt + "\n"
        elif suffix in (".txt", ".md", ".py"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
            except UnicodeDecodeError:
                with open(file_path, "r", encoding="gbk") as f:
                    text = f.read()
        return text.replace("\r", "").replace("\n\n", "\n").strip()

    @staticmethod
    def sliding_chunk(text: str, chunk_size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP) -> List[str]:
        chunks = []
        start = 0
        total_len = len(text)
        while start < total_len:
            end = min(start + chunk_size, total_len)
            seg = text[start:end].strip()
            if seg:
                chunks.append(seg)
            start += chunk_size - overlap
        return chunks

    # FAQ 问答对切块：以 "Q：..."（可带 markdown 标题前缀）行作为条目起点，
    # 后续 A 内容整段归入同一条目，直到下一个 Q 出现，保证问答对不被定长窗口切散。
    QA_Q_PATTERN = re.compile(r"^(?:#{1,6}\s*)?Q\s*[：:]\s*(.+)$")

    @staticmethod
    def qa_chunk(text: str, max_len: int = 4096) -> List[str]:
        lines = text.split("\n")
        items = []          # list of (question, [answer_lines])
        cur_q, cur_body = None, []
        for ln in lines:
            m = DocumentProcessor.QA_Q_PATTERN.match(ln.strip())
            if m:
                if cur_q is not None:
                    items.append((cur_q, cur_body))
                cur_q, cur_body = m.group(1).strip(), []
            elif cur_q is not None:
                cur_body.append(ln)
        if cur_q is not None:
            items.append((cur_q, cur_body))
        if not items:
            # 无 Q 标记的文档回退滑动窗口切块
            return DocumentProcessor.sliding_chunk(text)

        chunks = []
        for q, body in items:
            full = f"Q：{q}\n" + "\n".join(body).strip()
            if len(full) <= max_len:
                chunks.append(full)
                continue
            # 超长条目：首个块 = Q + 答案前段；续块前缀保留问题，保证检索命中上下文
            pos = 0
            while pos < len(full):
                prefix = "" if pos == 0 else f"Q：{q}（续）\n"
                seg = (prefix + full[pos:pos + max_len - len(prefix)]).strip()
                if seg:
                    chunks.append(seg)
                pos += max_len - len(prefix)
        return chunks

    @staticmethod
    def chunk_text(text: str, chunk_mode: str = "sliding") -> List[str]:
        """chunk_mode: 'sliding'（商品/政策/活动文档） | 'qa'（FAQ 问答对）"""
        if chunk_mode == "qa":
            return DocumentProcessor.qa_chunk(text)
        return DocumentProcessor.sliding_chunk(text)

    def process_and_index_file(self, local_path: str, bucket_name: str, chunk_mode: str = "auto"):
        filename = os.path.basename(local_path)
        suffix = os.path.splitext(filename)[1].lower()

        if suffix not in config.SUPPORT_SUFFIX:
            logger.info(f"忽略不支持的文件格式: {filename}")
            return

        # 1.MinIO上传
        self.minio_service.ensure_bucket(bucket_name)
        self.minio_service.upload_file(local_path, filename, bucket_name)

        # 2.解析文本+切片（FAQ 桶默认按问答对切块，避免定长窗口切散 Q/A）
        text = self.extract_file_text(local_path)
        if not text:
            logger.warning(f"文件内容为空，跳过: {filename}")
            return
        if chunk_mode == "auto":
            chunk_mode = "qa" if bucket_name == "rag-faq" else "sliding"
        chunks = self.chunk_text(text, chunk_mode)
        mtime = os.path.getmtime(local_path)
        pub_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")

        # 3.仅写入Milvus向量库
        self.milvus_service.insert_chunks(chunks, filename, suffix, bucket_name, pub_str)
        logger.info(f"文件 [{filename}] 成功写入知识库，切片数: {len(chunks)}，切块模式: {chunk_mode}")
