# -*- coding: utf-8 -*-
"""
知识库摄入脚本：把电商语料目录下的文档解析、切片、向量化并写入 Milvus。

- 按一级子目录映射 bucket（config.ECOMMERCE_SUBDIR_BUCKET_MAP），faq 子目录自动走问答对切块；
- 默认跳过 MinIO 上传（--with-minio 可开启，需 MinIO 服务已启动）；
- 幂等：已入库（file_name 已存在）的文档自动跳过。
"""
import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config.config as config
from pipeline.document_processor import DocumentProcessor
from services.milvus_service import MilvusService
from utils.logger import logger


def list_docs(doc_root: str) -> list:
    """遍历 doc_root 一层子目录，返回 [(Path, subdir_name)]，未知子目录告警跳过"""
    root = Path(doc_root)
    found = []
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        if sub.name not in config.ECOMMERCE_SUBDIR_BUCKET_MAP:
            logger.warning(f"子目录 [{sub.name}] 不在 bucket 映射中，跳过: {config.ECOMMERCE_SUBDIR_BUCKET_MAP}")
            continue
        for suffix in config.SUPPORT_SUFFIX:
            found.extend((f, sub.name) for f in sorted(sub.glob(f"*{suffix}")))
    # 兼容：顶层散落的支持文件按默认 bucket（rag-product）摄入
    for suffix in config.SUPPORT_SUFFIX:
        found.extend((f, None) for f in sorted(root.glob(f"*{suffix}")))
    return found


def get_existing_docs(milvus: MilvusService) -> set:
    existing = set()
    offset = 0
    while True:
        rows = milvus.fetch_all_docs(offset=offset, limit=1000)
        if not rows:
            break
        for r in rows:
            existing.add(r.get("file_name"))
        if len(rows) < 1000:
            break
        offset += 1000
    return existing


def ingest(doc_root: str, with_minio: bool):
    milvus = MilvusService()
    existing = get_existing_docs(milvus)
    logger.info(f"知识库已有 {len(existing)} 个文档，跳过已入库文件")

    files = list_docs(doc_root)
    if not files:
        logger.warning(f"{doc_root} 下没有支持的文件（{config.SUPPORT_SUFFIX}）")
        return

    processor = DocumentProcessor() if with_minio else None
    for f, subdir in files:
        name = f.name
        if name in existing:
            logger.info(f"跳过（已入库）: {name}")
            continue
        bucket = config.ECOMMERCE_SUBDIR_BUCKET_MAP.get(subdir, "rag-product")
        chunk_mode = "qa" if bucket == "rag-faq" else "sliding"
        logger.info(f"开始摄入: {name} (bucket={bucket}, chunk_mode={chunk_mode})")
        if with_minio:
            processor.process_and_index_file(str(f), bucket, chunk_mode=chunk_mode)
        else:
            text = DocumentProcessor.extract_file_text(str(f))
            if not text:
                logger.warning(f"文件内容为空，跳过: {name}")
                continue
            chunks = DocumentProcessor.chunk_text(text, chunk_mode)
            pub = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(f)))
            milvus.insert_chunks(chunks, name, f.suffix.lower(), bucket, pub)
            logger.info(f"[{name}] 摄入完成，切片数: {len(chunks)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="摄入电商语料文档到 Milvus 知识库（子目录→bucket）")
    parser.add_argument("--doc-root", default=config.ECOMMERCE_ROOT_DIR, help="文档根目录（默认 docs/ecommerce）")
    parser.add_argument("--with-minio", action="store_true", help="同时上传 MinIO（需 MinIO 已启动）")
    args = parser.parse_args()
    ingest(args.doc_root, args.with_minio)
