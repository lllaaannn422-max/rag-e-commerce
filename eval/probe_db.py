# -*- coding: utf-8 -*-
"""探查 Milvus 知识库中实际入库的文档与 bucket 分布（评测前确认数据范围）"""
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config.config as config
from pymilvus import connections, Collection

db = Path(config.MILVUS_DB_PATH).resolve()
print(f"Milvus DB: {db}")
connections.connect(alias="probe", uri=db.as_posix())
coll = Collection(config.COLLECTION_NAME, using="probe")
coll.load()

rows = coll.query(expr="id >= 0", output_fields=["file_name", "source_bucket"], limit=50000)
counter = Counter((r["file_name"], r["source_bucket"]) for r in rows)
print("===== 知识库文档分布 (chunk 数 / 文件名 / bucket) =====")
for (fname, bucket), cnt in sorted(counter.items()):
    print(f"{cnt}\t{fname}\t{bucket}")
print("TOTAL_CHUNKS:", len(rows))
