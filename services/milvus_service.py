# milvus_service.py
from pathlib import Path
from typing import List, Dict, Any, Optional
from pymilvus import connections, Collection, FieldSchema, DataType, CollectionSchema, utility
import config.config as config
from utils.logger import logger
# from services.embedding_service import EmbeddingService
from services.embedding_local_service import EmbeddingService as LocalEmbeddingService  # 改用local

class MilvusService:
    def __init__(self):
        # self.embed_service = EmbeddingService()
        self.embed_service = LocalEmbeddingService()
        self._init_connection()
        self.collection = self._init_collection()

    def _init_connection(self):
        db_file = Path(config.MILVUS_DB_PATH).resolve()
        db_file.parent.mkdir(parents=True, exist_ok=True)
        uri_path = db_file.as_posix()
        logger.info(f"初始化 Milvus 数据库连接: {uri_path}")
        connections.connect(alias="default", uri=uri_path)

    def _create_collection(self) -> Collection:
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="chunk_text", dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="file_suffix", dtype=DataType.VARCHAR, max_length=16),
            FieldSchema(name="source_bucket", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="publish_time", dtype=DataType.VARCHAR, max_length=30),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=config.EMBED_DIM)
        ]
        schema = CollectionSchema(fields=fields, description="知识库向量集合")
        coll = Collection(name=config.COLLECTION_NAME, schema=schema)
        # 电商客服语料仅几十个 chunk：FLAT 精确检索（免训练、无聚类），
        # 规避 IVF 小库聚类导致部分向量不可达（实测 44 向量只召回 ~26）与训练点不足警告
        index_params = {"index_type": "FLAT", "metric_type": "COSINE", "params": {}}
        coll.create_index(field_name="vector", index_params=index_params)
        coll.load()
        return coll

    def _init_collection(self) -> Collection:
        if utility.has_collection(config.COLLECTION_NAME):
            coll = Collection(config.COLLECTION_NAME)
            field_names = [f.name for f in coll.schema.fields]
            if "publish_time" not in field_names:
                coll.drop()
                return self._create_collection()
            coll.load()
            return coll
        return self._create_collection()

    def insert_chunks(self, chunks: List[str], filename: str, suffix: str, bucket: str, publish_time: str):
        vectors = self.embed_service.encode(chunks, is_query=False)
        data = [
            chunks,
            [filename] * len(chunks),
            [suffix] * len(chunks),
            [bucket] * len(chunks),
            [publish_time] * len(chunks),
            vectors.tolist()
        ]
        self.collection.insert(data)
        self.collection.flush()
        logger.info(f"成功将 {len(chunks)} 条 Chunk 数据插入到 Milvus。")

    def search_knowledge(self, query: str, top_k: int = 100, filter_expr: Optional[str] = None):
            """线程安全的单 Query 向量检索能力"""
            # 1. 生成向量
            encoded = self.embed_service.encode(query, is_query=True)
            
            # 2. 判断返回结构并提取单条向量
            # 如果返回的是 numpy 数组或 PyTorch Tensor
            if hasattr(encoded, "tolist"):
                vec_list = encoded.tolist()
            else:
                vec_list = encoded

            # 如果返回的是二维列表/数组 (例如 shape为 [1, dim])，取第一条；若本身就是一维列表，直接使用
            if isinstance(vec_list, list) and len(vec_list) > 0 and isinstance(vec_list[0], list):
                query_vec = vec_list[0]
            else:
                query_vec = vec_list

            # 3. 确保元素类型均为标准的 Python float（防止 numpy.float16 报错）
            query_vec = [float(x) for x in query_vec]

            # FLAT 索引无聚类，nprobe 无效，params 留空即可
            search_params = {"metric_type": "COSINE", "params": {}}
            
            # 4. 传给 Milvus 的 data 必须是二维列表 [[float, float, ...]]
            results = self.collection.search(
                data=[query_vec],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                expr=filter_expr,
                output_fields=["chunk_text", "file_name", "file_suffix", "source_bucket", "publish_time"]
            )
            return results

    def fetch_all_docs(self, offset: int = 0, limit: int = 1000):
        return self.collection.query(
            expr="id >= 0",
            output_fields=["chunk_text", "file_name", "source_bucket", "publish_time"],
            offset=offset,
            limit=limit
        )