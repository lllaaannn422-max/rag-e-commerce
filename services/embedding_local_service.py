# services/embedding_local_service.py
import numpy as np
from typing import List, Union
import torch
from sentence_transformers import SentenceTransformer
import os
import sys

# project_root = os.path.dirname(os.path.dirname(__file__))  # 这是 rag_project_v2
# sys.path.insert(0, project_root)  # 插入到最前面，确保优先

# import config
# print("config path:", config.__file__)  # 可打印验证
import config.config as config  # 直接导入 config.config

from utils.logger import logger
# 如果还需要重试装饰器，可以保留导入，但本地调用不需要重试，这里保留以防其他地方用到
from utils.decorators import retry_on_exception


class EmbeddingService:
    """基于本地 SentenceTransformer 模型的 Embedding 服务类（单例）"""
    
    _instance = None
    _model = None
    _dim = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
            cls._instance._init_model()
        return cls._instance

    def _init_model(self):
        """加载本地模型（只执行一次）"""
        if EmbeddingService._model is not None:
            return

        # 从 config 读取模型路径，若未配置则使用默认名称（会自动从 Hugging Face 下载）
        model_path = getattr(config, "LOCAL_EMBED_MODEL_PATH", "BAAI/bge-m3")
        logger.info(f"正在加载本地 Embedding 模型: {model_path} ...")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        EmbeddingService._model = SentenceTransformer(model_path, device=device)
        EmbeddingService._dim = EmbeddingService._model.get_embedding_dimension()
        logger.info(f"模型加载完成，设备: {device}，输出维度: {EmbeddingService._dim}")
        
        # 自动更新 config.EMBED_DIM，确保维度一致
        if hasattr(config, "EMBED_DIM"):
            if config.EMBED_DIM != EmbeddingService._dim:
                logger.warning(
                    f"config.EMBED_DIM ({config.EMBED_DIM}) 与模型维度 ({EmbeddingService._dim}) 不一致，已自动修正。"
                )
            config.EMBED_DIM = EmbeddingService._dim

    # ---------- 公共接口（保持与原有代码兼容） ----------
    def get_embeddings(self, texts: Union[str, List[str]], is_query: bool = False) -> List[List[float]]:
        """
        获取文本的向量表示。

        :param texts: 单个字符串或字符串列表
        :param is_query: 是否为查询语句；Qwen3-Embedding 查询需带指令前缀（prompt_name="query"）
        :return: 向量列表 [ [dim_1, dim_2, ...], ... ]
        """
        if isinstance(texts, str):
            input_data = [texts]
        else:
            input_data = texts

        if not input_data:
            return []

        # 调用本地模型编码（query 使用模型内置的查询指令，document 不加前缀）
        embeddings = EmbeddingService._model.encode(
            input_data,
            prompt_name="query" if is_query else None,
            normalize_embeddings=True,      # 建议开启，提高检索效果
            batch_size=32,                  # CPU/GPU 均可，可调
            show_progress_bar=False
        )
        return embeddings.tolist()

    def encode(self, texts: Union[str, List[str]], is_query: bool = False, **kwargs) -> np.ndarray:
        """
        通用向量化接口（兼容现有调用）
        返回 numpy.ndarray，与之前行为一致
        """
        embeddings = self.get_embeddings(texts, is_query=is_query)
        if not embeddings:
            return np.array([])

        if isinstance(texts, str):
            return np.array(embeddings[0], dtype=np.float32)
        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> List[float]:
        """对单个查询语句进行向量化"""
        embeddings = self.get_embeddings(query, is_query=True)
        return embeddings[0] if embeddings else []

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """批量对文档块进行向量化"""
        return self.get_embeddings(documents, is_query=False)

    # 别名（保持兼容）
    get_embedding = embed_query


# ---------- 测试入口 ----------
if __name__ == "__main__":
    embed_service = EmbeddingService()
    
    # 测试单文本
    vec1 = embed_service.encode("令狐冲的爱人是谁", is_query=True)
    print(f"单文本向量维度: {len(vec1)}，类型: {type(vec1)}")
    print(f"tolist() 成功: {isinstance(vec1.tolist(), list)}")
    
    # 测试列表
    vec2 = embed_service.encode(["令狐冲", "任盈盈"])
    print(f"列表向量 shape: {vec2.shape}，类型: {type(vec2)}")
    print(f"tolist() 成功: {isinstance(vec2.tolist(), list)}")