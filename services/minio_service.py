# minio_service.py
from minio import Minio
import config.config as config
from utils.decorators import retry_on_exception
from utils.logger import logger

class MinioService:
    def __init__(self):
        self.client = Minio(
            config.MINIO_ENDPOINT,
            access_key=config.MINIO_ACCESS_KEY,
            secret_key=config.MINIO_SECRET_KEY,
            secure=False
        )

    @retry_on_exception()
    def ensure_bucket(self, bucket_name: str):
        if not self.client.bucket_exists(bucket_name):
            self.client.make_bucket(bucket_name)
            logger.info(f"成功创建 Bucket: {bucket_name}")

    @retry_on_exception()
    def upload_file(self, local_path: str, obj_name: str, bucket_name: str) -> str:
        self.client.fput_object(bucket_name, obj_name, local_path)
        return obj_name

    @retry_on_exception()
    def download_file(self, obj_name: str, save_path: str, bucket_name: str) -> str:
        self.client.fget_object(bucket_name, obj_name, save_path)
        return save_path