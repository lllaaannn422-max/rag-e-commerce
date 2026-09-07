import sys
import os
import time
import functools
from utils.logger import logger
# import config
# 将项目根目录添加到 sys.path 最前面
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

# 导入项目配置
import config.config as config
print("config file:", config.__file__)  # 调试用，可删除


def retry_on_exception(max_retries=config.MAX_RETRIES, delay=config.RETRY_DELAY, exceptions=(Exception,)):
    """高可用容错重试装饰器"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    logger.warning(f"执行 {func.__name__} 失败 (尝试 {attempt}/{max_retries}): {e}")
                    if attempt < max_retries:
                        time.sleep(delay)
                    else:
                        logger.error(f"执行 {func.__name__} 在重试 {max_retries} 次后终止: {e}")
                        raise last_exception
            return None
        return wrapper
    return decorator