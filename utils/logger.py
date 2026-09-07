import os
import sys
import logging
from pathlib import Path

# 获取日志输出目录（在项目根目录下创建 logs 文件夹）
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE_PATH = LOG_DIR / "rag_system.log"

def setup_logger(name: str = "rag_system", log_file: str = str(LOG_FILE_PATH), level=logging.INFO) -> logging.Logger:
    """
    配置并返回统一的日志记录器
    :param name: logger 名称
    :param log_file: 日志输出文件路径
    :param level: 日志级别
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重复添加 Handler
    if logger.handlers:
        return logger

    # 统一日志格式：[时间] - [模块名] - [日志级别] - [日志消息]
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 1. 文件日志 Handler (UTF-8 编码)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 2. 控制台日志 Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

# 全局默认 Logger 实例
logger = setup_logger()