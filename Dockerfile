FROM python:3.10-slim

WORKDIR /app

# 安装系统级依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 拷贝项目代码
COPY . .

# 创建日志与索引目录
RUN mkdir -p /app/logs /app/whoosh_index /app/docs

EXPOSE 8000

# 默认使用 uvicorn 启动，线上可通过 docker-compose 覆盖使用 gunicorn 多 worker 模式
CMD ["uvicorn", "api_server:app", "--host", "127.0.0.1", "--port", "8000", "--workers", "4"]