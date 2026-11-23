# ==============================================================================
# Dockerfile: smart-cs-agent 生产容器化构建规范
# ==============================================================================

FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 设置 Python 容器环境变量
# PYTHONDONTWRITEBYTECODE: 禁止生成 .pyc 字节码文件，减小容器文件层体积
# PYTHONUNBUFFERED: 强制标准输出与标准错误不缓冲，实时透传容器日志
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 复制依赖清单并以 --no-cache-dir 方式安装，关闭 pip 缓存减小镜像体积
COPY requirements.txt .
RUN pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt

# 复制业务源码与初始数据
COPY app/ ./app/
COPY data/ ./data/

# 暴露服务端口
EXPOSE 8000

# 生产级存活探测健康检查（纯 Python 标准库探测 /health 接口，无需额外安装 curl/wget 包）
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).getcode() == 200 else 1)"

# 容器启动命令（透传反向代理请求头与 IP 白名单）
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips=*"]
