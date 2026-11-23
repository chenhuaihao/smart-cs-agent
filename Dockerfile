# ==============================================================================
# Dockerfile: smart-cs-agent v2.0 多阶段轻量生产构建
#   Stage 1 (node:20-alpine)    : 编译 React + TypeScript + Vite 前端产物 -> frontend/dist
#   Stage 2 (python:3.12-slim)  : 运行 FastAPI + LangGraph + SQLite + Chroma 轻量后端并托管前端
# ==============================================================================

# ---------- Stage 1: 前端静态构建 ----------
FROM node:20-alpine AS frontend-builder

WORKDIR /build/frontend

# 优先拷贝依赖清单利用 Docker 构建缓存
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build


# ---------- Stage 2: Python 后端与单容器运行态 ----------
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 仅安装轻量依赖（严禁引入 torch / sentence-transformers 等重型本地模型）
COPY requirements.txt .
RUN pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt

# 复制后端源码、知识库初始文档、评测汇总产物以及 Stage 1 构建好的前端静态产物
COPY app/ ./app/
COPY data/knowledge/ ./data/knowledge/
COPY eval/eval_summary.json ./eval/eval_summary.json
COPY --from=frontend-builder /build/frontend/dist ./frontend/dist

EXPOSE 8000

# 生产级存活探测健康检查（纯 Python 标准库探测 /health）
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).getcode() == 200 else 1)"

# 容器启动命令（透传反向代理请求头与 IP 白名单）
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips=*"]
