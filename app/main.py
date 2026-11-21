"""
app/main.py

FastAPI 应用入口：初始化数据库、挂载所有路由。
"""
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict

from fastapi import FastAPI

from app.db.database import init_db
from app.routers.orders import router as orders_router
from app.routers.tickets import router as tickets_router
from app.routers.chat import router as chat_router
from app.routers.knowledge import router as knowledge_router

# 配置全局日志格式
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("smart-cs-agent")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI 生命周期管理：
    服务启动前自动执行 init_db() 并等待初始化 SQLite 数据库及数据表，
    确保应用就绪前数据底座已完全建立。
    """
    logger.info("应用正在启动，执行数据库初始化检测..")
    init_db()
    logger.info("数据库初始化完成，系统就绪")
    yield
    logger.info("应用正在关闭...")


app = FastAPI(
    title="Smart CS Agent API",
    description="企业智能售后与工单分流 Agent 核心后端接口",
    version="0.3.0",
    lifespan=lifespan,
)

# 挂载订单、工单、对话、知识库检索路由
app.include_router(orders_router)
app.include_router(tickets_router)
app.include_router(chat_router)
app.include_router(knowledge_router)


@app.get("/health", summary="健康检查接口", tags=["system"])
def health_check() -> Dict[str, str]:
    """
    服务存活探测与健康检查
    """
    return {"status": "ok", "service": "smart-cs-agent"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
