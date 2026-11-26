"""
app/main.py

FastAPI 应用入口：初始化数据库、挂载所有业务路由与前端静态工作台。
"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Dict

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.agent.core import get_active_running_turn_ids, notify_approval_expired
from app.agent.knowledge import get_retriever
from app.agent.run_guard import RunGuardWatchdog
from app.db.database import init_db
from app.db.seed import seed_orders, seed_tickets
from app.routers.orders import router as orders_router
from app.routers.tickets import router as tickets_router
from app.routers.chat import router as chat_router
from app.routers.knowledge import router as knowledge_router
from app.routers.eval_report import router as eval_report_router

# 静态资源与主页路径（通过 Path(__file__) 定位，不依赖运行目录）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_STATIC_DIR = Path(__file__).resolve().parent / "static"
_INDEX_HTML = _STATIC_DIR / "index.html"
# v2.0 P3：React + Vite 生产构建目录（存在时优先挂载，未构建时自动回退 v1.0 index.html）
_FRONTEND_DIST_DIR = _PROJECT_ROOT / "frontend" / "dist"
_FRONTEND_INDEX_HTML = _FRONTEND_DIST_DIR / "index.html"
_FRONTEND_ASSETS_DIR = _FRONTEND_DIST_DIR / "assets"

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
    seed_orders()
    seed_tickets()
    logger.info("数据库初始化、订单补录与演示工单就绪")
    # 预热知识库检索单例（前置 jieba 词典加载与 Chroma 持久化索引复用，消除首次检索冷启动）
    try:
        get_retriever()
    except Exception as e:
        logger.warning("知识库检索器启动预热异常（不影响服务启动，将在首次请求时重试）: %s", e)

    # v3.0 P1：启动运行态看门狗（先回收上次异常退出遗留的僵尸 running 与过期 HITL，再后台周期巡检）
    watchdog = RunGuardWatchdog(
        get_active_turn_ids=get_active_running_turn_ids,
        on_approval_expired=notify_approval_expired,
    )
    await watchdog.start()
    try:
        yield
    finally:
        await watchdog.stop()
        logger.info("应用正在关闭...")


app = FastAPI(
    title="Smart CS Agent API",
    description="企业智能售后与工单分流 Agent 核心后端接口",
    version="3.0.0",
    lifespan=lifespan,
)

# 挂载订单、工单、对话、知识库检索与只读评测报告路由
app.include_router(orders_router)
app.include_router(tickets_router)
app.include_router(chat_router)
app.include_router(knowledge_router)
app.include_router(eval_report_router)

# 挂载静态资源服务（供 /static/ 下静态文件直接访问）
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

# 若已执行 frontend 生产构建（npm run build），挂载 Vite 资源目录供单端口预览
if _FRONTEND_ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=_FRONTEND_ASSETS_DIR), name="frontend-assets")


@app.get("/", summary="前端智能售后工作台", tags=["frontend"])
def read_index() -> FileResponse:
    """
    提供智能售后工作台单页应用 (SPA)：
    优先返回 v2.0 React 构建产物（frontend/dist/index.html），未构建时平滑回退至 v1.0 静态页。
    """
    if _FRONTEND_INDEX_HTML.exists():
        return FileResponse(_FRONTEND_INDEX_HTML)
    return FileResponse(_INDEX_HTML)


@app.get("/health", summary="健康检查接口", tags=["system"])
def health_check() -> Dict[str, str]:
    """
    服务存活探测与健康检查
    """
    return {"status": "ok", "service": "smart-cs-agent"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )
