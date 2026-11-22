import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator
from dotenv import load_dotenv

# 加载 .env 环境变量配置
load_dotenv()

# 从环境变量获取数据库存储路径，未配置时回退到项目根目录下的 data/smart_cs.db
DEFAULT_DB_PATH = "./data/smart_cs.db"
DB_PATH = os.getenv("SQLITE_DB_PATH", DEFAULT_DB_PATH)


def _ensure_db_dir_exists(db_path: str) -> None:
    """
    检查并确保数据库文件所在的父级目录存在。
    sqlite3.connect 在目录不存在时会直接抛出 OperationalError，因此必须先主动创建目录。
    """
    db_file = Path(db_path)
    parent_dir = db_file.parent
    if not parent_dir.exists():
        parent_dir.mkdir(parents=True, exist_ok=True)


@contextmanager
def get_db_connection() -> Generator[sqlite3.Connection, None, None]:
    """
    提供 SQLite 连接上下文管理器：
    1. 启用 row_factory = sqlite3.Row，以便支持按字段名（如 row['order_id']）字典式读取；
    2. 使用 try...except...finally 保障：发生异常时自动回滚事务，离开作用域时无论如何都安全关闭连接，
       防止在 FastAPI 并发请求下出现文件句柄泄漏或 database is locked 错误。
    """
    _ensure_db_dir_exists(DB_PATH)
    # timeout=15.0 配合 check_same_thread=False，防止并发评测多线程建单时触发 database is locked
    conn = sqlite3.connect(DB_PATH, timeout=15.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """
    初始化数据表结构（幂等设计）：
    使用 CREATE TABLE IF NOT EXISTS，服务每次启动或重复执行均安全无副作用。
    
    包含三张核心业务表：
    - orders: 模拟电商订单数据
    - tickets: 售后工单流转记录
    - chat_logs: 完整对话记录与工具调用延迟追踪
    """
    create_orders_sql = """
    CREATE TABLE IF NOT EXISTS orders (
        order_id TEXT PRIMARY KEY,
        user_name TEXT NOT NULL,
        product_name TEXT NOT NULL,
        amount REAL NOT NULL,
        status TEXT NOT NULL,
        tracking_no TEXT,
        created_at TEXT NOT NULL
    );
    """

    create_tickets_sql = """
    CREATE TABLE IF NOT EXISTS tickets (
        ticket_id TEXT PRIMARY KEY,
        order_id TEXT NOT NULL,
        issue_type TEXT NOT NULL,
        reason TEXT,
        status TEXT NOT NULL DEFAULT '待处理',
        summary TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY (order_id) REFERENCES orders (order_id)
    );
    """

    create_chat_logs_sql = """
    CREATE TABLE IF NOT EXISTS chat_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        tool_called TEXT,
        latency_ms INTEGER,
        created_at TEXT NOT NULL
    );
    """

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(create_orders_sql)
        cursor.execute(create_tickets_sql)
        cursor.execute(create_chat_logs_sql)
        conn.commit()


if __name__ == "__main__":
    init_db()
    print(f"数据库表初始化完成，文件路径: {DB_PATH}")
