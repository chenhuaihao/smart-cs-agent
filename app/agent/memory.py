"""
app/agent/memory.py

多轮上下文记忆（v2.0 从 core.py 原样迁出，SQL 与行为与 v1.0 完全一致）：
  - load_history：读取同一 session 最近 6 轮（12 条）对话作为上下文
  - save_message：把本轮 user / assistant 消息写入 chat_logs
同步接口 /api/chat 与流式接口 /api/chat/stream 共用这一份记忆，两种调用方式可以混用同一 session。
"""

import logging
from datetime import datetime

from app.db.database import get_db_connection

logger = logging.getLogger(__name__)

# 历史消息条数限制（最近 12 条 = 6 轮对话）
HISTORY_LIMIT = 12


def load_history(session_id: str) -> list[dict]:
    """
    从 chat_logs 表读取指定 session 最近 HISTORY_LIMIT 条对话记录，
    按 id DESC 取最新的，然后反转为时间正序，作为多轮上下文拼入 messages。
    只取 role 和 content，tool_called / latency_ms 是内部统计，不需要传给大模型。
    """
    sql = """
    SELECT role, content
    FROM chat_logs
    WHERE session_id = ?
    ORDER BY id DESC
    LIMIT ?;
    """
    try:
        with get_db_connection() as conn:
            rows = conn.execute(sql, (session_id, HISTORY_LIMIT)).fetchall()
    except Exception as e:
        # 读取历史失败不影响主流程，降级为空历史继续处理
        logger.error("load_history 数据库异常 session_id=%s: %s", session_id, e)
        return []

    # 查出是倒序的（最新在前），反转后得到时间正序的历史消息
    return [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]


def save_message(
    session_id: str,
    role: str,
    content: str,
    tool_called: str | None = None,
    latency_ms: int | None = None,
) -> None:
    """
    将单条消息写入 chat_logs 表。
    tool_called 仅在 assistant 消息时有值（逗号拼接的工具名列表）；
    latency_ms 仅在 assistant 消息时有值（端到端耗时，毫秒）。
    """
    sql = """
    INSERT INTO chat_logs (session_id, role, content, tool_called, latency_ms, created_at)
    VALUES (?, ?, ?, ?, ?, ?);
    """
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with get_db_connection() as conn:
            conn.execute(sql, (session_id, role, content, tool_called, latency_ms, created_at))
            conn.commit()
    except Exception as e:
        # 持久化失败记录日志，不阻断接口响应（已生成的回复照常返回）
        logger.error("save_message 写入失败 session_id=%s role=%s: %s", session_id, role, e)
