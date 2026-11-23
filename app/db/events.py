"""
app/db/events.py

v2.0 事务事件存储：为 SSE 流式推送与 after_seq 断线补发提供唯一可信数据源。

核心约定：
  1. 一次用户请求 = 一个 turn（chat_turns 一行），状态 running → completed / failed；
  2. turn 内每个事件按 seq 从 1 开始严格单调递增（chat_events，UNIQUE(turn_id, seq)）；
  3. 业务层必须"先落库拿到 seq，再推送给客户端"——客户端看到的每个 seq 都一定能被补发接口查到；
  4. 终态事件（done / error）与 turn 状态更新在同一事务提交，补发方不会读到"有 done 但仍 running"的中间态。
"""

import json
import logging
import sqlite3
import uuid
from datetime import datetime
from typing import Any

from app.db.database import get_db_connection

logger = logging.getLogger(__name__)

TURN_RUNNING = "running"
TURN_COMPLETED = "completed"
TURN_FAILED = "failed"


def _now() -> str:
    """毫秒精度时间戳，便于排查同一 turn 内事件的先后与间隔。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def _row_to_event(row: sqlite3.Row, turn_id: str) -> dict[str, Any]:
    """数据库行 → 对外统一事件结构（SSE 推送与补发接口共用，保证两条通道数据完全一致）。"""
    return {
        "turn_id": turn_id,
        "seq": row["seq"],
        "type": row["event_type"],
        "data": json.loads(row["payload"]),
        "created_at": row["created_at"],
    }


def create_turn(session_id: str, user_message: str) -> str:
    """创建一个 running 状态的 turn，返回 turn_id（uuid4 十六进制，不可猜测）。"""
    turn_id = uuid.uuid4().hex
    sql = """
    INSERT INTO chat_turns (turn_id, session_id, user_message, status, created_at)
    VALUES (?, ?, ?, ?, ?);
    """
    try:
        with get_db_connection() as conn:
            conn.execute(sql, (turn_id, session_id, user_message, TURN_RUNNING, _now()))
            conn.commit()
    except sqlite3.Error:
        logger.exception("create_turn 写入失败 session_id=%s", session_id)
        raise
    logger.info("create_turn 成功 turn_id=%s session_id=%s", turn_id, session_id)
    return turn_id


def _insert_event(
    cursor: sqlite3.Cursor, turn_id: str, event_type: str, payload: dict[str, Any]
) -> dict[str, Any]:
    """
    在调用方已开启的写事务中分配 seq 并插入事件。

    为什么 seq 用 MAX(seq)+1 且必须处于 BEGIN IMMEDIATE 事务内：
      IMMEDIATE 会在事务开始时就拿到库级写锁，"读 MAX → 写入"之间不可能插入其他写者，
      因此同一 turn 内 seq 天然连续、严格递增；UNIQUE(turn_id, seq) 再兜底拒绝任何重复。
    """
    cursor.execute("SELECT status FROM chat_turns WHERE turn_id = ?;", (turn_id,))
    turn = cursor.fetchone()
    if turn is None:
        raise ValueError(f"turn_id={turn_id} 不存在，拒绝写入事件")
    if turn["status"] != TURN_RUNNING:
        # 终态之后不允许再追加事件，否则补发方会在 done 之后读到"幽灵事件"
        raise ValueError(f"turn_id={turn_id} 已结束（status={turn['status']}），拒绝写入事件")

    cursor.execute(
        "SELECT COALESCE(MAX(seq), 0) + 1 AS next_seq FROM chat_events WHERE turn_id = ?;",
        (turn_id,),
    )
    seq: int = cursor.fetchone()["next_seq"]
    created_at = _now()
    cursor.execute(
        """
        INSERT INTO chat_events (turn_id, seq, event_type, payload, created_at)
        VALUES (?, ?, ?, ?, ?);
        """,
        (turn_id, seq, event_type, json.dumps(payload, ensure_ascii=False), created_at),
    )
    return {"turn_id": turn_id, "seq": seq, "type": event_type, "data": payload, "created_at": created_at}


def append_event(turn_id: str, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    """事务安全地追加一条非终态事件，返回含 seq 的完整事件记录。"""
    try:
        with get_db_connection() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            record = _insert_event(conn.cursor(), turn_id, event_type, payload)
            conn.commit()
    except (sqlite3.Error, ValueError):
        # 异常时 get_db_connection 已自动回滚；这里补充结构化日志后继续上抛，由编排层决定如何收尾
        logger.exception("append_event 失败 turn_id=%s type=%s", turn_id, event_type)
        raise
    return record


def finish_turn(
    turn_id: str,
    status: str,
    final_event_type: str,
    final_payload: dict[str, Any],
    reply: str,
    tools_called: list[str],
    citations: list[str],
    ttft_ms: int | None,
    latency_ms: int,
    error: str | None = None,
) -> dict[str, Any]:
    """
    结束 turn：在同一事务内写入终态事件（done / error）并更新 chat_turns 汇总字段。
    返回终态事件记录（含 seq），供调用方推送。
    """
    if status not in (TURN_COMPLETED, TURN_FAILED):
        raise ValueError(f"非法终态 status={status}")

    update_sql = """
    UPDATE chat_turns
    SET status = ?, reply = ?, tools_called = ?, citations = ?,
        ttft_ms = ?, latency_ms = ?, error = ?, finished_at = ?
    WHERE turn_id = ?;
    """
    try:
        with get_db_connection() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            cursor = conn.cursor()
            # 先插事件（此时 status 仍是 running，可通过校验），再翻转状态
            record = _insert_event(cursor, turn_id, final_event_type, final_payload)
            cursor.execute(
                update_sql,
                (
                    status, reply,
                    json.dumps(tools_called, ensure_ascii=False),
                    json.dumps(citations, ensure_ascii=False),
                    ttft_ms, latency_ms, error, _now(), turn_id,
                ),
            )
            conn.commit()
    except (sqlite3.Error, ValueError):
        logger.exception("finish_turn 失败 turn_id=%s status=%s", turn_id, status)
        raise
    logger.info(
        "finish_turn turn_id=%s status=%s ttft_ms=%s latency_ms=%d", turn_id, status, ttft_ms, latency_ms
    )
    return record


def get_turn(turn_id: str) -> dict[str, Any] | None:
    """查询 turn 概要信息，不存在返回 None。"""
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT turn_id, session_id, status, ttft_ms, latency_ms FROM chat_turns WHERE turn_id = ?;",
            (turn_id,),
        ).fetchone()
    return dict(row) if row else None


def get_events_after(turn_id: str, after_seq: int = 0) -> list[dict[str, Any]]:
    """
    断线补发：返回 seq > after_seq 的全部事件，按 seq 升序。
    客户端只需记住最后收到的 SSE id，重连时作为 after_seq 传入即可"不重不漏"。
    """
    sql = """
    SELECT seq, event_type, payload, created_at
    FROM chat_events
    WHERE turn_id = ? AND seq > ?
    ORDER BY seq ASC;
    """
    with get_db_connection() as conn:
        rows = conn.execute(sql, (turn_id, after_seq)).fetchall()
    return [_row_to_event(row, turn_id) for row in rows]
