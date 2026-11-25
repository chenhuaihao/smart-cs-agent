"""
app/db/events.py

v2.0 事务事件存储：为 SSE 流式推送、after_seq 断线补发与 HITL 人工确认提供唯一可信数据源。

核心约定：
  1. 一次用户请求 = 一个 turn（chat_turns 一行），状态 running / waiting_approval → completed / failed；
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
TURN_WAITING_APPROVAL = "waiting_approval"
TURN_COMPLETED = "completed"
TURN_FAILED = "failed"
APPROVAL_EXPIRED = "expired"

# 允许追加事件的活跃状态（waiting_approval 期间也可在恢复瞬间写入后续事件）
_ACTIVE_STATUSES = {TURN_RUNNING, TURN_WAITING_APPROVAL}


def _now() -> str:
    """毫秒精度时间戳，便于排查同一 turn 内事件的先后与间隔。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def _parse_ts(ts_str: str | None) -> datetime | None:
    """解析 SQLite 中保存的时间戳（兼容带毫秒与不带毫秒格式），异常返回 None。"""
    if not ts_str:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(ts_str.strip(), fmt)
        except ValueError:
            continue
    return None


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
    处于 BEGIN IMMEDIATE 事务内，保证同一 turn 内 seq 连续严格递增。
    """
    cursor.execute("SELECT status FROM chat_turns WHERE turn_id = ?;", (turn_id,))
    turn = cursor.fetchone()
    if turn is None:
        raise ValueError(f"turn_id={turn_id} 不存在，拒绝写入事件")
    if turn["status"] not in _ACTIVE_STATUSES:
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
        logger.exception("append_event 失败 turn_id=%s type=%s", turn_id, event_type)
        raise
    return record


def save_pending_approval(
    turn_id: str,
    tool_name: str,
    call_id: str,
    args: dict[str, Any],
    summary: str,
) -> dict[str, Any]:
    """
    高风险写操作 HITL 暂停：
    在同一事务中：
      1. 持久化待确认请求到 chat_approvals（status='waiting_approval'）；
      2. 将 chat_turns.status 设为 'waiting_approval'；
      3. 写入 event: approval_required 事件并返回该事件记录（带 seq）。
    """
    payload = {
        "turn_id": turn_id,
        "tool": tool_name,
        "call_id": call_id,
        "args": args,
        "summary": summary,
    }
    now = _now()
    try:
        with get_db_connection() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            cursor = conn.cursor()
            record = _insert_event(cursor, turn_id, "approval_required", payload)
            cursor.execute(
                """
                INSERT OR REPLACE INTO chat_approvals
                    (turn_id, tool_name, call_id, args, summary, status, created_at, decided_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, NULL);
                """,
                (
                    turn_id,
                    tool_name,
                    call_id,
                    json.dumps(args, ensure_ascii=False),
                    summary,
                    TURN_WAITING_APPROVAL,
                    now,
                ),
            )
            cursor.execute(
                "UPDATE chat_turns SET status = ? WHERE turn_id = ?;",
                (TURN_WAITING_APPROVAL, turn_id),
            )
            conn.commit()
    except (sqlite3.Error, ValueError):
        logger.exception("save_pending_approval 失败 turn_id=%s tool=%s", turn_id, tool_name)
        raise
    logger.info("save_pending_approval turn_id=%s tool=%s summary=%s", turn_id, tool_name, summary)
    return record


def resolve_pending_approval(turn_id: str, approved: bool) -> dict[str, Any] | None:
    """
    原子 CAS 记录用户对高风险写操作的审批决策，并将 turn 状态切回 running 以便继续收尾。
    使用 BEGIN IMMEDIATE 同时校验 chat_approvals 与 chat_turns 均处于 waiting_approval，
    确保并发双击或已过期/已收敛的 turn 绝不会被重复审批或重复建单。
    返回待确认请求详情字典；若状态不匹配则返回 None。
    """
    decision_status = "approved" if approved else "rejected"
    try:
        with get_db_connection() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            cursor = conn.cursor()
            row = cursor.execute(
                """
                SELECT a.turn_id, a.tool_name, a.call_id, a.args, a.summary,
                       a.status AS approval_status, t.status AS turn_status
                FROM chat_approvals a
                JOIN chat_turns t ON t.turn_id = a.turn_id
                WHERE a.turn_id = ?;
                """,
                (turn_id,),
            ).fetchone()
            if (
                row is None
                or row["approval_status"] != TURN_WAITING_APPROVAL
                or row["turn_status"] != TURN_WAITING_APPROVAL
            ):
                conn.rollback()
                return None
            cursor.execute(
                "UPDATE chat_approvals SET status = ?, decided_at = ? WHERE turn_id = ? AND status = ?;",
                (decision_status, _now(), turn_id, TURN_WAITING_APPROVAL),
            )
            cursor.execute(
                "UPDATE chat_turns SET status = ? WHERE turn_id = ? AND status = ?;",
                (TURN_RUNNING, turn_id, TURN_WAITING_APPROVAL),
            )
            conn.commit()
            return {
                "turn_id": row["turn_id"],
                "tool_name": row["tool_name"],
                "call_id": row["call_id"],
                "args": json.loads(row["args"]),
                "summary": row["summary"],
                "status": decision_status,
            }
    except sqlite3.Error:
        logger.exception("resolve_pending_approval 失败 turn_id=%s", turn_id)
        raise


def expire_pending_approval(turn_id: str, reason: str = "审批等待超时自动取消") -> dict[str, Any] | None:
    """
    原子过期回收超时未审批的 HITL 请求：
      1. 仅当 chat_approvals 与 chat_turns 均仍为 waiting_approval 时生效（防与并发 /approve 竞态）；
      2. 将 chat_approvals.status 置为 'expired'，写入 stage(approval_expired) + done 终态事件；
      3. 将 chat_turns.status 置为 'completed'，明确告知未创建任何工单。
    若该 turn 已被用户审批或已结束，安全回滚并返回 None。
    """
    reply = "工单确认已超时自动取消（超过审批等待时限），未创建任何售后工单。如需继续办理，请重新发起申请。"
    now_str = _now()
    try:
        with get_db_connection() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            cursor = conn.cursor()
            row = cursor.execute(
                """
                SELECT a.status AS approval_status, t.status AS turn_status
                FROM chat_approvals a
                JOIN chat_turns t ON t.turn_id = a.turn_id
                WHERE a.turn_id = ?;
                """,
                (turn_id,),
            ).fetchone()
            if (
                row is None
                or row["approval_status"] != TURN_WAITING_APPROVAL
                or row["turn_status"] != TURN_WAITING_APPROVAL
            ):
                conn.rollback()
                return None

            _insert_event(cursor, turn_id, "stage", {"stage": "approval_expired", "reason": reason})
            done_payload = {
                "reply": reply,
                "tools_called": [],
                "citations": [],
                "ttft_ms": None,
                "latency_ms": 0,
                "expired": True,
            }
            record = _insert_event(cursor, turn_id, "done", done_payload)
            cursor.execute(
                "UPDATE chat_approvals SET status = ?, decided_at = ? WHERE turn_id = ?;",
                (APPROVAL_EXPIRED, now_str, turn_id),
            )
            cursor.execute(
                """
                UPDATE chat_turns
                SET status = ?, reply = ?, tools_called = ?, citations = ?,
                    error = ?, finished_at = ?
                WHERE turn_id = ?;
                """,
                (TURN_COMPLETED, reply, "[]", "[]", reason, now_str, turn_id),
            )
            conn.commit()
            logger.info("expire_pending_approval 完成 turn_id=%s reason=%s", turn_id, reason)
            return record
    except (sqlite3.Error, ValueError):
        logger.exception("expire_pending_approval 失败 turn_id=%s", turn_id)
        raise


def reap_stale_turns(
    approval_ttl_sec: float = 300.0,
    running_ttl_sec: float = 120.0,
    active_turn_ids: set[str] | None = None,
    on_startup: bool = False,
) -> dict[str, Any]:
    """
    批量收敛僵尸 running 轮次与超时 waiting_approval 轮次：
      - on_startup=True：服务启动时，任何残留在库中的 running 轮次都必为上次进程崩溃/重启遗留，直接收敛为 failed；
      - 周期巡检时：仅收敛不在 active_turn_ids 且创建时间距今 >= running_ttl_sec 的僵尸 running 轮次；
      - 对 waiting_approval 轮次：无论是否重启，只要审批创建时间距今 >= approval_ttl_sec，原子置为 expired。
    """
    now_dt = datetime.now()
    active_set = active_turn_ids or set()
    reaped_running = 0
    expired_turn_ids: list[str] = []

    # 1. 排查待收敛的 running 轮次
    with get_db_connection() as conn:
        running_rows = conn.execute(
            "SELECT turn_id, created_at FROM chat_turns WHERE status = ?;",
            (TURN_RUNNING,),
        ).fetchall()
        approval_rows = conn.execute(
            """
            SELECT a.turn_id, a.created_at
            FROM chat_approvals a
            JOIN chat_turns t ON t.turn_id = a.turn_id
            WHERE a.status = ? AND t.status = ?;
            """,
            (TURN_WAITING_APPROVAL, TURN_WAITING_APPROVAL),
        ).fetchall()

    for row in running_rows:
        tid = row["turn_id"]
        if not on_startup:
            if tid in active_set:
                continue
            created_dt = _parse_ts(row["created_at"])
            if created_dt is not None and (now_dt - created_dt).total_seconds() < running_ttl_sec:
                continue
        reason = "服务重启导致运行中任务中断，已自动收敛" if on_startup else "后台任务超时未收敛，已自动回收"
        if _reap_single_running_turn(tid, reason):
            reaped_running += 1

    # 2. 排查超时未审批的 waiting_approval 轮次
    for row in approval_rows:
        tid = row["turn_id"]
        created_dt = _parse_ts(row["created_at"])
        if created_dt is None or (now_dt - created_dt).total_seconds() >= approval_ttl_sec:
            reason = f"超过审批等待时限（{int(approval_ttl_sec)}s）自动取消"
            if expire_pending_approval(tid, reason=reason) is not None:
                expired_turn_ids.append(tid)

    return {
        "reaped_running": reaped_running,
        "expired_approvals": len(expired_turn_ids),
        "expired_turn_ids": expired_turn_ids,
    }


def _reap_single_running_turn(turn_id: str, reason: str) -> bool:
    """在 BEGIN IMMEDIATE 事务中原子收敛单条僵尸 running 轮次为 failed。"""
    now_str = _now()
    reply = "抱歉，该轮对话因服务重启或任务中断未能完成，请重新发送您的问题。"
    try:
        with get_db_connection() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            cursor = conn.cursor()
            row = cursor.execute(
                "SELECT status FROM chat_turns WHERE turn_id = ?;",
                (turn_id,),
            ).fetchone()
            if row is None or row["status"] != TURN_RUNNING:
                conn.rollback()
                return False
            _insert_event(
                cursor,
                turn_id,
                "error",
                {"message": reply, "reply": reply, "reason": reason, "reaped": True},
            )
            cursor.execute(
                """
                UPDATE chat_turns
                SET status = ?, reply = ?, error = ?, finished_at = ?
                WHERE turn_id = ?;
                """,
                (TURN_FAILED, reply, reason, now_str, turn_id),
            )
            conn.commit()
            return True
    except (sqlite3.Error, ValueError):
        logger.exception("_reap_single_running_turn 失败 turn_id=%s", turn_id)
        return False


def get_pending_approval(turn_id: str) -> dict[str, Any] | None:
    """查询指定 turn 的审批记录。"""
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT turn_id, tool_name, call_id, args, summary, status FROM chat_approvals WHERE turn_id = ?;",
            (turn_id,),
        ).fetchone()
    if row is None:
        return None
    return {
        "turn_id": row["turn_id"],
        "tool_name": row["tool_name"],
        "call_id": row["call_id"],
        "args": json.loads(row["args"]),
        "summary": row["summary"],
        "status": row["status"],
    }


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
            "SELECT turn_id, session_id, user_message, status, ttft_ms, latency_ms FROM chat_turns WHERE turn_id = ?;",
            (turn_id,),
        ).fetchone()
    return dict(row) if row else None


def get_events_after(turn_id: str, after_seq: int = 0) -> list[dict[str, Any]]:
    """
    断线补发：返回 seq > after_seq 的全部事件，按 seq 升序。
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
