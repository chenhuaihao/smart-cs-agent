import datetime
import logging
import uuid
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from app.core.limiter import (
    check_api_read_rate_limit,
    check_api_write_rate_limit,
    check_demo_admin_token,
)
from app.db.database import get_db_connection
from app.db.seed import reset_demo_data
from app.schemas.ticket import TicketCreate, TicketResponse, TicketStatusUpdate

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/tickets", tags=["tickets"])


def _generate_ticket_id() -> str:
    """
    生成全局唯一且可读的工单号：
    格式: TKT{YYYYMMDD}-{6位大写随机十六进制字符}，例如 TKT20261121-8F3E2A。
    通过时间前缀方便按日期归档索引，后置随机十六进制串避免高并发撞号。
    """
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    random_hex = uuid.uuid4().hex[:6].upper()
    return f"TKT{date_str}-{random_hex}"


@router.post(
    "",
    response_model=TicketResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(check_api_write_rate_limit)],
    summary="创建售后工单",
    description="为指定订单创建售后工单。前置校验关联订单是否存在，默认状态为'待处理'。"
)
def create_ticket(ticket_in: TicketCreate) -> TicketResponse:
    """
    创建工单核心流程：
    1. 校验 order_id 真实性（外键存在性校验）：若订单不存在直接返回 404，防止写入脏数据；
    2. 生成防重 ticket_id；
    3. 参数化原生 SQL 插入 tickets 表；
    4. 返回创建好的工单出参模型。
    """
    clean_order_id = ticket_in.order_id.strip()

    with get_db_connection() as conn:
        cursor = conn.cursor()

        # 1. 严格校验关联订单是否存在
        cursor.execute("SELECT 1 FROM orders WHERE order_id = ?;", (clean_order_id,))
        if not cursor.fetchone():
            logger.warning("创建工单失败：关联订单 %s 不存在", clean_order_id)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"关联订单 {clean_order_id} 不存在，无法创建售后工单"
            )

        # 2. 构造工单实体数据
        ticket_id = _generate_ticket_id()
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        initial_status = "待处理"

        insert_sql = """
        INSERT INTO tickets (ticket_id, order_id, issue_type, reason, status, summary, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?);
        """
        cursor.execute(
            insert_sql,
            (
                ticket_id,
                clean_order_id,
                ticket_in.issue_type,
                ticket_in.reason,
                initial_status,
                ticket_in.summary,
                now_str,
            ),
        )
        conn.commit()
        logger.info("成功创建工单：%s，关联订单：%s", ticket_id, clean_order_id)

    return TicketResponse(
        ticket_id=ticket_id,
        order_id=clean_order_id,
        issue_type=ticket_in.issue_type,
        reason=ticket_in.reason,
        status=initial_status,
        summary=ticket_in.summary,
        created_at=now_str,
    )


@router.get(
    "",
    response_model=List[TicketResponse],
    dependencies=[Depends(check_api_read_rate_limit)],
    summary="查询工单列表",
    description="获取工单记录，支持按关联订单号（order_id）或处理状态（status）进行筛选。"
)
def list_tickets(
    order_id: Optional[str] = Query(None, description="按关联订单单号过滤"),
    ticket_status: Optional[str] = Query(None, alias="status", description="按工单处理状态过滤（如：待处理、处理中、已结单）")
) -> List[TicketResponse]:
    """
    工单列表多维度查询：
    支持按 order_id 和 status 动态条件筛选，使用参数化列表动态绑定防注入。
    """
    conditions = []
    params = []

    if order_id and isinstance(order_id, str):
        conditions.append("order_id = ?")
        params.append(order_id.strip())
    if ticket_status and isinstance(ticket_status, str):
        conditions.append("status = ?")
        params.append(ticket_status.strip())

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    query_sql = f"""
    SELECT ticket_id, order_id, issue_type, reason, status, summary, created_at
    FROM tickets
    {where_clause}
    ORDER BY created_at DESC;
    """

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query_sql, tuple(params))
        rows = cursor.fetchall()

    return [TicketResponse.model_validate(dict(row)) for row in rows]


# 合法流转状态集合（支持中文及常见别名映射，确保数据库与展示一致）
_STATUS_ALIAS_MAP = {
    "open": "待处理",
    "pending": "待处理",
    "待处理": "待处理",
    "processing": "处理中",
    "in_progress": "处理中",
    "处理中": "处理中",
    "completed": "已完成",
    "done": "已完成",
    "resolved": "已完成",
    "已完成": "已完成",
    "已结单": "已结单",
    "已关闭": "已关闭",
}


# 严格单向状态流转规则：待处理 → 处理中/已完成；处理中 → 已完成；已完成/已结单/已关闭 为终态，严禁逆向流转
_ALLOWED_TRANSITIONS: Dict[str, set[str]] = {
    "待处理": {"处理中", "已完成"},
    "处理中": {"已完成"},
    "已完成": set(),
    "已结单": set(),
    "已关闭": set(),
}


@router.patch(
    "/{ticket_id}",
    response_model=TicketResponse,
    dependencies=[Depends(check_api_write_rate_limit)],
    summary="更新工单处理状态（严格单向状态机）",
    description="仅允许合法单向流转：待处理 → 处理中 / 已完成，处理中 → 已完成；已完成为终态严禁逆向流转（非法流转返回 400）。"
)
def update_ticket_status(ticket_id: str, payload: TicketStatusUpdate) -> TicketResponse:
    """
    工单状态流转更新：
    1. 校验目标状态合法性（兼容中英文字符映射）；
    2. 校验 ticket_id 存在性并读取当前状态，不存在返回 404；
    3. 校验严格单向状态机，拒绝终态逆向流转或同态重复流转（返回 400）；
    4. 参数化 UPDATE 原生 SQL 并返回最新工单完整出参。
    """
    raw_status = payload.status.strip()
    target_status = _STATUS_ALIAS_MAP.get(raw_status.lower(), _STATUS_ALIAS_MAP.get(raw_status))
    if not target_status:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"非法工单状态 '{raw_status}'，支持的状态包括：待处理、处理中、已完成、已结单、已关闭"
        )

    clean_ticket_id = ticket_id.strip()

    with get_db_connection() as conn:
        cursor = conn.cursor()
        # 1. 检查工单是否存在并获取当前状态
        cursor.execute("SELECT status FROM tickets WHERE ticket_id = ?;", (clean_ticket_id,))
        existing = cursor.fetchone()
        if not existing:
            logger.warning("更新工单状态失败：工单 %s 不存在", clean_ticket_id)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"工单 {clean_ticket_id} 不存在"
            )

        current_status: str = str(existing["status"]).strip()
        allowed_targets = _ALLOWED_TRANSITIONS.get(current_status, set())
        if target_status not in allowed_targets:
            if not allowed_targets:
                msg = f"工单 {clean_ticket_id} 当前已处于终态「{current_status}」，严禁逆向流转至「{target_status}」"
            else:
                msg = (
                    f"非法状态流转：工单 {clean_ticket_id} 当前为「{current_status}」，"
                    f"仅允许单向流转至 {', '.join(sorted(allowed_targets))}，不允许流转至「{target_status}」"
                )
            logger.warning("拒绝非法工单状态流转：%s (%s -> %s)", clean_ticket_id, current_status, target_status)
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=msg)

        # 2. 原生参数化更新
        cursor.execute(
            "UPDATE tickets SET status = ? WHERE ticket_id = ?;",
            (target_status, clean_ticket_id),
        )
        conn.commit()

        # 3. 查出最新数据返回
        cursor.execute(
            """
            SELECT ticket_id, order_id, issue_type, reason, status, summary, created_at
            FROM tickets
            WHERE ticket_id = ?;
            """,
            (clean_ticket_id,),
        )
        row = cursor.fetchone()

    logger.info("工单状态流转成功：%s (%s -> %s)", clean_ticket_id, current_status, target_status)
    return TicketResponse.model_validate(dict(row))


@router.post(
    "/reset",
    dependencies=[Depends(check_api_write_rate_limit), Depends(check_demo_admin_token)],
    summary="重置演示订单与工单数据",
    description="一键清空并重建 24 条标准演示订单与 3 条多状态演示工单（若配置 DEMO_ADMIN_TOKEN 则需校验口令）。"
)
def reset_demo_tickets() -> Dict[str, Any]:
    """
    一键重置演示数据接口：
    同步清空并重建 orders（24 条）与 tickets（3 条），防止演示环境脏数据累积。
    """
    orders_count, tickets_count = reset_demo_data()
    logger.info(
        "调用 /api/tickets/reset 重置演示数据，恢复 %d 条标准订单与 %d 条演示工单",
        orders_count,
        tickets_count,
    )
    return {
        "message": f"演示数据已重置（{orders_count} 条标准订单 + {tickets_count} 条演示工单）",
        "orders_count": orders_count,
        "count": tickets_count,
    }

