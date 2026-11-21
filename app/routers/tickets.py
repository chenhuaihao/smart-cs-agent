import datetime
import logging
import uuid
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query, status
from app.db.database import get_db_connection
from app.schemas.ticket import TicketCreate, TicketResponse

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

    if order_id:
        conditions.append("order_id = ?")
        params.append(order_id.strip())
    if ticket_status:
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
