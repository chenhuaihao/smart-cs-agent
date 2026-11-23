import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from app.core.limiter import check_api_read_rate_limit
from app.db.database import get_db_connection
from app.schemas.order import OrderResponse

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/orders",
    tags=["orders"],
    dependencies=[Depends(check_api_read_rate_limit)],
)


@router.get(
    "/{order_id}",
    response_model=OrderResponse,
    summary="按单号查询订单详情",
    description="根据订单唯一单号获取订单完整详情，若单号不存在则明确返回 404。"
)
def get_order_by_id(order_id: str) -> OrderResponse:
    """
    订单精准查询：
    使用原生 SQL 参数化语句防 SQL 注入，通过 row_factory 字典方式提取字段。
    """
    query_sql = """
    SELECT order_id, user_name, product_name, amount, status, tracking_no, created_at
    FROM orders
    WHERE order_id = ?;
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query_sql, (order_id.strip(),))
        row = cursor.fetchone()

    if not row:
        logger.warning("查询订单失败：未找到单号为 %s 的订单", order_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"未找到订单号为 {order_id} 的订单信息"
        )

    return OrderResponse.model_validate(dict(row))


@router.get(
    "",
    response_model=List[OrderResponse],
    summary="获取订单列表",
    description="获取所有订单列表，支持按订单履约状态（status）进行筛选。"
)
def list_orders(
    order_status: Optional[str] = Query(None, alias="status", description="订单状态筛选（如：待发货、运输中、已签收、退款中）")
) -> List[OrderResponse]:
    """
    订单列表查询：
    支持按履约状态过滤，默认按下单时间倒序展示。
    采用参数化构造 SQL，避免直接字符串拼接带来的安全风险。
    """
    if order_status:
        query_sql = """
        SELECT order_id, user_name, product_name, amount, status, tracking_no, created_at
        FROM orders
        WHERE status = ?
        ORDER BY created_at DESC;
        """
        params = (order_status.strip(),)
    else:
        query_sql = """
        SELECT order_id, user_name, product_name, amount, status, tracking_no, created_at
        FROM orders
        ORDER BY created_at DESC;
        """
        params = ()

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query_sql, params)
        rows = cursor.fetchall()

    return [OrderResponse.model_validate(dict(row)) for row in rows]
