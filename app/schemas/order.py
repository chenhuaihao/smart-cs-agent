from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class OrderResponse(BaseModel):
    """
    订单数据出参模型 (Pydantic v2)
    """
    model_config = ConfigDict(from_attributes=True)

    order_id: str = Field(..., description="订单唯一编号，例如 ORD20261101")
    user_name: str = Field(..., description="下单用户姓名")
    product_name: str = Field(..., description="商品名称与规格")
    amount: float = Field(..., description="订单实付金额（元）")
    status: str = Field(..., description="订单履约状态（如：待发货、运输中、派送中、已签收、退款中等）")
    tracking_no: Optional[str] = Field(None, description="物流单号，未发货时可为空")
    created_at: str = Field(..., description="订单创建时间，格式 YYYY-MM-DD HH:MM:SS")
