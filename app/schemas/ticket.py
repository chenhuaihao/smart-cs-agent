from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict, Field

# 受控工单问题类型枚举，限制客户端只能传入合法业务场景
IssueType = Literal["退货", "换货", "物流催单", "售后咨询", "转人工"]


class TicketCreate(BaseModel):
    """
    创建工单入参模型 (Pydantic v2)
    """
    order_id: str = Field(..., min_length=1, description="关联订单编号，如 ORD20261101")
    issue_type: IssueType = Field(..., description="工单类型（限制：退货、换货、物流催单、售后咨询、转人工）")
    reason: Optional[str] = Field(None, max_length=500, description="申请原因详细阐述")
    summary: Optional[str] = Field(None, max_length=200, description="工单简述或处理备注")


class TicketResponse(BaseModel):
    """
    工单出参模型 (Pydantic v2)
    """
    model_config = ConfigDict(from_attributes=True)

    ticket_id: str = Field(..., description="工单编号，格式为 TKT{YYYYMMDD}-{6位大写随机十六进制}")
    order_id: str = Field(..., description="关联订单编号")
    issue_type: str = Field(..., description="工单类型")
    reason: Optional[str] = Field(None, description="申请原因")
    status: str = Field(default="待处理", description="工单处理状态（默认为'待处理'）")
    summary: Optional[str] = Field(None, description="工单概要")
    created_at: str = Field(..., description="工单创建时间，格式 YYYY-MM-DD HH:MM:SS")
