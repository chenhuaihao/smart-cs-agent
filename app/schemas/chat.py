"""
app/schemas/chat.py

对话接口的 Pydantic v2 请求/响应模型。
"""

from typing import List
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """
    POST /api/chat 入参模型
    """
    session_id: str = Field(
        ...,
        min_length=1,
        description="会话唯一标识，同一用户同一会话保持一致，用于多轮上下文追踪",
    )
    message: str = Field(
        ...,
        min_length=1,
        description="用户本轮输入内容，不可为空字符串",
    )


class ChatResponse(BaseModel):
    """
    POST /api/chat 出参模型
    """
    session_id: str = Field(..., description="对应请求的会话标识")
    reply: str = Field(..., description="助手生成的自然语言回复")
    tools_called: List[str] = Field(
        default_factory=list,
        description="本轮实际调用的工具名列表，纯问答时为空列表",
    )
    latency_ms: int = Field(..., description="本次请求端到端耗时，单位毫秒")
