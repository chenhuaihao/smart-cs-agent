"""
app/schemas/chat.py

对话接口的 Pydantic v2 请求/响应模型。
POST /api/chat/stream 复用 ChatRequest 作为入参；SSE 帧的 data 与 ChatEvent 结构一致。
"""

from typing import Any, List, Literal, Optional
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
    citations: List[str] = Field(
        default_factory=list,
        description=(
            "本轮命中知识库条款的去重来源列表，格式为 '文件名#章节名'，"
            "如 'return_policy.md#耳机类退货规则'。"
            "未调用知识库或无命中时为空列表。"
        ),
    )


# ---------------------------------------------------------------------------
# v2.0 流式事件协议
# ---------------------------------------------------------------------------

EventType = Literal["stage", "tool_call", "citation", "token", "done", "error"]


class ChatEvent(BaseModel):
    """
    单条对话事件。SSE 推送的 data 与断线补发接口返回的元素是同一结构，
    客户端可以用同一套解析逻辑处理实时流与补发数据。
    """
    turn_id: str = Field(..., description="本轮请求唯一标识")
    seq: int = Field(..., ge=1, description="turn 内严格递增的事件序号，从 1 开始；即 SSE 的 id 字段")
    type: EventType = Field(..., description="事件类型")
    data: dict[str, Any] = Field(default_factory=dict, description="事件负载，结构随 type 变化")
    created_at: str = Field(..., description="事件落库时间（毫秒精度）")


class TurnEventsResponse(BaseModel):
    """
    GET /api/chat/turns/{turn_id}/events 出参模型
    """
    turn_id: str = Field(..., description="本轮请求唯一标识")
    status: Literal["running", "completed", "failed"] = Field(
        ..., description="turn 当前状态；running 表示仍在生成，客户端可稍后带新的 after_seq 继续拉取"
    )
    events: List[ChatEvent] = Field(default_factory=list, description="seq > after_seq 的事件，按 seq 升序")
    last_seq: Optional[int] = Field(
        None, description="本次返回的最大 seq，下次补发可直接作为 after_seq；无新事件时为 None"
    )
