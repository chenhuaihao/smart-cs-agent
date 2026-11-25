"""
app/routers/chat.py

对话路由：
  POST /api/chat                          v1.0 同步接口（默认 auto_approve=True，契约 100% 不变）
  POST /api/chat/stream                   v2.0 SSE 流式接口（默认 auto_approve=False，高风险写工具触发 HITL）
  GET  /api/chat/turns/{turn_id}/events   v2.0 断线补发接口（按 after_seq 增量拉取，不限流）
  POST /api/chat/turns/{turn_id}/approve  v2.0 P1 高风险写操作人工确认接口（不挂载对话限流）
"""

import asyncio
import json
import logging
import time
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from app.agent.core import approve_turn_stream, arun_agent, stream_agent
from app.core.limiter import check_chat_rate_limit, check_daily_quota_limit
from app.db.events import (
    TURN_WAITING_APPROVAL,
    create_turn,
    get_events_after,
    get_turn,
    resolve_pending_approval,
)
from app.schemas.chat import (
    ApprovalRequest,
    ChatEvent,
    ChatRequest,
    ChatResponse,
    TurnEventsResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post(
    "",
    response_model=ChatResponse,
    summary="发送消息给智能客服 Agent",
    dependencies=[Depends(check_chat_rate_limit), Depends(check_daily_quota_limit)],
    description=(
        "接收用户消息，经过多轮 Function Calling 推理后返回助手回复。"
        "同一 session_id 的请求会共享对话历史（最近 6 轮）。"
        "若本轮调用了知识库检索且有命中，citations 字段返回溯源条款来源。"
        "默认开启 auto_approve=True，保持 v1.0 契约与自动化评测 100% 兼容。"
    ),
)
async def chat(request: ChatRequest) -> ChatResponse:
    """核心同步对话接口（入参 / 出参 / 限流与 v1.0 完全一致，默认 auto_approve=True）。"""
    logger.info("POST /api/chat session_id=%s message='%s'", request.session_id, request.message[:80])

    try:
        reply, tools_called, latency_ms, citations = await arun_agent(
            session_id=request.session_id,
            user_message=request.message,
            auto_approve=True,
        )
    except Exception as e:
        logger.exception("POST /api/chat 未预期异常 session_id=%s: %s", request.session_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Agent 处理请求时发生内部错误，请稍后重试。",
        )

    return ChatResponse(
        session_id=request.session_id,
        reply=reply,
        tools_called=tools_called,
        latency_ms=latency_ms,
        citations=citations,
    )


def _format_sse(event: dict[str, Any]) -> str:
    """单条事件 → SSE 帧。id 使用 seq，event 使用事件类型，data 为单行 JSON。"""
    data = json.dumps(event, ensure_ascii=False)
    return f"id: {event['seq']}\nevent: {event['type']}\ndata: {data}\n\n"


async def _sse_frames(events: AsyncIterator[dict[str, Any]]) -> AsyncIterator[str]:
    async for event in events:
        yield _format_sse(event)


@router.post(
    "/stream",
    summary="流式对话（SSE，支持 HITL 建单确认）",
    dependencies=[Depends(check_chat_rate_limit), Depends(check_daily_quota_limit)],
    response_class=StreamingResponse,
    description=(
        "以 text/event-stream 推送本轮事件：stage / tool_call / citation / token / approval_required / done / error。"
        "当模型决定调用高风险写工具 create_after_sales_ticket 时，暂停落库并推送 approval_required 事件，"
        "等待客户端调用 POST /api/chat/turns/{turn_id}/approve 进行确认。"
    ),
)
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """先同步建 turn，再把事件流包装为 SSE 响应（auto_approve=False）。"""
    started_at = time.monotonic()
    logger.info("POST /api/chat/stream session_id=%s message='%s'", request.session_id, request.message[:80])

    try:
        turn_id = await asyncio.to_thread(create_turn, request.session_id, request.message)
    except Exception as e:
        logger.exception("POST /api/chat/stream 创建 turn 失败 session_id=%s: %s", request.session_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Agent 处理请求时发生内部错误，请稍后重试。",
        )

    events = stream_agent(
        request.session_id,
        request.message,
        turn_id=turn_id,
        started_at=started_at,
        auto_approve=False,
    )
    return StreamingResponse(
        _sse_frames(events),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Turn-Id": turn_id,
        },
    )


@router.get(
    "/turns/{turn_id}/events",
    response_model=TurnEventsResponse,
    summary="断线补发：拉取 turn 内 seq > after_seq 的事件",
    description="不计入对话限流与每日额度（只读数据库，不触发模型调用）。",
)
def get_turn_events(
    turn_id: str,
    after_seq: int = Query(0, ge=0, description="客户端最后收到的 seq，0 表示从头拉取"),
) -> TurnEventsResponse:
    turn = get_turn(turn_id)
    if turn is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"turn_id={turn_id} 不存在")

    events = [ChatEvent(**e) for e in get_events_after(turn_id, after_seq)]
    return TurnEventsResponse(
        turn_id=turn_id,
        status=turn["status"],
        events=events,
        last_seq=events[-1].seq if events else None,
    )


@router.post(
    "/turns/{turn_id}/approve",
    summary="高风险写操作显式授权（HITL 确认）",
    response_class=StreamingResponse,
    description=(
        "对处于 waiting_approval 状态的 turn 执行人工授权决策（不挂载对话限流）："
        "approved=true 时真正执行 create_after_sales_ticket 落库并继续流式返回 tool_call / token / done 事件；"
        "approved=false 时取消建单不落库，告知用户已取消并正常结束该轮次。"
    ),
)
async def approve_turn(turn_id: str, body: ApprovalRequest) -> StreamingResponse:
    turn = await asyncio.to_thread(get_turn, turn_id)
    if turn is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"turn_id={turn_id} 不存在")
    if turn["status"] != TURN_WAITING_APPROVAL:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"turn_id={turn_id} 当前状态为 {turn['status']}，无需或已完成审批",
        )

    # 在发送 HTTP 200 流响应头前执行 SQLite BEGIN IMMEDIATE 原子 CAS 抢占审批权，
    # 彻底消除并发双击 TOCTOU 竞态：仅首个请求拿到 200 并建单，后续重复请求一律返回 409 Conflict。
    pending = await asyncio.to_thread(resolve_pending_approval, turn_id, body.approved)
    if pending is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"turn_id={turn_id} 已被其他请求审批或已超时过期，请勿重复提交",
        )

    events = approve_turn_stream(turn_id, body.approved, pending=pending)
    return StreamingResponse(
        _sse_frames(events),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Turn-Id": turn_id,
        },
    )
