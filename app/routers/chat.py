"""
app/routers/chat.py

对话路由：
  POST /api/chat                          v1.0 同步接口（契约不变）
  POST /api/chat/stream                   v2.0 SSE 流式接口（事件先落库后推送）
  GET  /api/chat/turns/{turn_id}/events   v2.0 断线补发接口（按 after_seq 增量拉取）
"""

import asyncio
import json
import logging
import time
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from app.agent.core import arun_agent, stream_agent
from app.core.limiter import check_chat_rate_limit, check_daily_quota_limit
from app.db.events import create_turn, get_events_after, get_turn
from app.schemas.chat import ChatEvent, ChatRequest, ChatResponse, TurnEventsResponse

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
        "受双重限流保护：先执行单 IP 频次限流（默认 15 次/分/IP），"
        "再执行北京时间每日全局额度熔断（默认 200 次/天）。"
    ),
)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    核心对话接口（入参 / 出参 / 限流与 v1.0 完全一致）：
    1. 直接 await 异步版 arun_agent，复用服务主事件循环与 Provider 连接池
       （若仍走同步 run_agent，每个请求都要在线程池里新建一次事件循环）；
    2. 捕获意外异常，返回 HTTP 500 + 结构化日志，不暴露内部堆栈。
    """
    logger.info("POST /api/chat session_id=%s message='%s'", request.session_id, request.message[:80])

    try:
        reply, tools_called, latency_ms, citations = await arun_agent(
            session_id=request.session_id,
            user_message=request.message,
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
    """
    单条事件 → SSE 帧。id 使用 seq：浏览器 EventSource 重连时会自动带上 Last-Event-ID，
    客户端也可以把它作为 after_seq 调补发接口。json.dumps 默认不含换行，保证 data 只占一行。
    """
    data = json.dumps(event, ensure_ascii=False)
    return f"id: {event['seq']}\nevent: {event['type']}\ndata: {data}\n\n"


async def _sse_frames(events: AsyncIterator[dict[str, Any]]) -> AsyncIterator[str]:
    async for event in events:
        yield _format_sse(event)


@router.post(
    "/stream",
    summary="流式对话（SSE）",
    dependencies=[Depends(check_chat_rate_limit), Depends(check_daily_quota_limit)],
    response_class=StreamingResponse,
    description=(
        "以 text/event-stream 推送本轮事件：stage / tool_call / citation / token / done / error。"
        "每帧格式为 `id: <seq>`、`event: <type>`、`data: <json>`；事件先写入 SQLite 再推送，"
        "断线后可调用 GET /api/chat/turns/{turn_id}/events?after_seq=<最后收到的 id> 补发。"
        "响应头 X-Turn-Id 返回本轮 turn_id。限流策略与 POST /api/chat 相同。"
    ),
)
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """先同步建 turn（失败可直接返回 500），再把事件流包装为 SSE 响应。"""
    started_at = time.monotonic()  # ttft_ms / latency_ms 以进入路由为起点
    logger.info("POST /api/chat/stream session_id=%s message='%s'", request.session_id, request.message[:80])

    try:
        turn_id = await asyncio.to_thread(create_turn, request.session_id, request.message)
    except Exception as e:
        # 响应头尚未发出，此时失败仍可返回标准 HTTP 错误；流开始后的异常则以 error 事件表达
        logger.exception("POST /api/chat/stream 创建 turn 失败 session_id=%s: %s", request.session_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Agent 处理请求时发生内部错误，请稍后重试。",
        )

    events = stream_agent(request.session_id, request.message, turn_id=turn_id, started_at=started_at)
    return StreamingResponse(
        _sse_frames(events),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 关闭 Nginx 代理缓冲，否则 token 会被攒批后才下发
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
    # 必须先读状态、后读事件：done/error 与 status 在同一事务提交，
    # 若先读到 completed，则随后读事件一定包含终态事件；反过来读可能出现"已完成却缺 done"的窗口
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
