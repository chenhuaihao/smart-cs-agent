"""
app/routers/chat.py

对话路由：POST /api/chat
将用户消息交给 Agent 核心处理，返回结构化的 ChatResponse。
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, status
from app.agent.core import run_agent
from app.core.limiter import check_chat_rate_limit
from app.schemas.chat import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post(
    "",
    response_model=ChatResponse,
    summary="发送消息给智能客服 Agent",
    dependencies=[Depends(check_chat_rate_limit)],
    description=(
        "接收用户消息，经过多轮 Function Calling 推理后返回助手回复。"
        "同一 session_id 的请求会共享对话历史（最近 6 轮）。"
        "若本轮调用了知识库检索且有命中，citations 字段返回溯源条款来源。"
        "受内存滑动窗口限流保护（默认 15 次/分/IP）。"
    ),
)
def chat(request: ChatRequest) -> ChatResponse:
    """
    核心对话接口：
    1. 调用 run_agent 执行推理（含 Function Calling 循环）
    2. 捕获意外异常，返回 HTTP 500 + 结构化日志，不暴露内部堆栈
    """
    logger.info(
        "POST /api/chat session_id=%s message='%s'",
        request.session_id,
        request.message[:80],  # 日志截断，避免过长输出
    )

    try:
        # run_agent 现在返回 4 元组（第三阶段新增 citations）
        reply, tools_called, latency_ms, citations = run_agent(
            session_id=request.session_id,
            user_message=request.message,
        )
    except Exception as e:
        # 未预期的顶层异常：记录完整堆栈，对外返回 500 而非暴露细节
        logger.exception(
            "POST /api/chat 未预期异常 session_id=%s: %s",
            request.session_id, e,
        )
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
