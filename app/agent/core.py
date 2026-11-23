"""
app/agent/core.py

Agent 核心调度（v2.0 P1）：
  - 消费 LangGraph 事件流 → 统一事件协议 → 先落库后推送；
  - 集成整轮超时护栏（TURN_TIMEOUT_SEC），超时后安全取消任务并走降级收尾，不抛裸 500；
  - 集成高风险写工具 HITL 挂起与恢复（approval_required → POST /turns/{turn_id}/approve）；
  - 保持 POST /api/chat 与 run_agent(session_id, user_message) 默认 auto_approve=True 向后兼容。
"""

import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncIterator

from app.agent.graph import (
    LLM_ERROR_REPLY,
    MAX_STEPS_FALLBACK_REPLY,
    RECURSION_LIMIT,
    agent_graph,
    merge_citations,
)
from app.agent.guardrails import (
    EvidenceLedger,
    build_fallback_reply,
    get_max_steps,
    get_step_timeout_sec,
    get_turn_timeout_sec,
)
from app.agent.memory import load_history, save_message
from app.agent.prompts import build_system_prompt
from app.agent.tools import execute_tool
from app.db import events as event_store
from app.llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)

_FORWARD_EVENTS = {"stage", "tool_call", "citation", "token"}
_background_tasks: set["asyncio.Task[None]"] = set()


class _ApprovalWaitContext:
    """进程内活跃 turn 的 HITL 挂起同步上下文（与 SQLite chat_approvals 持久化双写配合）。"""

    def __init__(self, queue: "asyncio.Queue[dict[str, Any] | None]") -> None:
        self.queue = queue
        self.event = asyncio.Event()
        self.approved: bool = False
        self.done_event = asyncio.Event()


_approval_waiters: dict[str, _ApprovalWaitContext] = {}


def _elapsed_ms(started_at: float) -> int:
    return int((time.monotonic() - started_at) * 1000)


async def _emit(turn_id: str, event_type: str, data: dict[str, Any]) -> dict[str, Any]:
    """先落库拿 seq，再返回完整事件；SQLite 写入放线程池，避免阻塞事件循环。"""
    return await asyncio.to_thread(event_store.append_event, turn_id, event_type, data)


def _build_approval_summary(tool_name: str, args: dict[str, Any]) -> str:
    """生成可读的待确认摘要。"""
    if tool_name == "create_after_sales_ticket":
        oid = args.get("order_id", "未知订单")
        itype = args.get("issue_type", "售后")
        summary = args.get("summary") or args.get("reason") or ""
        return f"即将为订单 {oid} 创建【{itype}】售后工单：{summary}"
    return f"请求执行高风险工具 {tool_name}"


async def _consume_graph_events(
    turn_id: str,
    initial_state: dict[str, Any],
    config: dict[str, Any],
    started_at: float,
    wait_ctx: _ApprovalWaitContext,
) -> tuple[dict[str, Any], list[str], list[str], int | None]:
    """消费图事件并实时落库入队（通过 wait_ctx.queue 动态路由到当前活跃的 SSE 队列）。"""
    ttft_ms: int | None = None
    tools_called: list[str] = []
    citations: list[str] = []
    final_state: dict[str, Any] = {}

    async for ev in agent_graph.astream_events(initial_state, config=config, version="v2"):
        if ev["event"] == "on_custom_event" and ev["name"] in _FORWARD_EVENTS:
            name, data = ev["name"], ev["data"]
            if name == "token" and ttft_ms is None:
                ttft_ms = _elapsed_ms(started_at)
            elif name == "tool_call" and data.get("status") != "cancelled":
                tools_called.append(data["name"])
            elif name == "citation":
                citations = merge_citations(citations, data["citations"])
            await wait_ctx.queue.put(await _emit(turn_id, name, data))
        elif ev["event"] == "on_chain_end" and not ev.get("parent_ids"):
            final_state = ev["data"].get("output") or {}

    return final_state, tools_called, citations, ttft_ms


async def _run_turn(
    turn_id: str,
    session_id: str,
    user_message: str,
    started_at: float,
    queue: "asyncio.Queue[dict[str, Any] | None]",
    provider: BaseLLMProvider | None,
    auto_approve: bool,
) -> None:
    """
    生产者：执行整轮推理并把"已落库"的事件放进队列。
    支持 HITL 暂停：当 auto_approve=False 且触发高风险写工具时，写入 approval_required 并挂起，
    同时向当前 SSE 流发送哨兵结束首段响应（或由 /approve 接口唤醒继续产出后续事件）。
    """
    ttft_ms: int | None = None
    tools_called: list[str] = []
    citations: list[str] = []
    final_state: dict[str, Any] = {}
    status, final_type, error = event_store.TURN_COMPLETED, "done", None
    cancelled = False
    turn_timeout = get_turn_timeout_sec()
    wait_ctx = _ApprovalWaitContext(queue)
    _approval_waiters[turn_id] = wait_ctx

    async def _approval_handler(call: dict[str, Any]) -> bool:
        raw_args = call.get("arguments", "{}")
        args = raw_args if isinstance(raw_args, dict) else json.loads(raw_args or "{}")
        summary = _build_approval_summary(call["name"], args)
        record = await asyncio.to_thread(
            event_store.save_pending_approval,
            turn_id,
            call["name"],
            call["id"],
            args,
            summary,
        )
        await wait_ctx.queue.put(record)
        # 发出 approval_required 后向当前 SSE 流推送结束哨兵，使客户端立即收到事件并展示确认按钮；
        # 后台图任务保持在 wait_ctx.event.wait() 上挂起，等 /approve 唤醒后继续走完剩余节点
        await wait_ctx.queue.put(None)
        await wait_ctx.event.wait()
        return wait_ctx.approved

    try:
        await queue.put(await _emit(turn_id, "stage", {"stage": "received", "session_id": session_id}))

        history = await asyncio.to_thread(load_history, session_id)
        messages = [
            {"role": "system", "content": build_system_prompt()},
            *history,
            {"role": "user", "content": user_message},
        ]
        config: dict[str, Any] = {
            "recursion_limit": RECURSION_LIMIT,
            "configurable": {
                "auto_approve": auto_approve,
                "approval_handler": _approval_handler,
                "max_steps": get_max_steps(),
                "step_timeout_sec": get_step_timeout_sec(),
            },
        }
        if provider is not None:
            config["configurable"]["provider"] = provider

        initial_state = {
            "messages": messages,
            "steps": 0,
            "retry_count": 0,
            "evidence_ledger": EvidenceLedger(),
        }

        final_state, tools_called, citations, ttft_ms = await asyncio.wait_for(
            _consume_graph_events(turn_id, initial_state, config, started_at, wait_ctx),
            timeout=turn_timeout,
        )

        reply = final_state.get("final_reply")
        if reply is None:
            reply = MAX_STEPS_FALLBACK_REPLY
        tools_called = final_state.get("tools_called", tools_called)
        citations = final_state.get("citations", citations)
    except asyncio.TimeoutError:
        logger.warning("turn 整轮超时 turn_id=%s timeout=%ss", turn_id, turn_timeout)
        reason = f"整轮对话处理超时（>{turn_timeout}s）"
        reply = build_fallback_reply(reason)
        await wait_ctx.queue.put(await _emit(turn_id, "stage", {"stage": "fallback", "reason": reason}))
        await wait_ctx.queue.put(await _emit(turn_id, "token", {"delta": reply, "step": 0, "fallback": True}))
        if ttft_ms is None:
            ttft_ms = _elapsed_ms(started_at)
    except asyncio.CancelledError:
        cancelled = True
        reply, status, final_type, error = (
            build_fallback_reply("请求任务被取消"),
            event_store.TURN_FAILED,
            "error",
            "cancelled",
        )
    except Exception as e:
        logger.exception("turn 执行异常，进入降级收尾 turn_id=%s session_id=%s", turn_id, session_id)
        reason = "后端模型或依赖服务暂时不可用"
        reply = build_fallback_reply(reason)
        status, final_type = event_store.TURN_COMPLETED, "done"
        error = f"{type(e).__name__}: {e}"
        try:
            await wait_ctx.queue.put(await _emit(turn_id, "stage", {"stage": "fallback", "reason": reason}))
            await wait_ctx.queue.put(await _emit(turn_id, "token", {"delta": reply, "step": 0, "fallback": True}))
        except Exception:
            logger.exception("写入异常降级事件失败 turn_id=%s", turn_id)

    latency_ms = _elapsed_ms(started_at)
    try:
        await asyncio.to_thread(save_message, session_id, "user", user_message)
        await asyncio.to_thread(
            save_message,
            session_id,
            "assistant",
            reply,
            ",".join(tools_called) if tools_called else None,
            latency_ms,
        )
        payload: dict[str, Any] = {
            "reply": reply,
            "tools_called": tools_called,
            "citations": citations,
            "ttft_ms": ttft_ms,
            "latency_ms": latency_ms,
        }
        if final_type == "error":
            payload["message"] = "Agent 处理请求时发生内部错误，请稍后重试。"
        record = await asyncio.to_thread(
            event_store.finish_turn,
            turn_id,
            status,
            final_type,
            payload,
            reply,
            tools_called,
            citations,
            ttft_ms,
            latency_ms,
            error,
        )
        await wait_ctx.queue.put(record)
    except Exception:
        logger.exception("终态事件落库失败 turn_id=%s", turn_id)
    finally:
        wait_ctx.done_event.set()
        await wait_ctx.queue.put(None)
        _approval_waiters.pop(turn_id, None)
        logger.info(
            "turn 结束 turn_id=%s status=%s tools=%s citations=%s ttft_ms=%s latency_ms=%d",
            turn_id, status, tools_called, citations, ttft_ms, latency_ms,
        )
    if cancelled:
        raise asyncio.CancelledError()


async def approve_turn_stream(turn_id: str, approved: bool) -> AsyncIterator[dict[str, Any]]:
    """
    处理高风险写操作 HITL 审批决策（POST /api/chat/turns/{turn_id}/approve）：
      - 优先唤醒内存中挂起的图任务，将后续 tool_call / token / done 事件以流式产出；
      - 若挂起任务不在当前进程内存（例如跨重启恢复），则直接基于 SQLite chat_approvals 记录执行/取消并收尾。
    """
    started_at = time.monotonic()
    pending = await asyncio.to_thread(event_store.resolve_pending_approval, turn_id, approved)
    if pending is None:
        raise ValueError(f"turn_id={turn_id} 当前不在 waiting_approval 状态或不存在待审批记录")

    wait_ctx = _approval_waiters.get(turn_id)
    if wait_ctx is not None:
        resume_queue: "asyncio.Queue[dict[str, Any] | None]" = asyncio.Queue()
        wait_ctx.queue = resume_queue
        wait_ctx.approved = approved
        wait_ctx.event.set()
        while (item := await resume_queue.get()) is not None:
            yield item
        return

    # 兜底：若内存中无挂起协程，直接基于 SQLite 记录执行工具并收尾
    turn = await asyncio.to_thread(event_store.get_turn, turn_id)
    session_id = (turn or {}).get("session_id", "")
    user_message = (turn or {}).get("user_message", "")
    tool_name = pending["tool_name"]
    call_id = pending["call_id"]
    args = pending["args"]

    tools_called: list[str] = []
    if approved:
        t0 = time.monotonic()
        res_str = await asyncio.to_thread(execute_tool, tool_name, args)
        dur = int((time.monotonic() - t0) * 1000)
        res_obj = json.loads(res_str)
        tools_called.append(tool_name)
        yield await _emit(
            turn_id,
            "tool_call",
            {"call_id": call_id, "name": tool_name, "arguments": args, "status": "success", "duration_ms": dur},
        )
        tid = res_obj.get("ticket_id", "")
        reply = f"已为您成功创建售后工单（工单号：{tid}，状态：待处理），售后专员将尽快为您跟进。"
    else:
        yield await _emit(
            turn_id,
            "tool_call",
            {"call_id": call_id, "name": tool_name, "arguments": args, "status": "cancelled", "duration_ms": 0},
        )
        reply = "已按照您的指示取消创建售后工单。如需其他帮助，请随时告诉我。"

    yield await _emit(turn_id, "token", {"delta": reply, "step": 1})
    latency_ms = _elapsed_ms(started_at)
    await asyncio.to_thread(save_message, session_id, "user", user_message)
    await asyncio.to_thread(
        save_message, session_id, "assistant", reply, ",".join(tools_called) if tools_called else None, latency_ms
    )
    done_payload = {
        "reply": reply,
        "tools_called": tools_called,
        "citations": [],
        "ttft_ms": latency_ms,
        "latency_ms": latency_ms,
    }
    yield await asyncio.to_thread(
        event_store.finish_turn,
        turn_id,
        event_store.TURN_COMPLETED,
        "done",
        done_payload,
        reply,
        tools_called,
        [],
        latency_ms,
        latency_ms,
        None,
    )


async def stream_agent(
    session_id: str,
    user_message: str,
    turn_id: str | None = None,
    started_at: float | None = None,
    provider: BaseLLMProvider | None = None,
    auto_approve: bool = False,
) -> AsyncIterator[dict[str, Any]]:
    """
    流式执行一轮对话，逐条产出已落库的事件。
    流式模式默认 auto_approve=False（对高风险写操作启用 HITL 显式授权）。
    """
    started_at = started_at if started_at is not None else time.monotonic()
    if turn_id is None:
        turn_id = await asyncio.to_thread(event_store.create_turn, session_id, user_message)

    queue: "asyncio.Queue[dict[str, Any] | None]" = asyncio.Queue()
    task = asyncio.create_task(
        _run_turn(turn_id, session_id, user_message, started_at, queue, provider, auto_approve)
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    try:
        while (item := await queue.get()) is not None:
            yield item
    finally:
        if not task.done():
            logger.info("流阶段结束或客户端断开 turn_id=%s，任务在后台保持或等待 HITL 确认", turn_id)


async def arun_agent(
    session_id: str,
    user_message: str,
    provider: BaseLLMProvider | None = None,
    auto_approve: bool = True,
) -> tuple[str, list[str], int, list[str]]:
    """
    异步收集版：默认开启 auto_approve=True，保持 POST /api/chat 与 eval/run_eval.py 100% 兼容不中断。
    """
    started_at = time.monotonic()
    final: dict[str, Any] | None = None
    async for event in stream_agent(
        session_id,
        user_message,
        started_at=started_at,
        provider=provider,
        auto_approve=auto_approve,
    ):
        if event["type"] in ("done", "error"):
            final = event["data"]

    if final is None:
        return LLM_ERROR_REPLY, [], _elapsed_ms(started_at), []
    return final["reply"], final["tools_called"], final["latency_ms"], final["citations"]


def run_agent(session_id: str, user_message: str) -> tuple[str, list[str], int, list[str]]:
    """v1.0 同步兼容入口：默认 auto_approve=True，签名与返回值保持 4 元组不变。"""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(arun_agent(session_id, user_message, auto_approve=True))

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, arun_agent(session_id, user_message, auto_approve=True)).result()
