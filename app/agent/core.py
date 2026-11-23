"""
app/agent/core.py

Agent 核心调度（v2.0）：消费 LangGraph 事件流 → 统一事件协议 → 先落库后推送。

统一事件协议（每条事件都先写入 chat_events 拿到 seq，再交给调用方）：
  stage     阶段变化：received / thinking / tool_running
  tool_call 单个工具执行完成：name / arguments / status / duration_ms
  citation  新命中的知识库条款："文件名#章节名" 列表
  token     模型文本增量（以 done.reply 为最终权威文本）
  done      终态-成功：reply / tools_called / citations / ttft_ms / latency_ms
  error     终态-失败：message + 兜底 reply，同样携带耗时指标

对外入口：
  stream_agent(...)  异步生成器，供 SSE 路由逐条推送
  arun_agent(...)    异步收集版，返回 v1.0 同款 4 元组
  run_agent(...)     同步兼容入口，签名与返回值与 v1.0 完全一致
"""

import asyncio
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
from app.agent.memory import load_history, save_message
from app.agent.prompts import build_system_prompt
from app.db import events as event_store
from app.llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)

# 图内允许透传给客户端的自定义事件；其他内部事件一律丢弃，避免协议被意外扩散
_FORWARD_EVENTS = {"stage", "tool_call", "citation", "token"}
# 后台 turn 任务的强引用：asyncio 只持有 Task 弱引用，不保存会被 GC 提前回收
_background_tasks: set["asyncio.Task[None]"] = set()


def _elapsed_ms(started_at: float) -> int:
    return int((time.monotonic() - started_at) * 1000)


async def _emit(turn_id: str, event_type: str, data: dict[str, Any]) -> dict[str, Any]:
    """先落库拿 seq，再返回完整事件；SQLite 写入放线程池，避免阻塞事件循环。"""
    return await asyncio.to_thread(event_store.append_event, turn_id, event_type, data)


async def _run_turn(
    turn_id: str,
    session_id: str,
    user_message: str,
    started_at: float,
    queue: "asyncio.Queue[dict[str, Any] | None]",
    provider: BaseLLMProvider | None,
) -> None:
    """
    生产者：执行整轮推理并把"已落库"的事件放进队列。
    作为独立后台任务运行——客户端断开不会中断推理，剩余事件照常落库，重连后可用 after_seq 补发。
    """
    ttft_ms: int | None = None
    tools_called: list[str] = []
    citations: list[str] = []
    final_state: dict[str, Any] = {}
    status, final_type, error = event_store.TURN_COMPLETED, "done", None
    cancelled = False

    try:
        await queue.put(await _emit(turn_id, "stage", {"stage": "received", "session_id": session_id}))

        history = await asyncio.to_thread(load_history, session_id)
        messages = [{"role": "system", "content": build_system_prompt()}, *history,
                    {"role": "user", "content": user_message}]
        config: dict[str, Any] = {"recursion_limit": RECURSION_LIMIT, "configurable": {}}
        if provider is not None:
            config["configurable"]["provider"] = provider

        async for ev in agent_graph.astream_events({"messages": messages, "steps": 0}, config=config, version="v2"):
            if ev["event"] == "on_custom_event" and ev["name"] in _FORWARD_EVENTS:
                name, data = ev["name"], ev["data"]
                if name == "token" and ttft_ms is None:
                    ttft_ms = _elapsed_ms(started_at)  # 首 token 相对请求开始的耗时
                elif name == "tool_call":
                    tools_called.append(data["name"])
                elif name == "citation":
                    citations = merge_citations(citations, data["citations"])
                await queue.put(await _emit(turn_id, name, data))
            elif ev["event"] == "on_chain_end" and not ev.get("parent_ids"):
                # parent_ids 为空 = 根图结束，output 即最终 AgentState
                final_state = ev["data"].get("output") or {}

        reply = final_state.get("final_reply")
        if reply is None:
            reply = MAX_STEPS_FALLBACK_REPLY  # 理论不可达：图结束却无回复时兜底，绝不返回 None
        tools_called = final_state.get("tools_called", tools_called)
        citations = final_state.get("citations", citations)
    except asyncio.CancelledError:
        # 进程关闭等场景任务被取消：仍要把 turn 标记为 failed，避免补发方永远看到 running
        cancelled = True
        reply, status, final_type, error = LLM_ERROR_REPLY, event_store.TURN_FAILED, "error", "cancelled"
    except Exception as e:
        # 模型调用失败 / 图执行异常：与 v1.0 一致返回友好兜底文案，详细堆栈只进日志
        logger.exception("turn 执行失败 turn_id=%s session_id=%s", turn_id, session_id)
        reply, status, final_type = LLM_ERROR_REPLY, event_store.TURN_FAILED, "error"
        error = f"{type(e).__name__}: {e}"

    latency_ms = _elapsed_ms(started_at)
    try:
        # 与 v1.0 一致：成功或兜底都写入 chat_logs，保证同 session 多轮记忆连续
        await asyncio.to_thread(save_message, session_id, "user", user_message)
        await asyncio.to_thread(
            save_message, session_id, "assistant", reply,
            ",".join(tools_called) if tools_called else None, latency_ms,
        )
        payload: dict[str, Any] = {
            "reply": reply, "tools_called": tools_called, "citations": citations,
            "ttft_ms": ttft_ms, "latency_ms": latency_ms,
        }
        if final_type == "error":
            payload["message"] = "Agent 处理请求时发生内部错误，请稍后重试。"  # 不向客户端暴露异常细节
        record = await asyncio.to_thread(
            event_store.finish_turn, turn_id, status, final_type, payload,
            reply, tools_called, citations, ttft_ms, latency_ms, error,
        )
        await queue.put(record)
    except Exception:
        # 终态落库失败时不能推送"未落库"的事件（违背先写后推约定），只记录日志并结束流
        logger.exception("终态事件落库失败 turn_id=%s", turn_id)
    finally:
        await queue.put(None)  # 流结束哨兵
        logger.info(
            "turn 结束 turn_id=%s status=%s tools=%s citations=%s ttft_ms=%s latency_ms=%d",
            turn_id, status, tools_called, citations, ttft_ms, latency_ms,
        )
    if cancelled:
        raise asyncio.CancelledError()


async def stream_agent(
    session_id: str,
    user_message: str,
    turn_id: str | None = None,
    started_at: float | None = None,
    provider: BaseLLMProvider | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """
    流式执行一轮对话，逐条产出已落库的事件 {turn_id, seq, type, data, created_at}。
    turn_id 可由路由层预先创建（便于在响应头返回、并在建 turn 失败时直接返回 500）。
    """
    started_at = started_at if started_at is not None else time.monotonic()
    if turn_id is None:
        turn_id = await asyncio.to_thread(event_store.create_turn, session_id, user_message)

    queue: "asyncio.Queue[dict[str, Any] | None]" = asyncio.Queue()
    task = asyncio.create_task(_run_turn(turn_id, session_id, user_message, started_at, queue, provider))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    try:
        while (item := await queue.get()) is not None:
            yield item
    finally:
        # 消费方提前退出（客户端断开）时不取消任务：让推理跑完并落库，供断线补发
        if not task.done():
            logger.info("客户端提前断开 turn_id=%s，后台继续执行，可通过 after_seq 补发", turn_id)


async def arun_agent(
    session_id: str, user_message: str, provider: BaseLLMProvider | None = None
) -> tuple[str, list[str], int, list[str]]:
    """异步收集版：消费完整事件流，从终态事件取结果，返回 (reply, tools_called, latency_ms, citations)。"""
    started_at = time.monotonic()
    final: dict[str, Any] | None = None
    async for event in stream_agent(session_id, user_message, started_at=started_at, provider=provider):
        if event["type"] in ("done", "error"):
            final = event["data"]

    if final is None:
        # 终态事件落库失败的极端情况：仍返回兜底 4 元组，保持接口契约
        return LLM_ERROR_REPLY, [], _elapsed_ms(started_at), []
    return final["reply"], final["tools_called"], final["latency_ms"], final["citations"]


def run_agent(session_id: str, user_message: str) -> tuple[str, list[str], int, list[str]]:
    """
    v1.0 同步兼容入口：签名与返回值 (reply, tools_called, latency_ms, citations) 保持不变，
    同一 session_id 自动携带最近 6 轮上下文。供脚本 / 评测等同步场景调用。
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # 当前线程没有事件循环（常规同步调用）：直接新建循环执行
        return asyncio.run(arun_agent(session_id, user_message))

    # 已处于运行中的事件循环内（如在 async 函数里误调同步入口）：asyncio.run 会报错，
    # 因此放到独立线程的新循环中执行（Provider 已按事件循环隔离客户端，可安全跨循环）
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, arun_agent(session_id, user_message)).result()
