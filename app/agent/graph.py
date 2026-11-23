"""
app/agent/graph.py

v2.0 LangGraph 状态机：用 StateGraph 重构 v1.0 的 for 循环式 Function Calling。

    START → agent ──(有 tool_calls)──→ tools ──→ agent → ... → END
                 └──(无 tool_calls / 步数耗尽)──→ END

  - agent_node：调用 BaseLLMProvider 流式推理，逐 token 发出 token 事件；
  - tool_node ：复用 app/agent/tools.py 的 execute_tool，累积 tools_called 与 citations；
  - 节点内通过 adispatch_custom_event 发出业务事件（stage / token / tool_call / citation），
    由 core.py 用 astream_events(version="v2") 统一消费、落库、推送。
节点本身不碰 SQLite 事件表，保证"图只负责推理，持久化与协议由 core 统一兜底"。
"""

import asyncio
import json
import logging
import operator
import time
from typing import Annotated, Any, TypedDict

from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from app.agent.tools import TOOLS_SCHEMA, execute_tool
from app.llm.base import BaseLLMProvider, ToolCallRequest
from app.llm.factory import get_llm_provider

logger = logging.getLogger(__name__)

# 与 v1.0 一致：最多 6 次大模型推理（知识库检索可能多消耗一步）
MAX_STEPS = 6
# LangGraph 每个节点执行算一次 superstep：6 轮 × (agent + tools) + 兜底余量
RECURSION_LIMIT = MAX_STEPS * 2 + 4

MAX_STEPS_FALLBACK_REPLY = (
    "非常抱歉，我在处理您的请求时遇到了一点复杂情况，未能及时给出回复。"
    "建议您稍后重试，或联系人工客服为您进一步处理。"
)
LLM_ERROR_REPLY = "抱歉，我暂时无法连接到后端服务，请稍后重试。如有紧急问题，可联系人工客服。"


def merge_citations(left: list[str] | None, right: list[str] | None) -> list[str]:
    """citations 的 reducer：跨多次工具调用保序去重（先出现的优先）。"""
    merged = list(left or [])
    for item in right or []:
        if item not in merged:
            merged.append(item)
    return merged


class AgentState(TypedDict, total=False):
    """图状态。messages 直接使用 OpenAI 字典格式，保持与 Provider 协议一致、无需转换。"""

    messages: Annotated[list[dict[str, Any]], operator.add]
    pending_tool_calls: list[dict[str, Any]]  # 本步模型请求的工具调用，tool_node 消费后清空
    tools_called: Annotated[list[str], operator.add]
    citations: Annotated[list[str], merge_citations]
    final_reply: str
    steps: int


def extract_citations(tool_name: str, tool_result: str) -> list[str]:
    """
    从 search_knowledge_base 的结构化 JSON 结果中提取 "文件名#章节名"（保序去重）。
    直接解析工具结果而非依赖模型措辞，可靠性 100%（与 v1.0 逻辑一致）。
    """
    if tool_name != "search_knowledge_base":
        return []
    try:
        data = json.loads(tool_result)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        # {"error": ...} 等非列表结构没有可提取的引用
        return []

    citations: list[str] = []
    for chunk in data:
        source, section = chunk.get("source", ""), chunk.get("section", "")
        key = f"{source}#{section}"
        if source and section and key not in citations:
            citations.append(key)
    return citations


def _get_provider(config: RunnableConfig) -> BaseLLMProvider:
    """优先使用调用方经 config 注入的 Provider（便于测试 / 多租户），否则取全局单例。"""
    provider = (config.get("configurable") or {}).get("provider")
    return provider if provider is not None else get_llm_provider()


def _safe_json(arguments: str) -> Any:
    """工具参数解析失败时原样返回字符串，事件推送不应因模型输出脏 JSON 而中断。"""
    try:
        return json.loads(arguments)
    except (json.JSONDecodeError, TypeError):
        return arguments


async def agent_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """推理节点：流式调用模型，文本增量实时发 token 事件，工具调用交给 tool_node。"""
    steps = state.get("steps", 0)
    if steps >= MAX_STEPS:
        # 步数耗尽仍在要工具：不再调用模型，直接友好兜底（行为与 v1.0 一致）
        logger.warning("agent_node 达到最大步数 %d，返回兜底回复", MAX_STEPS)
        return {"final_reply": MAX_STEPS_FALLBACK_REPLY, "pending_tool_calls": []}

    step = steps + 1
    await adispatch_custom_event("stage", {"stage": "thinking", "step": step}, config=config)

    text_parts: list[str] = []
    tool_calls: list[ToolCallRequest] = []
    # Provider 异常不在节点内吞掉：上抛给 core，由 core 统一写 error 终态事件
    async for chunk in _get_provider(config).astream_chat(state["messages"], TOOLS_SCHEMA):
        if chunk.type == "text_delta" and chunk.text:
            text_parts.append(chunk.text)
            await adispatch_custom_event("token", {"delta": chunk.text, "step": step}, config=config)
        elif chunk.type == "tool_calls":
            tool_calls.extend(chunk.tool_calls)

    text = "".join(text_parts)
    logger.info("agent_node step=%d text_len=%d tool_calls=%d", step, len(text), len(tool_calls))

    if tool_calls:
        assistant_msg = {
            "role": "assistant",
            "content": text or None,
            "tool_calls": [
                {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments}}
                for tc in tool_calls
            ],
        }
        return {
            "messages": [assistant_msg],
            "pending_tool_calls": [tc.model_dump() for tc in tool_calls],
            "steps": step,
        }

    return {
        "messages": [{"role": "assistant", "content": text}],
        "final_reply": text,
        "pending_tool_calls": [],
        "steps": step,
    }


async def tool_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """工具节点：顺序执行本步全部工具调用，发出 tool_call / citation 事件并回填 tool 消息。"""
    calls = state.get("pending_tool_calls") or []
    await adispatch_custom_event(
        "stage", {"stage": "tool_running", "tools": [c["name"] for c in calls]}, config=config
    )

    known_citations = list(state.get("citations") or [])
    tool_messages: list[dict[str, Any]] = []
    names: list[str] = []
    new_citations: list[str] = []

    for call in calls:
        started = time.monotonic()
        # execute_tool 是同步函数（含 SQLite / BM25），放到线程池执行，避免阻塞事件循环里的其他 SSE 连接
        result: str = await asyncio.to_thread(execute_tool, call["name"], call["arguments"])
        duration_ms = int((time.monotonic() - started) * 1000)
        names.append(call["name"])

        parsed = _safe_json(result)
        failed = isinstance(parsed, dict) and "error" in parsed
        # 只推送工具名、参数与执行状态，不推送工具原始 JSON（含订单隐私 / 知识库原文）
        await adispatch_custom_event(
            "tool_call",
            {
                "call_id": call["id"],
                "name": call["name"],
                "arguments": _safe_json(call["arguments"]),
                "status": "error" if failed else "success",
                "duration_ms": duration_ms,
            },
            config=config,
        )

        fresh = [
            c for c in extract_citations(call["name"], result)
            if c not in known_citations and c not in new_citations
        ]
        if fresh:
            new_citations.extend(fresh)
            await adispatch_custom_event(
                "citation", {"call_id": call["id"], "citations": fresh}, config=config
            )

        tool_messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
        logger.info("tool_node 完成 tool=%s status=%s duration_ms=%d", call["name"], not failed, duration_ms)

    return {
        "messages": tool_messages,
        "tools_called": names,
        "citations": new_citations,
        "pending_tool_calls": [],
    }


def _route_after_agent(state: AgentState) -> str:
    """有待执行的工具调用则进入 tools，否则结束。"""
    return "tools" if state.get("pending_tool_calls") else END


def build_graph() -> Any:
    """构建并编译状态机（无 checkpointer：对话记忆仍由 chat_logs 负责，保持 v1.0 语义）。"""
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", _route_after_agent, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


# 模块级编译一次即可复用：编译后的图是无状态的，每次 astream_events 传入独立的初始 state
agent_graph = build_graph()
