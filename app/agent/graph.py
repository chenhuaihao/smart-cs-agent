"""
app/agent/graph.py

v2.0 P1 LangGraph 状态机：引入证据账本（EvidenceLedger）、规则校验（policy_verifier）、
降级节点（fallback_node）、步数/单步超时护栏与高风险写操作 HITL 挂起。

    START → agent ──(有 tool_calls)────→ tools ──→ agent
              │
              ├──(无 tool_calls)───────→ policy_verifier ──(pass)─────────────→ END
              │                                 ├──(fail & retry_count <= 2)──→ agent
              │                                 └──(fail & retry_count > 2)───→ fallback → END
              └──(步数耗尽 / 超时 / 异常)─→ fallback ─────────────────────────────→ END
"""

import asyncio
import json
import logging
import operator
import time
from typing import Annotated, Any, Callable, TypedDict

from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from app.agent.guardrails import (
    HIGH_RISK_TOOLS,
    MAX_VERIFY_RETRIES,
    EvidenceLedger,
    build_fallback_reply,
    build_input_blocked_reply,
    check_input_safety,
    get_max_steps,
    get_step_timeout_sec,
    merge_ledger,
    record_tool_evidence,
    verify_reply_structured,
)
from app.agent.tools import TOOLS_SCHEMA, execute_tool
from app.llm.base import BaseLLMProvider, ToolCallRequest
from app.llm.factory import get_llm_provider

logger = logging.getLogger(__name__)

# 默认步数上限与 LangGraph superstep 递归上限余量
MAX_STEPS = get_max_steps()
RECURSION_LIMIT = 40

MAX_STEPS_FALLBACK_REPLY = build_fallback_reply("单轮工具推理步数达到安全上限")
LLM_ERROR_REPLY = "抱歉，我暂时无法连接到后端服务，请稍后重试。如有紧急问题，可联系人工客服。"


def merge_citations(left: list[str] | None, right: list[str] | None) -> list[str]:
    """citations 的 reducer：跨多次工具调用保序去重（先出现的优先）。"""
    merged = list(left or [])
    for item in right or []:
        if item not in merged:
            merged.append(item)
    return merged


class AgentState(TypedDict, total=False):
    """图状态（含 P1 新增的 evidence_ledger、retry_count 与 fallback_reason）。"""

    messages: Annotated[list[dict[str, Any]], operator.add]
    pending_tool_calls: list[dict[str, Any]]
    tools_called: Annotated[list[str], operator.add]
    citations: Annotated[list[str], merge_citations]
    evidence_ledger: Annotated[EvidenceLedger, merge_ledger]
    final_reply: str
    steps: int
    retry_count: int
    verifier_passed: bool
    fallback_reason: str


def extract_citation_details(tool_name: str, tool_result: str) -> list[dict[str, Any]]:
    """
    从 search_knowledge_base 的结构化 JSON 结果中提取引用详情（保序去重）。
    每条包含 source（'文件名#章节名'）、retrieval_source（bm25/vector/hybrid）与 score。
    """
    if tool_name != "search_knowledge_base":
        return []
    try:
        data = json.loads(tool_result)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []

    details: list[dict[str, Any]] = []
    seen: set[str] = set()
    for chunk in data:
        if not isinstance(chunk, dict):
            continue
        source, section = chunk.get("source", ""), chunk.get("section", "")
        key = f"{source}#{section}"
        if not (source and section) or key in seen:
            continue
        seen.add(key)
        retrieval_source = str(chunk.get("retrieval_source", "bm25"))
        if retrieval_source not in ("bm25", "vector", "hybrid"):
            retrieval_source = "bm25"
        try:
            score = float(chunk.get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        details.append(
            {
                "source": key,
                "retrieval_source": retrieval_source,
                "score": score,
            }
        )
    return details


def extract_citations(tool_name: str, tool_result: str) -> list[str]:
    """从 search_knowledge_base 的结构化 JSON 结果中提取 "文件名#章节名"（保序去重）。"""
    return [d["source"] for d in extract_citation_details(tool_name, tool_result)]


def _get_provider(config: RunnableConfig) -> BaseLLMProvider:
    provider = (config.get("configurable") or {}).get("provider")
    return provider if provider is not None else get_llm_provider()


def _safe_json(arguments: str | dict[str, Any]) -> Any:
    if isinstance(arguments, dict):
        return arguments
    try:
        return json.loads(arguments)
    except (json.JSONDecodeError, TypeError):
        return arguments


async def _consume_llm_stream(
    provider: BaseLLMProvider,
    messages: list[dict[str, Any]],
    step: int,
    config: RunnableConfig,
) -> tuple[str, list[ToolCallRequest]]:
    """消费一次 Provider 流式推理并实时派发 token 事件。"""
    text_parts: list[str] = []
    tool_calls: list[ToolCallRequest] = []
    async for chunk in provider.astream_chat(messages, TOOLS_SCHEMA):
        if chunk.type == "text_delta" and chunk.text:
            text_parts.append(chunk.text)
            await adispatch_custom_event("token", {"delta": chunk.text, "step": step}, config=config)
        elif chunk.type == "tool_calls":
            tool_calls.extend(chunk.tool_calls)
    return "".join(text_parts), tool_calls


async def agent_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """推理节点：带步数预算与单步超时护栏，流式调用模型。"""
    cfg = config.get("configurable") or {}
    max_steps = int(cfg.get("max_steps") or get_max_steps())
    step_timeout = float(cfg.get("step_timeout_sec") or get_step_timeout_sec())

    steps = state.get("steps", 0)
    if steps == 0:
        # v3.0 P2 输入安全前置护栏：在调用 LLM 前 0ms 拦截高置信 Prompt Injection / 越权注入
        user_messages = [m.get("content", "") for m in state.get("messages", []) if m.get("role") == "user"]
        latest_user_text = str(user_messages[-1]) if user_messages else ""
        blocked = check_input_safety(latest_user_text)
        if blocked is not None:
            code, reason = blocked
            logger.warning("input_guard 拦截高风险注入输入 code=%s", code)
            reply = build_input_blocked_reply(reason)
            await adispatch_custom_event(
                "stage",
                {"stage": "input_guard_blocked", "code": code, "reason": reason},
                config=config,
            )
            await adispatch_custom_event(
                "token",
                {"delta": reply, "step": 0},
                config=config,
            )
            return {
                "messages": [{"role": "assistant", "content": reply}],
                "final_reply": reply,
                "pending_tool_calls": [],
                "steps": 1,
                "verifier_passed": True,
                "fallback_reason": "",
            }

    if steps >= max_steps:
        logger.warning("agent_node 达到最大步数 %d，转入 fallback_node", max_steps)
        return {
            "pending_tool_calls": [],
            "fallback_reason": f"单轮推理/工具步数已达上限（MAX_STEPS={max_steps}）",
        }

    step = steps + 1
    await adispatch_custom_event("stage", {"stage": "thinking", "step": step}, config=config)

    try:
        text, tool_calls = await asyncio.wait_for(
            _consume_llm_stream(_get_provider(config), state["messages"], step, config),
            timeout=step_timeout,
        )
    except asyncio.TimeoutError:
        logger.warning("agent_node 单步 LLM 超时 step=%d timeout=%ss", step, step_timeout)
        return {
            "pending_tool_calls": [],
            "steps": step,
            "fallback_reason": f"大模型单步推理超时（>{step_timeout}s）",
        }

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
            "fallback_reason": "",
        }

    return {
        "messages": [{"role": "assistant", "content": text}],
        "final_reply": text,
        "pending_tool_calls": [],
        "steps": step,
        "fallback_reason": "",
    }


async def tool_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """
    工具节点：
      1. 遇到高风险写工具（create_after_sales_ticket）且 auto_approve=False 时，
         通过 approval_handler 持久化待确认请求、推送 approval_required 事件并挂起等待；
      2. 执行工具后自动更新 evidence_ledger，并通过 citation / stage 事件透传。
    """
    cfg = config.get("configurable") or {}
    auto_approve: bool = bool(cfg.get("auto_approve", True))
    approval_handler: Callable[[dict[str, Any]], Any] | None = cfg.get("approval_handler")
    step_timeout = float(cfg.get("step_timeout_sec") or get_step_timeout_sec())

    calls = state.get("pending_tool_calls") or []
    await adispatch_custom_event(
        "stage", {"stage": "tool_running", "tools": [c["name"] for c in calls]}, config=config
    )

    current_ledger = state.get("evidence_ledger") or EvidenceLedger()
    known_citations = list(state.get("citations") or [])
    tool_messages: list[dict[str, Any]] = []
    names: list[str] = []
    new_citations: list[str] = []

    for call in calls:
        tool_name = call["name"]
        parsed_args = _safe_json(call["arguments"])

        # 高风险写操作 HITL 拦截：暂停落库，等待用户显式确认
        if tool_name in HIGH_RISK_TOOLS and not auto_approve and approval_handler is not None:
            approved = await approval_handler(call)
            if not approved:
                cancel_res = json.dumps(
                    {"cancelled": True, "message": "用户已取消创建售后工单，请告知用户已取消建单，切勿声称已创建。"},
                    ensure_ascii=False,
                )
                await adispatch_custom_event(
                    "tool_call",
                    {
                        "call_id": call["id"],
                        "name": tool_name,
                        "arguments": parsed_args,
                        "status": "cancelled",
                        "duration_ms": 0,
                    },
                    config=config,
                )
                tool_messages.append({"role": "tool", "tool_call_id": call["id"], "content": cancel_res})
                continue

        started = time.monotonic()
        try:
            result: str = await asyncio.wait_for(
                asyncio.to_thread(execute_tool, tool_name, call["arguments"]),
                timeout=step_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("tool_node 单步工具超时 tool=%s timeout=%ss", tool_name, step_timeout)
            return {
                "pending_tool_calls": [],
                "fallback_reason": f"工具 {tool_name} 执行超时（>{step_timeout}s）",
            }

        duration_ms = int((time.monotonic() - started) * 1000)
        names.append(tool_name)

        parsed = _safe_json(result)
        failed = isinstance(parsed, dict) and "error" in parsed
        await adispatch_custom_event(
            "tool_call",
            {
                "call_id": call["id"],
                "name": tool_name,
                "arguments": parsed_args,
                "status": "error" if failed else "success",
                "duration_ms": duration_ms,
            },
            config=config,
        )

        all_details = extract_citation_details(tool_name, result)
        fresh_details = [
            d for d in all_details
            if d["source"] not in known_citations and d["source"] not in new_citations
        ]
        fresh = [d["source"] for d in fresh_details]
        if fresh:
            new_citations.extend(fresh)
            await adispatch_custom_event(
                "citation",
                {
                    "call_id": call["id"],
                    "citations": fresh,
                    "citation_details": fresh_details,
                },
                config=config,
            )

        # 更新证据账本并通过 stage 事件透传最新账本摘要（含 citation_details）
        current_ledger = record_tool_evidence(
            current_ledger, tool_name, parsed_args, result, fresh, fresh_details
        )
        await adispatch_custom_event(
            "stage",
            {
                "stage": "evidence_updated",
                "tool": tool_name,
                "evidence_ledger": current_ledger.model_dump(),
            },
            config=config,
        )

        tool_messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
        logger.info("tool_node 完成 tool=%s status=%s duration_ms=%d", tool_name, not failed, duration_ms)

    return {
        "messages": tool_messages,
        "tools_called": names,
        "citations": new_citations,
        "evidence_ledger": current_ledger,
        "pending_tool_calls": [],
    }


async def policy_verifier_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """
    规则校验节点：用 EvidenceLedger 硬校验模型最终回复，输出结构化违规分类码。
    校验失败时将具体违规原因注入 messages 触发 agent_node 重写（最多重试 MAX_VERIFY_RETRIES=2 次）。
    """
    if state.get("verifier_passed") is True and not state.get("pending_tool_calls"):
        return {"verifier_passed": True, "fallback_reason": ""}

    reply = state.get("final_reply") or ""
    ledger = state.get("evidence_ledger") or EvidenceLedger()
    violations = verify_reply_structured(reply, ledger)
    reasons = [v.message for v in violations]
    violation_codes = [v.code for v in violations]

    await adispatch_custom_event(
        "stage",
        {
            "stage": "verifying",
            "passed": len(violations) == 0,
            "reasons": reasons,
            "violation_codes": violation_codes,
        },
        config=config,
    )

    if not violations:
        return {"verifier_passed": True, "fallback_reason": ""}

    retry_count = state.get("retry_count", 0) + 1
    reason_text = "；".join(reasons)
    logger.warning(
        "policy_verifier 拦截 retry=%d/%d codes=%s reasons=%s",
        retry_count, MAX_VERIFY_RETRIES, violation_codes, reason_text,
    )

    if retry_count <= MAX_VERIFY_RETRIES:
        feedback_msg = {
            "role": "user",
            "content": (
                f"【系统规则校验未通过（第 {retry_count}/{MAX_VERIFY_RETRIES} 次重写）】\n"
                f"违规分类：{', '.join(violation_codes)}\n"
                f"违规原因：{reason_text}\n"
                "请立即纠正：1) 若未查询订单，必须先调用 query_order 工具或请用户提供订单号，严禁凭空编造订单状态与运单号；"
                "2) 若未调用或用户取消了 create_after_sales_ticket，严禁声称已创建工单；"
                "3) 严禁在回复正文中暴露任何 .md 内部文件名，请改用自然的中文规则名称。请重新生成合规答复或调用所需工具。"
            ),
        }
        return {
            "verifier_passed": False,
            "retry_count": retry_count,
            "messages": [feedback_msg],
        }

    return {
        "verifier_passed": False,
        "retry_count": retry_count,
        "fallback_reason": f"回复连续 {MAX_VERIFY_RETRIES} 次未通过证据账本规则校验（{reason_text}）",
    }


async def fallback_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """降级节点：当校验连续失败、步数超限或超时时，输出带明确原因的安全兜底提示并引导转人工。"""
    reason = state.get("fallback_reason") or "触发安全护栏限制"
    reply = build_fallback_reply(reason)
    logger.warning("fallback_node 触发降级 reason=%s", reason)
    await adispatch_custom_event(
        "stage", {"stage": "fallback", "reason": reason}, config=config
    )
    await adispatch_custom_event(
        "token", {"delta": reply, "step": state.get("steps", 0), "fallback": True}, config=config
    )
    return {"final_reply": reply}


def _route_after_agent(state: AgentState) -> str:
    if state.get("fallback_reason"):
        return "fallback"
    if state.get("pending_tool_calls"):
        return "tools"
    return "policy_verifier"


def _route_after_tools(state: AgentState) -> str:
    if state.get("fallback_reason"):
        return "fallback"
    return "agent"


def _route_after_verifier(state: AgentState) -> str:
    if state.get("verifier_passed"):
        return END
    if state.get("retry_count", 0) <= MAX_VERIFY_RETRIES and not state.get("fallback_reason"):
        return "agent"
    return "fallback"


def build_graph() -> Any:
    """构建并编译 P1 状态机（agent / tools / policy_verifier / fallback）。"""
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("policy_verifier", policy_verifier_node)
    graph.add_node("fallback", fallback_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        _route_after_agent,
        {"tools": "tools", "policy_verifier": "policy_verifier", "fallback": "fallback"},
    )
    graph.add_conditional_edges(
        "tools",
        _route_after_tools,
        {"agent": "agent", "fallback": "fallback"},
    )
    graph.add_conditional_edges(
        "policy_verifier",
        _route_after_verifier,
        {"agent": "agent", "fallback": "fallback", END: END},
    )
    graph.add_edge("fallback", END)
    return graph.compile()


agent_graph = build_graph()
