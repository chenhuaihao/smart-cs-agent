"""
tests/test_p1_guardrails.py

v2.0 P1 阶段验收测试（使用 Mock BaseLLMProvider，不消耗真实 API 额度）：
  1. 未查单直接编造订单状态或假称已建单时，被 policy_verifier 拦截并触发重写；
  2. 连续校验失败（>2 次）、超过步数上限（MAX_STEPS）或超时时，正确走 fallback_node 输出降级回复；
  3. 流式模式触发 create_after_sales_ticket 时发出 approval_required 且数据库暂不新增工单；
     调用 /approve 传入 approved=true 后工单落库，传入 approved=false 则取消不落库；
  4. 同步 POST /api/chat 默认 auto_approve=True，保持 v1.0 建单契约 100% 兼容。
"""

import asyncio
import json
from typing import Any, AsyncIterator, Callable

import pytest
from fastapi.testclient import TestClient

from app.db import database
from app.db.database import get_db_connection
from app.llm.base import BaseLLMProvider, LLMStreamChunk, ToolCallRequest
from app.llm.factory import set_llm_provider
from app.main import app


class ScriptedProvider(BaseLLMProvider):
    """按自定义脚本或回调依次产出 LLMStreamChunk 的可控 Mock Provider。"""

    name = "scripted-mock"

    def __init__(self, handler: Callable[[int, list[dict[str, Any]]], list[LLMStreamChunk]]) -> None:
        self._handler = handler
        self.call_count = 0
        self.history_calls: list[list[dict[str, Any]]] = []

    async def astream_chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[LLMStreamChunk]:
        self.call_count += 1
        self.history_calls.append(list(messages))
        chunks = self._handler(self.call_count, messages)
        for ch in chunks:
            yield ch


def parse_sse(body: str) -> list[dict[str, Any]]:
    """解析 SSE 响应体为 [{id, event, data}]。"""
    frames = []
    for block in body.replace("\r\n", "\n").strip().split("\n\n"):
        if not block.strip():
            continue
        fields: dict[str, str] = {}
        for line in block.split("\n"):
            key, sep, value = line.partition(": ")
            assert sep, f"非法 SSE 行：{line!r}"
            fields[key] = value
        frames.append({"id": int(fields["id"]), "event": fields["event"], "data": json.loads(fields["data"])})
    return frames


def _count_tickets(order_id: str) -> int:
    with get_db_connection() as conn:
        row = conn.execute("SELECT COUNT(*) AS cnt FROM tickets WHERE order_id = ?;", (order_id,)).fetchone()
    return int(row["cnt"])


@pytest.fixture()
def client(tmp_path, monkeypatch) -> TestClient:
    """每个测试独立临时 SQLite 库 + 初始化种子数据。"""
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "test_p1.db"))
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "1000")
    monkeypatch.setenv("DAILY_CHAT_LIMIT", "100000")
    with TestClient(app) as test_client:
        yield test_client
    set_llm_provider(None)


# ---------------------------------------------------------------------------
# 1. 未查单编造订单状态 / 假称已建单 → policy_verifier 拦截并触发重写成功
# ---------------------------------------------------------------------------

def test_verifier_intercepts_unverified_order_and_rewrites(client: TestClient) -> None:
    """
    第 1 次推理：模型未调用 query_order 就直接编造 "您的订单 ORD20261101 已签收"；
    policy_verifier 拦截并反馈违规原因；
    第 2 次推理：模型收到反馈后调用 query_order("ORD20261101")；
    第 3 次推理：模型根据真实账本给出合规答复，校验通过。
    """
    def script(call_idx: int, messages: list[dict[str, Any]]) -> list[LLMStreamChunk]:
        if call_idx == 1:
            # 违规：未查单直接断言订单状态
            return [LLMStreamChunk(type="text_delta", text="经核实，您的订单 ORD20261101 已签收。")]
        if call_idx == 2:
            # 确认收到了系统规则校验反馈
            assert "系统规则校验未通过" in (messages[-1].get("content") or "")
            return [
                LLMStreamChunk(
                    type="tool_calls",
                    tool_calls=[
                        ToolCallRequest(
                            id="call_q1",
                            name="query_order",
                            arguments=json.dumps({"order_id": "ORD20261101"}),
                        )
                    ],
                )
            ]
        return [
            LLMStreamChunk(
                type="text_delta",
                text="已为您核实订单 ORD20261101（智能降噪无线蓝牙耳机 Pro），当前状态为已签收，运单号 SF1428570001。",
            )
        ]

    provider = ScriptedProvider(script)
    set_llm_provider(provider)

    resp = client.post("/api/chat/stream", json={"session_id": "s-verify-order", "message": "查下 ORD20261101"})
    assert resp.status_code == 200
    frames = parse_sse(resp.text)

    # 断言触发了 verifying 失败 → 重写 → query_order → evidence_updated → verifying 通过 → done
    verifying_events = [
        f["data"]["data"]
        for f in frames
        if f["event"] == "stage" and f["data"]["data"].get("stage") == "verifying"
    ]
    assert len(verifying_events) == 2
    assert verifying_events[0]["passed"] is False
    assert any("ORD20261101" in r for r in verifying_events[0]["reasons"])
    assert verifying_events[1]["passed"] is True

    evidence_events = [
        f["data"]["data"]
        for f in frames
        if f["event"] == "stage" and f["data"]["data"].get("stage") == "evidence_updated"
    ]
    assert len(evidence_events) == 1
    assert "ORD20261101" in evidence_events[0]["evidence_ledger"]["orders"]

    done = frames[-1]["data"]["data"]
    assert done["tools_called"] == ["query_order"]
    assert "SF1428570001" in done["reply"]


def test_verifier_intercepts_fake_ticket_creation_and_rewrites(client: TestClient) -> None:
    """
    第 1 次推理：模型未调用建单工具却谎称 "已为您创建退货工单 TKT20261123-AAAAAA"；
    policy_verifier 拦截并要求重写；
    第 2 次推理：模型纠正为向用户索要订单号与具体退货原因，校验通过。
    """
    def script(call_idx: int, messages: list[dict[str, Any]]) -> list[LLMStreamChunk]:
        if call_idx == 1:
            return [
                LLMStreamChunk(
                    type="text_delta",
                    text="好的，已为您创建退货工单 TKT20261123-AAAAAA，请耐心等待专员处理。",
                )
            ]
        assert "系统规则校验未通过" in (messages[-1].get("content") or "")
        return [
            LLMStreamChunk(
                type="text_delta",
                text="抱歉，创建退货工单前需要您先提供具体的订单号和退货原因，请您补充后我为您办理。",
            )
        ]

    set_llm_provider(ScriptedProvider(script))
    resp = client.post("/api/chat", json={"session_id": "s-verify-ticket", "message": "帮我退货"})
    assert resp.status_code == 200
    body = resp.json()
    assert "请您补充" in body["reply"]
    assert "TKT20261123-AAAAAA" not in body["reply"]


# ---------------------------------------------------------------------------
# 2. 连续校验失败 (>2 次) 或超过步数上限 (MAX_STEPS) → fallback_node 降级回复
# ---------------------------------------------------------------------------

def test_verifier_consecutive_failures_route_to_fallback(client: TestClient) -> None:
    """模型连续 3 次（首次 + 2 次重试）编造未查订单状态 → 触发 fallback_node 输出降级兜底并引导转人工。"""
    def always_hallucinate(call_idx: int, messages: list[dict[str, Any]]) -> list[LLMStreamChunk]:
        return [LLMStreamChunk(type="text_delta", text="您的订单 ORD20261101 已签收，运单号 SF1428570001。")]

    provider = ScriptedProvider(always_hallucinate)
    set_llm_provider(provider)

    resp = client.post("/api/chat/stream", json={"session_id": "s-fallback-verify", "message": "查单"})
    assert resp.status_code == 200
    frames = parse_sse(resp.text)

    # 首次推理 + 2 次重试 = 共调用 3 次 LLM，随后进入 fallback
    assert provider.call_count == 3
    fallback_stages = [
        f["data"]["data"]
        for f in frames
        if f["event"] == "stage" and f["data"]["data"].get("stage") == "fallback"
    ]
    assert len(fallback_stages) == 1
    assert "证据账本规则校验" in fallback_stages[0]["reason"]

    done = frames[-1]["data"]["data"]
    assert "安全护栏" in done["reply"] and "人工客服" in done["reply"]


def test_exceeding_max_steps_routes_to_fallback(client: TestClient, monkeypatch) -> None:
    """当模型无限循环调用工具达到 MAX_STEPS 上限时，自动转入 fallback_node 正常收尾。"""
    monkeypatch.setenv("MAX_STEPS", "2")

    def infinite_tools(call_idx: int, messages: list[dict[str, Any]]) -> list[LLMStreamChunk]:
        return [
            LLMStreamChunk(
                type="tool_calls",
                tool_calls=[
                    ToolCallRequest(
                        id=f"call_loop_{call_idx}",
                        name="query_order",
                        arguments=json.dumps({"order_id": "ORD20261101"}),
                    )
                ],
            )
        ]

    set_llm_provider(ScriptedProvider(infinite_tools))
    resp = client.post("/api/chat/stream", json={"session_id": "s-fallback-steps", "message": "查单 ORD20261101"})
    assert resp.status_code == 200
    frames = parse_sse(resp.text)

    fallback_stages = [
        f["data"]["data"]
        for f in frames
        if f["event"] == "stage" and f["data"]["data"].get("stage") == "fallback"
    ]
    assert len(fallback_stages) == 1
    assert "MAX_STEPS=2" in fallback_stages[0]["reason"]

    done = frames[-1]["data"]["data"]
    assert "MAX_STEPS=2" in done["reply"] and "人工客服" in done["reply"]
    assert len(done["tools_called"]) == 2


# ---------------------------------------------------------------------------
# 3. 高风险写操作 HITL：流式暂停 approval_required → /approve (true 落库 / false 取消)
# ---------------------------------------------------------------------------

def _make_ticket_provider() -> ScriptedProvider:
    """模拟：先查单 query_order → 再请求 create_after_sales_ticket → 根据执行或取消结果给出最终回复。"""
    def script(call_idx: int, messages: list[dict[str, Any]]) -> list[LLMStreamChunk]:
        if call_idx == 1:
            return [
                LLMStreamChunk(
                    type="tool_calls",
                    tool_calls=[
                        ToolCallRequest(
                            id="call_order_1",
                            name="query_order",
                            arguments=json.dumps({"order_id": "ORD20261121A"}),
                        )
                    ],
                )
            ]
        if call_idx == 2:
            return [
                LLMStreamChunk(
                    type="tool_calls",
                    tool_calls=[
                        ToolCallRequest(
                            id="call_ticket_1",
                            name="create_after_sales_ticket",
                            arguments=json.dumps(
                                {
                                    "order_id": "ORD20261121A",
                                    "issue_type": "退货",
                                    "reason": "全新未拆封7天无理由退货",
                                    "summary": "申请7天无理由退货",
                                },
                                ensure_ascii=False,
                            ),
                        )
                    ],
                )
            ]
        last_tool_msg = messages[-1].get("content") or "{}"
        data = json.loads(last_tool_msg)
        if data.get("cancelled"):
            return [LLMStreamChunk(type="text_delta", text="好的，已为您取消创建退货售后工单。")]
        tid = data.get("ticket_id", "")
        return [
            LLMStreamChunk(
                type="text_delta",
                text=f"已为您成功创建退货工单 {tid}（状态：待处理），售后专员会尽快跟进。",
            )
        ]

    return ScriptedProvider(script)


def test_hitl_stream_pause_and_approve_true_creates_ticket(client: TestClient) -> None:
    """流式模式触发建单时暂停并发出 approval_required，此时数据库未新增工单；approved=true 后落库并完成。"""
    set_llm_provider(_make_ticket_provider())
    order_id = "ORD20261121A"
    before_cnt = _count_tickets(order_id)

    resp = client.post(
        "/api/chat/stream",
        json={"session_id": "s-hitl-approve", "message": "订单 ORD20261121A 全新未拆封，帮我建退货工单"},
    )
    assert resp.status_code == 200
    turn_id = resp.headers["x-turn-id"]
    frames = parse_sse(resp.text)

    # 第一段流以 approval_required 暂停结束，尚未发出 done，且数据库工单数未增加
    assert frames[-1]["event"] == "approval_required"
    approval_data = frames[-1]["data"]["data"]
    assert approval_data["turn_id"] == turn_id
    assert approval_data["tool"] == "create_after_sales_ticket"
    assert approval_data["args"]["order_id"] == order_id
    assert "ORD20261121A" in approval_data["summary"]
    assert _count_tickets(order_id) == before_cnt

    # 查询 turn 状态应为 waiting_approval
    status_resp = client.get(f"/api/chat/turns/{turn_id}/events")
    assert status_resp.status_code == 200
    assert status_resp.json()["status"] == "waiting_approval"

    # 调用 /approve 传入 approved=true → 真正落库并流式返回后续 tool_call / token / done
    approve_resp = client.post(f"/api/chat/turns/{turn_id}/approve", json={"approved": True})
    assert approve_resp.status_code == 200
    resume_frames = parse_sse(approve_resp.text)

    resume_types = [f["event"] for f in resume_frames]
    assert "tool_call" in resume_types and "token" in resume_types and resume_types[-1] == "done"
    assert _count_tickets(order_id) == before_cnt + 1

    done = resume_frames[-1]["data"]["data"]
    assert "create_after_sales_ticket" in done["tools_called"]
    assert "TKT" in done["reply"]

    # 校验整轮事件 seq 全程严格连续递增
    all_events = client.get(f"/api/chat/turns/{turn_id}/events").json()
    assert all_events["status"] == "completed"
    seqs = [e["seq"] for e in all_events["events"]]
    assert seqs == list(range(1, len(seqs) + 1))


def test_hitl_stream_pause_and_approve_false_cancels_without_db_write(client: TestClient) -> None:
    """流式模式触发建单后调用 /approve 传入 approved=false → 不落库，告知用户已取消并正常收尾。"""
    set_llm_provider(_make_ticket_provider())
    order_id = "ORD20261121A"
    before_cnt = _count_tickets(order_id)

    resp = client.post(
        "/api/chat/stream",
        json={"session_id": "s-hitl-reject", "message": "订单 ORD20261121A 帮我建退货工单"},
    )
    assert resp.status_code == 200
    turn_id = resp.headers["x-turn-id"]
    frames = parse_sse(resp.text)
    assert frames[-1]["event"] == "approval_required"
    assert _count_tickets(order_id) == before_cnt

    reject_resp = client.post(f"/api/chat/turns/{turn_id}/approve", json={"approved": False})
    assert reject_resp.status_code == 200
    resume_frames = parse_sse(reject_resp.text)
    assert resume_frames[-1]["event"] == "done"
    assert _count_tickets(order_id) == before_cnt

    done = resume_frames[-1]["data"]["data"]
    assert "取消" in done["reply"]
    assert "create_after_sales_ticket" not in done["tools_called"]


# ---------------------------------------------------------------------------
# 4. 同步接口 POST /api/chat 默认 auto_approve=True 向后兼容
# ---------------------------------------------------------------------------

def test_sync_chat_auto_approves_ticket_creation(client: TestClient) -> None:
    """旧版同步 POST /api/chat 默认 auto_approve=True，无需人工二次确认即可直接建单返回。"""
    set_llm_provider(_make_ticket_provider())
    order_id = "ORD20261121A"
    before_cnt = _count_tickets(order_id)

    resp = client.post(
        "/api/chat",
        json={"session_id": "s-sync-compat", "message": "订单 ORD20261121A 全新未拆封，直接帮我建退货工单"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert _count_tickets(order_id) == before_cnt + 1
    assert body["tools_called"] == ["query_order", "create_after_sales_ticket"]
    assert "TKT" in body["reply"]
