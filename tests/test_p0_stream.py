"""
tests/test_p0_stream.py

v2.0 P0 验收测试（使用 Mock Provider，不消耗真实 API 额度）：
  1. POST /api/chat/stream 返回合法 SSE，事件顺序 stage → tool_call → citation → token → done，seq 严格递增；
  2. GET /api/chat/turns/{turn_id}/events?after_seq=N 准确补发断点后的事件，不重不漏，且不受对话限流；
  3. POST /api/chat 与 run_agent 旧契约 100% 兼容，多轮上下文（最近 6 轮）正常；
  4. 事件存储在并发写入下 seq 仍从 1 连续严格递增。
每个用例使用独立的临时 SQLite 文件，不污染 data/smart_cs.db。
"""

import copy
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncIterator

import pytest
from fastapi.testclient import TestClient

from app.agent.core import run_agent
from app.db import database
from app.db import events as event_store
from app.llm.base import BaseLLMProvider, LLMStreamChunk, ToolCallRequest
from app.llm.factory import set_llm_provider
from app.main import app

KB_QUESTION = "耳机拆封后还能退货吗？"
KB_REPLY_PIECES = ["根据店铺退货规则，", "耳机类商品拆封后", "如不影响二次销售可申请退货。"]


class MockProvider(BaseLLMProvider):
    """
    按最后一条消息决定行为的确定性 Mock：
      - 用户问"退货" → 请求调用 search_knowledge_base；
      - 最后一条是 tool 结果 → 分 3 段流式输出最终回复；
      - 其他（闲聊）→ 直接流式输出文本。
    同时记录每次收到的 messages，用于断言多轮上下文。
    """

    name = "mock"

    def __init__(self) -> None:
        self.calls: list[list[dict[str, Any]]] = []

    async def astream_chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[LLMStreamChunk]:
        self.calls.append(copy.deepcopy(messages))
        last = messages[-1]
        if last["role"] == "tool":
            for piece in KB_REPLY_PIECES:
                yield LLMStreamChunk(type="text_delta", text=piece)
            return
        if "退货" in (last.get("content") or ""):
            args = json.dumps({"query": last["content"]}, ensure_ascii=False)
            yield LLMStreamChunk(
                type="tool_calls",
                tool_calls=[ToolCallRequest(id="call_kb_1", name="search_knowledge_base", arguments=args)],
            )
            return
        for piece in ["您好，", "我是小智，", f"第{len(self.calls)}次为您服务。"]:
            yield LLMStreamChunk(type="text_delta", text=piece)


@pytest.fixture()
def client(tmp_path, monkeypatch) -> TestClient:
    """独立临时库 + 放宽限流 + 注入 Mock Provider；进入 TestClient 上下文会触发 lifespan 建表与种子数据。"""
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "test_smart_cs.db"))
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "1000")
    monkeypatch.setenv("DAILY_CHAT_LIMIT", "100000")
    provider = MockProvider()
    set_llm_provider(provider)
    with TestClient(app) as test_client:
        test_client.provider = provider  # type: ignore[attr-defined]
        yield test_client
    set_llm_provider(None)


def parse_sse(body: str) -> list[dict[str, Any]]:
    """把 SSE 文本解析为 [{id, event, data}]，并校验每帧三个字段齐全。"""
    frames = []
    for block in body.replace("\r\n", "\n").strip().split("\n\n"):
        fields: dict[str, str] = {}
        for line in block.split("\n"):
            key, sep, value = line.partition(": ")
            assert sep, f"非法 SSE 行：{line!r}"
            fields[key] = value
        assert set(fields) == {"id", "event", "data"}, f"SSE 帧字段不完整：{fields}"
        frames.append({"id": int(fields["id"]), "event": fields["event"], "data": json.loads(fields["data"])})
    return frames


def _stream(client: TestClient, session_id: str, message: str) -> tuple[Any, list[dict[str, Any]]]:
    resp = client.post("/api/chat/stream", json={"session_id": session_id, "message": message})
    assert resp.status_code == 200, resp.text
    return resp, parse_sse(resp.text)


# ---------------------------------------------------------------------------
# 1. SSE 流式接口
# ---------------------------------------------------------------------------

def test_stream_emits_ordered_sse_frames(client: TestClient) -> None:
    resp, frames = _stream(client, "s-stream", KB_QUESTION)

    assert resp.headers["content-type"].startswith("text/event-stream")
    assert resp.headers["cache-control"] == "no-cache"
    assert resp.headers["x-accel-buffering"] == "no"
    turn_id = resp.headers["x-turn-id"]

    # seq 从 1 开始严格连续递增，且 SSE id / event 与 data 内的 seq / type 一致
    ids = [f["id"] for f in frames]
    assert ids == list(range(1, len(frames) + 1))
    for f in frames:
        assert f["data"]["seq"] == f["id"]
        assert f["data"]["type"] == f["event"]
        assert f["data"]["turn_id"] == turn_id

    types = [f["event"] for f in frames]
    first = {t: types.index(t) for t in ("stage", "tool_call", "citation", "token", "done")}
    assert first["stage"] < first["tool_call"] < first["citation"] < first["token"] < first["done"]
    assert types[-1] == "done" and types.count("done") == 1 and "error" not in types

    tool_evt = next(f["data"]["data"] for f in frames if f["event"] == "tool_call")
    assert tool_evt["name"] == "search_knowledge_base" and tool_evt["status"] == "success"

    cited = [c for f in frames if f["event"] == "citation" for c in f["data"]["data"]["citations"]]
    tokens = "".join(f["data"]["data"]["delta"] for f in frames if f["event"] == "token")
    done = frames[-1]["data"]["data"]
    assert done["reply"] == tokens == "".join(KB_REPLY_PIECES)
    assert done["tools_called"] == ["search_knowledge_base"]
    assert done["citations"] == cited and all("#" in c for c in cited)
    assert isinstance(done["ttft_ms"], int) and 0 <= done["ttft_ms"] <= done["latency_ms"]


# ---------------------------------------------------------------------------
# 2. 断线补发接口
# ---------------------------------------------------------------------------

def test_replay_after_seq_returns_remaining_events(client: TestClient, monkeypatch) -> None:
    resp, frames = _stream(client, "s-replay", KB_QUESTION)
    turn_id = resp.headers["x-turn-id"]
    sse_events = [f["data"] for f in frames]
    breakpoint_seq = 3  # 模拟客户端收到 seq=3 后断线

    # 收紧限流：补发接口不应受影响，流式接口应被拦截
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "1")
    for _ in range(5):
        r = client.get(f"/api/chat/turns/{turn_id}/events", params={"after_seq": breakpoint_seq})
        assert r.status_code == 200, r.text
    assert client.post("/api/chat/stream", json={"session_id": "x", "message": "hi"}).status_code == 429

    body = r.json()
    assert body["turn_id"] == turn_id and body["status"] == "completed"
    # 与实时 SSE 推送的剩余事件逐字段一致：不重复（无 seq<=3）、不遗漏
    assert body["events"] == sse_events[breakpoint_seq:]
    assert body["last_seq"] == len(sse_events)

    full = client.get(f"/api/chat/turns/{turn_id}/events").json()
    assert full["events"] == sse_events

    tail = client.get(f"/api/chat/turns/{turn_id}/events", params={"after_seq": len(sse_events)}).json()
    assert tail["events"] == [] and tail["last_seq"] is None

    assert client.get("/api/chat/turns/not-exist/events").status_code == 404
    assert client.get(f"/api/chat/turns/{turn_id}/events", params={"after_seq": -1}).status_code == 422


# ---------------------------------------------------------------------------
# 3. 旧接口兼容 + 多轮上下文
# ---------------------------------------------------------------------------

def test_legacy_chat_contract_and_multi_turn_memory(client: TestClient) -> None:
    provider: MockProvider = client.provider  # type: ignore[attr-defined]
    sid = "s-legacy"

    r1 = client.post("/api/chat", json={"session_id": sid, "message": "你好"})
    assert r1.status_code == 200
    b1 = r1.json()
    assert set(b1) == {"session_id", "reply", "tools_called", "latency_ms", "citations"}
    assert b1["session_id"] == sid and b1["tools_called"] == [] and b1["citations"] == []
    assert isinstance(b1["latency_ms"], int) and b1["reply"].startswith("您好")

    r2 = client.post("/api/chat", json={"session_id": sid, "message": KB_QUESTION})
    b2 = r2.json()
    assert b2["reply"] == "".join(KB_REPLY_PIECES)
    assert b2["tools_called"] == ["search_knowledge_base"] and b2["citations"]

    # 第 2 轮首次推理时，上下文中应带上第 1 轮的 user / assistant
    turn2_msgs = provider.calls[-2]
    assert turn2_msgs[0]["role"] == "system"
    assert {"role": "user", "content": "你好"} in turn2_msgs
    assert {"role": "assistant", "content": b1["reply"]} in turn2_msgs

    # 入参校验契约不变
    assert client.post("/api/chat", json={"session_id": sid, "message": ""}).status_code == 422

    # 同步兼容入口：返回 4 元组，且与 HTTP 接口共享同一 session 记忆
    result = run_agent(sid, "再问一下")
    assert isinstance(result, tuple) and len(result) == 4
    reply, tools_called, latency_ms, citations = result
    assert isinstance(reply, str) and tools_called == [] and isinstance(latency_ms, int) and citations == []
    assert {"role": "assistant", "content": b2["reply"]} in provider.calls[-1]

    # 最近 6 轮上限：再聊多轮后，上下文 = system + 12 条历史 + 本轮 user
    for i in range(5):
        run_agent(sid, f"闲聊 {i}")
    assert len(provider.calls[-1]) == 1 + 12 + 1


# ---------------------------------------------------------------------------
# 4. 事件存储并发安全
# ---------------------------------------------------------------------------

def test_append_event_seq_strictly_increasing_under_concurrency(client: TestClient) -> None:
    turn_id = event_store.create_turn("s-concurrency", "并发写入")
    with ThreadPoolExecutor(max_workers=8) as pool:
        records = list(pool.map(lambda i: event_store.append_event(turn_id, "token", {"i": i}), range(40)))

    assert sorted(r["seq"] for r in records) == list(range(1, 41))
    assert [e["seq"] for e in event_store.get_events_after(turn_id, 0)] == list(range(1, 41))
