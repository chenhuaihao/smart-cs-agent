"""
tests/test_p4_concurrency_and_api.py

v3.0 P3 并发压力、限流熔断边界与业务 API 状态机回归测试（全离线、零外网依赖）：
  1. SQLite 事件表高并发写入：多 turn 并行 + 单 turn 多线程竞争，验证 seq 严格连续无空洞且终态后拒写；
  2. 单 IP 滑动窗口限流器并发边界：多线程并发抢占严格不超过 limit，返回 HTTP 429 与 Retry-After；
  3. 每日全局配额熔断器并发边界：多线程并发抢占严格不超过 DAILY_CHAT_LIMIT；
  4. 真实 IP 解析（X-Forwarded-For 多跳 / X-Real-IP / 直连回退）；
  5. 工单严格单向状态机（待处理 → 处理中 → 已完成，禁逆向流转与终态变更）；
  6. 订单查询、工单创建边界、演示数据重置口令/冷却双层保护、只读评测摘要接口契约校验。
"""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.core.limiter import (
    DailyQuotaLimiter,
    SlidingWindowLimiter,
    _demo_reset_limiter,
    get_client_ip,
)
from app.db import database
from app.db import events as event_store
from app.main import app


@pytest.fixture()
def isolated_api_client(tmp_path, monkeypatch) -> TestClient:
    """使用独立临时 SQLite 库启动完整 FastAPI 应用（离线纯 BM25 模式）。"""
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "v3_api_concurrency.db"))
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "1000")
    monkeypatch.setenv("API_READ_RATE_LIMIT_PER_MINUTE", "1000")
    monkeypatch.setenv("API_WRITE_RATE_LIMIT_PER_MINUTE", "1000")
    monkeypatch.setenv("DAILY_CHAT_LIMIT", "100000")
    monkeypatch.setenv("EMBEDDING_API_KEY", "your_embedding_api_key_here")
    monkeypatch.delenv("DEMO_ADMIN_TOKEN", raising=False)
    _demo_reset_limiter._history.clear()

    with TestClient(app) as client:
        yield client
    _demo_reset_limiter._history.clear()


# ---------------------------------------------------------------------------
# 1. SQLite 事件表高并发 seq 严格单调递增与终态写保护
# ---------------------------------------------------------------------------

def test_sqlite_event_store_multi_turn_high_concurrency_no_gaps(tmp_path, monkeypatch) -> None:
    """5 个 turn × 每 turn 15 条并发事件（共 75 并发写），每个 turn 的 seq 必须严格为 1..15。"""
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "v3_events_conc.db"))
    database.init_db()

    turn_ids = [event_store.create_turn(f"sess-{i}", f"msg-{i}") for i in range(5)]
    jobs = [(tid, idx) for tid in turn_ids for idx in range(15)]

    def _write(job: tuple[str, int]) -> int:
        tid, idx = job
        rec = event_store.append_event(tid, "token", {"delta": f"chunk-{idx}"})
        return rec["seq"]

    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(_write, jobs))

    assert len(results) == 75
    for tid in turn_ids:
        events = event_store.get_events_after(tid, after_seq=0)
        seqs = [e["seq"] for e in events]
        assert seqs == list(range(1, 16)), f"turn={tid} seq 出现空洞或乱序：{seqs}"

    # 终态写入后严禁再追加事件
    target = turn_ids[0]
    event_store.finish_turn(
        target,
        event_store.TURN_COMPLETED,
        "done",
        {"reply": "ok"},
        "ok",
        [],
        [],
        10,
        20,
        None,
    )
    with pytest.raises(ValueError, match="已结束"):
        event_store.append_event(target, "token", {"delta": "late"})


# ---------------------------------------------------------------------------
# 2. 限流器与每日配额熔断器并发线程安全测试
# ---------------------------------------------------------------------------

def test_sliding_window_limiter_thread_safety_and_retry_after() -> None:
    """20 线程同时冲撞 limit=6 的单 IP 滑动窗口，严格放行 6 次、拦截 14 次并携带 Retry-After。"""
    limiter = SlidingWindowLimiter(window_seconds=60, default_limit=6, env_var="__TEST_UNUSED_RATE__")

    def _attempt(_: int) -> tuple[bool, int | None]:
        try:
            limiter.check("203.0.113.9")
            return True, None
        except HTTPException as exc:
            assert exc.status_code == 429
            retry_after = int((exc.headers or {})["Retry-After"])
            return False, retry_after

    with ThreadPoolExecutor(max_workers=20) as pool:
        outcomes = list(pool.map(_attempt, range(20)))

    passed = [ok for ok, _ in outcomes if ok]
    blocked_retries = [ra for ok, ra in outcomes if not ok]
    assert len(passed) == 6
    assert len(blocked_retries) == 14
    assert all(1 <= ra <= 61 for ra in blocked_retries)


def test_daily_quota_limiter_thread_safety() -> None:
    """20 线程同时冲撞 limit=7 的全站每日熔断器，严格放行 7 次、熔断 13 次。"""
    quota = DailyQuotaLimiter(default_limit=7, default_ip_limit=7)

    def _attempt(_: int) -> bool:
        try:
            quota.check_and_increment()
            return True
        except HTTPException as exc:
            assert exc.status_code == 429
            return False

    with ThreadPoolExecutor(max_workers=20) as pool:
        outcomes = list(pool.map(_attempt, range(20)))

    assert outcomes.count(True) == 7
    assert outcomes.count(False) == 13
    assert quota.get_stats()["used"] == 7


def test_daily_quota_per_ip_isolation_protects_other_visitors() -> None:
    """单 IP 达到每日上限后仅拦截该 IP，绝不耗尽全站额度影响其他真实访客（如面试官）。"""
    quota = DailyQuotaLimiter(default_limit=10, default_ip_limit=3)
    for _ in range(3):
        quota.check_and_increment("198.51.100.10")
    with pytest.raises(HTTPException) as exc_info:
        quota.check_and_increment("198.51.100.10")
    assert exc_info.value.status_code == 429
    assert "单 IP 每日限 3 次" in str(exc_info.value.detail)

    # 另一个独立 IP 仍可正常访问
    quota.check_and_increment("198.51.100.20")
    assert quota.get_stats()["used"] == 4
    assert quota.get_stats()["active_ips"] == 2


def test_get_client_ip_proxy_trust_boundary(monkeypatch) -> None:
    """
    验证代理头信任边界（修复直连公网伪造 X-Forwarded-For 绕过限流漏洞）：
      - 默认 TRUST_PROXY_HEADERS=false：忽略伪造的 X-Forwarded-For / X-Real-IP，强制使用 TCP 对端 IP；
      - 显式 TRUST_PROXY_HEADERS=true：按 X-Forwarded-For 首跳 > X-Real-IP > request.client.host 解析。
    """
    spoofed_req = SimpleNamespace(
        headers={"X-Forwarded-For": "1.2.3.4, 10.0.0.1", "X-Real-IP": "5.6.7.8"},
        client=SimpleNamespace(host="192.168.1.55"),
    )

    # 默认直连模式：忽略伪造头，返回真实 TCP host
    monkeypatch.delenv("TRUST_PROXY_HEADERS", raising=False)
    assert get_client_ip(spoofed_req) == "192.168.1.55"

    # 开启受信任反代模式：正确解析转发头优先级
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "true")
    assert get_client_ip(spoofed_req) == "1.2.3.4"

    req_real_ip = SimpleNamespace(headers={"X-Real-IP": "198.51.100.8"}, client=SimpleNamespace(host="127.0.0.1"))
    assert get_client_ip(req_real_ip) == "198.51.100.8"

    req_direct = SimpleNamespace(headers={}, client=SimpleNamespace(host="192.168.1.55"))
    assert get_client_ip(req_direct) == "192.168.1.55"

    req_fallback = SimpleNamespace(headers={}, client=None)
    assert get_client_ip(req_fallback) == "127.0.0.1"


# ---------------------------------------------------------------------------
# 3. 工单单向状态机、订单/知识库/评测摘要/演示重置双层保护 API 契约测试
# ---------------------------------------------------------------------------

def test_ticket_unidirectional_state_machine_and_invalid_order_guard(isolated_api_client: TestClient) -> None:
    """验证工单严格单向流转（待处理 -> 处理中 -> 已完成），禁止任何逆向回退或终态篡改。"""
    client = isolated_api_client

    # 不存在的订单禁止建单
    bad = client.post(
        "/api/tickets",
        json={
            "order_id": "ORD99999999",
            "issue_type": "退货",
            "reason": "测试无效订单",
            "summary": "不存在订单",
        },
    )
    assert bad.status_code == 404

    # 合法订单建单默认状态为「待处理」
    created = client.post(
        "/api/tickets",
        json={
            "order_id": "ORD20261121",
            "issue_type": "退货",
            "reason": "未拆封无理由退货",
            "summary": "状态机测试单",
        },
    )
    assert created.status_code == 201
    ticket = created.json()
    tid = ticket["ticket_id"]
    assert ticket["status"] == "待处理"

    # 待处理 -> 处理中（允许）
    step1 = client.patch(f"/api/tickets/{tid}", json={"status": "处理中"})
    assert step1.status_code == 200 and step1.json()["status"] == "处理中"

    # 处理中 -> 待处理（逆向流转，严禁）
    rev1 = client.patch(f"/api/tickets/{tid}", json={"status": "待处理"})
    assert rev1.status_code in (400, 409)

    # 处理中 -> 已完成（允许）
    step2 = client.patch(f"/api/tickets/{tid}", json={"status": "已完成"})
    assert step2.status_code == 200 and step2.json()["status"] == "已完成"

    # 已完成 -> 处理中 / 待处理（终态不可逆，严禁）
    for forbidden_target in ("处理中", "待处理"):
        rev2 = client.patch(f"/api/tickets/{tid}", json={"status": forbidden_target})
        assert rev2.status_code in (400, 409)


def test_demo_reset_dual_guard_cooldown_and_admin_token(isolated_api_client: TestClient, monkeypatch) -> None:
    """验证重置接口双层保护：无口令时触发冷却 429；有口令时校验 X-Demo-Admin-Token（403/200）。"""
    client = isolated_api_client

    # 无口令模式：首次 200，紧接着第二次触发 429 冷却
    r1 = client.post("/api/tickets/reset")
    assert r1.status_code == 200
    assert r1.json()["orders_count"] == 24

    r2 = client.post("/api/tickets/reset")
    assert r2.status_code == 429
    assert "Retry-After" in r2.headers

    # 开启口令保护模式：无口令或错口令 403，正确口令 200
    monkeypatch.setenv("DEMO_ADMIN_TOKEN", "v3-secret-pass")
    r3 = client.post("/api/tickets/reset")
    assert r3.status_code == 403
    r4 = client.post("/api/tickets/reset", headers={"X-Demo-Admin-Token": "wrong"})
    assert r4.status_code == 403
    r5 = client.post("/api/tickets/reset", headers={"X-Demo-Admin-Token": "v3-secret-pass"})
    assert r5.status_code == 200


def test_core_read_endpoints_contracts(isolated_api_client: TestClient) -> None:
    """核验健康检查、订单列表/详情、知识库检索与只读评测报告接口的核心契约。"""
    client = isolated_api_client

    assert client.get("/health").json()["status"] == "ok"

    orders = client.get("/api/orders").json()
    assert isinstance(orders, list) and len(orders) == 24

    assert client.get("/api/orders/ORD20261101").status_code == 200
    assert client.get("/api/orders/ORD00000000").status_code == 404

    kb = client.get("/api/knowledge/search", params={"q": "未拆封耳机退货", "top_k": 3})
    assert kb.status_code == 200
    kb_data = kb.json()
    assert kb_data["total_hits"] >= 1
    assert all("retrieval_source" in item for item in kb_data["results"])

    ev = client.get("/api/eval/summary")
    assert ev.status_code == 200
    ev_data = ev.json()
    assert "rag_configs" in ev_data and "e2e_summary" in ev_data


def test_eval_forbidden_keyword_negation_awareness() -> None:
    """验证 v3.0 P4 评测判定修复：否定/疑问语境不误伤，肯定违规词精准命中。"""
    from eval.run_eval import _is_affirmative_forbidden_hit

    # 否定与疑问复述语境：不应误判为违规（解决旧报告 #042/#052/#087 假阴性）
    assert _is_affirmative_forbidden_hit("定制商品不支持7天无理由退货。", "支持7天无理由退货") is False
    assert _is_affirmative_forbidden_hit("该商品已拆封且无质量问题，无法支持7天无理由退货。", "支持7天无理由退货") is False
    assert _is_affirmative_forbidden_hit("关于是否支持24期免息，本店知识库暂未收录。", "支持24期免息") is False

    # 肯定承诺语境：必须精准命中
    assert _is_affirmative_forbidden_hit("放心，本店支持24期免息分期。", "支持24期免息") is True
    assert _is_affirmative_forbidden_hit("好的，该定制商品支持7天无理由退货。", "支持7天无理由退货") is True
