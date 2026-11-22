"""
eval/run_eval.py

企业智能售后与工单分流 Agent 自动化端到端评测套件。
特点：
  - 零重型外部依赖：使用 Python 标准库 urllib.request 进行 HTTP 调用；
  - 严格全维度断言：工具路由、知识库命中、结论召回、全局防技术泄漏；
  - 数据库无污染保障：评测前后自动快照工单表，评测结束自动清理新增测试工单；
  - 双重报告输出：终端实时表格输出 + 自动生成 Markdown 报告保存至 eval/eval_report.md。
"""

import json
import os
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# ---------------------------------------------------------------------------
# 路径与端点配置（基于当前脚本文件绝对路径定位）
# ---------------------------------------------------------------------------
_EVAL_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _EVAL_DIR.parent
_TEST_CASES_PATH = _EVAL_DIR / "test_cases.json"
_REPORT_PATH = _EVAL_DIR / "eval_report.md"
_DB_PATH = _PROJECT_ROOT / "data" / "smart_cs.db"

_BASE_URL = os.getenv("EVAL_BASE_URL", "http://127.0.0.1:8000")
_CHAT_API = f"{_BASE_URL}/api/chat"
_HEALTH_API = f"{_BASE_URL}/health"


# ---------------------------------------------------------------------------
# 数据库辅助与测试无污染清理
# ---------------------------------------------------------------------------

def _get_current_ticket_ids() -> Set[str]:
    """
    查询当前 tickets 表中已有的工单 ID 集合作为快照。
    """
    if not _DB_PATH.exists():
        return set()
    try:
        with sqlite3.connect(_DB_PATH) as conn:
            cursor = conn.cursor()
            # 确认 tickets 表是否存在
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tickets';")
            if not cursor.fetchone():
                return set()
            cursor.execute("SELECT ticket_id FROM tickets;")
            return {str(row[0]) for row in cursor.fetchall()}
    except Exception as e:
        print(f"[警告] 读取初始工单集合失败: {e}", file=sys.stderr)
        return set()


def _cleanup_test_artifacts(initial_ticket_ids: Set[str]) -> None:
    """
    清理评测期间新写入的测试工单和评测会话记录，恢复数据库初始状态。
    """
    if not _DB_PATH.exists():
        return
    try:
        with sqlite3.connect(_DB_PATH) as conn:
            cursor = conn.cursor()
            # 1. 清理 tickets 表新产生的工单
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tickets';")
            if cursor.fetchone():
                if initial_ticket_ids:
                    placeholders = ",".join("?" for _ in initial_ticket_ids)
                    cursor.execute(
                        f"DELETE FROM tickets WHERE ticket_id NOT IN ({placeholders});",
                        tuple(initial_ticket_ids),
                    )
                else:
                    cursor.execute("DELETE FROM tickets;")

            # 2. 清理 chat_logs 中以 eval_session_ 开头的评测对话日志
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chat_logs';")
            if cursor.fetchone():
                cursor.execute("DELETE FROM chat_logs WHERE session_id LIKE 'eval_session_%';")

            conn.commit()
            print("[清理] 评测产生的数据已成功清理，数据库已恢复原状。")
    except Exception as e:
        print(f"[警告] 清理评测数据时异常: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# HTTP 调用封装（标准库 urllib，单条超时 60s）
# ---------------------------------------------------------------------------

def _check_service_alive() -> bool:
    """检查后端服务是否存活"""
    try:
        req = urllib.request.Request(_HEALTH_API, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def _send_chat_request(session_id: str, message: str) -> Dict[str, Any]:
    """向 POST /api/chat 发送请求"""
    payload = json.dumps({"session_id": session_id, "message": message}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        _CHAT_API,
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body)


# ---------------------------------------------------------------------------
# 单用例校验逻辑
# ---------------------------------------------------------------------------

def _evaluate_case(case: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any], int]:
    """
    评测单个用例：
    返回 (is_pass, failure_reasons, response_data, duration_ms)
    """
    cid = case["id"]
    query = case["query"]
    session_id = case["session_id"]
    req_tools = set(case.get("required_tools", []))
    forbid_tools = set(case.get("forbidden_tools", []))
    expect_citations = case.get("expect_citations", False)
    must_contain_any = case.get("must_contain_any", [])
    must_not_contain = case.get("must_not_contain", [])

    start_t = time.perf_counter()
    reasons: List[str] = []
    res: Dict[str, Any] = {}

    try:
        res = _send_chat_request(session_id, query)
        duration_ms = int((time.perf_counter() - start_t) * 1000)
    except Exception as e:
        duration_ms = int((time.perf_counter() - start_t) * 1000)
        reasons.append(f"HTTP请求异常: {str(e)}")
        return False, reasons, {}, duration_ms

    actual_tools = set(res.get("tools_called", []))
    actual_citations = res.get("citations", [])
    reply = res.get("reply", "")

    # 1. 必调工具子集校验
    if not req_tools.issubset(actual_tools):
        missing = req_tools - actual_tools
        reasons.append(f"缺失必调工具: {list(missing)} (实际调用: {list(actual_tools)})")

    # 2. 禁调工具无交集校验
    if not forbid_tools.isdisjoint(actual_tools):
        hit_forbidden = forbid_tools.intersection(actual_tools)
        reasons.append(f"误触发禁用工具: {list(hit_forbidden)}")

    # 3. 知识库引用期望校验（True: 必须命中, False: 必须无引用, None: 政策引用可选放行）
    if expect_citations is True and len(actual_citations) == 0:
        reasons.append("期望命中知识库条款(citations > 0)，实际为 0 条")
    elif expect_citations is False and len(actual_citations) > 0:
        reasons.append(f"期望无知识库引用，实际产生 {len(actual_citations)} 条引用")

    # 4. 核心关键结论关键词命中
    if must_contain_any:
        if not any(kw in reply for kw in must_contain_any):
            reasons.append(f"回复未包含预期结论关键词之一 {must_contain_any}")

    # 5. 严禁包含词命中
    if must_not_contain:
        hit_not_contain = [kw for kw in must_not_contain if kw in reply]
        if hit_not_contain:
            reasons.append(f"回复中包含了不应出现的违规错误词 {hit_not_contain}")

    # 6. 全局安全性：绝不暴露内部 .md 路径
    if ".md" in reply:
        reasons.append("违反全局规范：回复正文暴露了内部 Markdown 文件名 (.md)")

    is_pass = len(reasons) == 0
    return is_pass, reasons, res, duration_ms


# ---------------------------------------------------------------------------
# 评测报告生成
# ---------------------------------------------------------------------------

def _generate_markdown_report(
    summary: Dict[str, Any],
    case_results: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """生成 Markdown 格式评测报告"""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md = [
        "# Smart CS Agent 自动化评测报告",
        "",
        f"> 评测时间：`{now_str}`  ",
        f"> 服务端点：`{_BASE_URL}`  ",
        f"> 测试用例集：`eval/test_cases.json`（共 {summary['total']} 条）",
        "",
        "## 一、评测指标概览",
        "",
        "| 指标名称 | 测定值 | 达标目标 | 结果判定 |",
        "| :--- | :---: | :---: | :---: |",
        f"| **总体用例通过率 (Pass Rate)** | **{summary['pass_rate']:.1f}%** ({summary['passed']}/{summary['total']}) | $\\ge 90\\%$ | {'✅ PASS' if summary['pass_rate'] >= 90 else '❌ FAIL'} |",
        f"| **工具路由准确率 (Tool Accuracy)** | **{summary['tool_accuracy']:.1f}%** | $\\ge 95\\%$ | {'✅ PASS' if summary['tool_accuracy'] >= 95 else '❌ FAIL'} |",
        f"| **防幻觉与对客安全率 (Safety Rate)** | **{summary['safety_rate']:.1f}%** | $100\\%$ | {'✅ PASS' if summary['safety_rate'] >= 100 else '❌ FAIL'} |",
        f"| **平均端到端响应耗时 (Avg Latency)** | **{summary['avg_latency']} ms** | $\\le 4000\\text{{ms}}$ | {'✅ PASS' if summary['avg_latency'] <= 4000 else '⚠️ WARN'} |",
        f"| **P95 响应耗时 (P95 Latency)** | **{summary['p95_latency']} ms** | $\\le 6500\\text{{ms}}$ | {'✅ PASS' if summary['p95_latency'] <= 6500 else '⚠️ WARN'} |",
        "",
        "---",
        "",
        "## 二、各分类表现明细",
        "",
        "| 用例分类 | 用例数 | 通过数 | 通过率 | 平均耗时 |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]

    # 分类聚合
    category_map: Dict[str, List[Dict[str, Any]]] = {}
    for r in case_results:
        cat = r["category"]
        category_map.setdefault(cat, []).append(r)

    for cat, items in category_map.items():
        cat_total = len(items)
        cat_passed = sum(1 for x in items if x["is_pass"])
        cat_rate = (cat_passed / cat_total) * 100
        cat_avg_t = int(sum(x["duration_ms"] for x in items) / cat_total)
        md.append(f"| {cat} | {cat_total} | {cat_passed} | {cat_rate:.1f}% | {cat_avg_t} ms |")

    md.extend([
        "",
        "---",
        "",
        "## 三、逐用例执行记录",
        "",
        "| ID | 场景分类 | 用户输入 (Query) | 判定 | 实际工具调用 | 耗时 | 备注 / 失败原因 |",
        "| :---: | :--- | :--- | :---: | :--- | :---: | :--- |",
    ])

    for r in case_results:
        status_tag = "✅ PASS" if r["is_pass"] else "❌ FAIL"
        tools_str = ", ".join(r["actual_tools"]) if r["actual_tools"] else "*(直接回复)*"
        reasons = list(r["reasons"])
        if not r["is_pass"]:
            if r["actual_citations"]:
                reasons.append(f"实际引用: `{r['actual_citations']}`")
            if r.get("reply"):
                clean_reply = r["reply"].replace("\n", " ").replace("|", "\\|")
                reasons.append(f"实际回复摘要: {clean_reply[:100]}...")
        reasons_str = "<br>".join(reasons) if reasons else "符合预期"
        clean_query = r["query"].replace("|", "\\|")
        md.append(
            f"| {r['id']:02d} | {r['category']} | {clean_query} | {status_tag} | `{tools_str}` | {r['duration_ms']} ms | {reasons_str} |"
        )

    md.append("")
    output_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\n[报告] 评测报告已成功写入：{output_path.resolve()}")


# ---------------------------------------------------------------------------
# 主执行入口
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print("   Smart CS Agent 自动化端到端业务评测系统   ")
    print("=" * 72)

    # 1. 检查测试用例文件
    if not _TEST_CASES_PATH.exists():
        print(f"[错误] 评测用例文件未找到: {_TEST_CASES_PATH}", file=sys.stderr)
        sys.exit(1)

    with open(_TEST_CASES_PATH, "r", encoding="utf-8") as f:
        test_cases: List[Dict[str, Any]] = json.load(f)

    # 2. 检查后端服务状态
    print(f"[探测] 正在连接后端服务: {_BASE_URL} ...")
    if not _check_service_alive():
        print(
            f"[错误] 无法连接到服务 {_HEALTH_API}！\n"
            "请确认已在另一个终端启动 FastAPI 服务：\n"
            "  .venv\\Scripts\\python.exe -m uvicorn app.main:app --reload\n",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"[就绪] 后端服务在线，开始执行 {len(test_cases)} 条测试用例。\n")

    # 3. 记录评测前初始工单 ID 集合（保障测试后无污染恢复）
    initial_ticket_ids = _get_current_ticket_ids()
    print(f"[快照] 数据库现有工单数量: {len(initial_ticket_ids)} 条（评测后将自动清理新增项）\n")

    case_results: List[Dict[str, Any]] = []

    try:
        header = f"{'ID':<4} | {'分类':<14} | {'状态':<6} | {'耗时':<8} | {'实际工具调用':<30}"
        print(header)
        print("-" * len(header.encode("gbk", errors="ignore")))

        tool_correct_count = 0
        safety_correct_count = 0

        for case in test_cases:
            cid = case["id"]
            cat = case["category"]
            query = case["query"]

            is_pass, reasons, res, duration_ms = _evaluate_case(case)
            actual_tools = res.get("tools_called", [])
            reply = res.get("reply", "")

            # 统计工具正确率
            req_tools = set(case.get("required_tools", []))
            forbid_tools = set(case.get("forbidden_tools", []))
            tool_ok = req_tools.issubset(set(actual_tools)) and forbid_tools.isdisjoint(set(actual_tools))
            if tool_ok:
                tool_correct_count += 1

            # 统计安全与防幻觉：无 .md 且超纲/假单号正常拒绝
            safe_ok = (".md" not in reply) and (
                not (cid in [19, 20] and not is_pass)
            )
            if safe_ok:
                safety_correct_count += 1

            status_text = "PASS" if is_pass else "FAIL"
            tools_str = ",".join(actual_tools) if actual_tools else "(direct)"

            # 控制台输出单条进度
            print(f"#{cid:02d} | {cat:<12} | {status_text:<6} | {duration_ms:>5}ms | {tools_str:<28}")
            if not is_pass:
                for r in reasons:
                    print(f"     ↳ [原因] {r}")
                if res.get("citations"):
                    print(f"     ↳ [实际Citations] {res.get('citations')}")
                if reply:
                    print(f"     ↳ [实际回复] {reply[:100]}...")

            case_results.append({
                "id": cid,
                "category": cat,
                "query": query,
                "is_pass": is_pass,
                "reasons": reasons,
                "actual_tools": actual_tools,
                "actual_citations": res.get("citations", []),
                "duration_ms": duration_ms,
                "reply": reply,
            })

            # 适度微休眠避免过载本地大模型并发
            time.sleep(0.3)

    finally:
        # 数据无污染清理，无论中途有无异常必然执行
        print("\n" + "-" * 72)
        _cleanup_test_artifacts(initial_ticket_ids)

    # 4. 指标统计
    total_count = len(case_results)
    passed_count = sum(1 for c in case_results if c["is_pass"])
    pass_rate = (passed_count / total_count * 100) if total_count > 0 else 0.0
    tool_accuracy = (tool_correct_count / total_count * 100) if total_count > 0 else 0.0
    safety_rate = (safety_correct_count / total_count * 100) if total_count > 0 else 0.0

    durations = sorted([c["duration_ms"] for c in case_results])
    avg_latency = int(sum(durations) / len(durations)) if durations else 0
    p95_idx = int(len(durations) * 0.95)
    p95_latency = durations[min(p95_idx, len(durations) - 1)] if durations else 0

    summary = {
        "total": total_count,
        "passed": passed_count,
        "failed": total_count - passed_count,
        "pass_rate": pass_rate,
        "tool_accuracy": tool_accuracy,
        "safety_rate": safety_rate,
        "avg_latency": avg_latency,
        "p95_latency": p95_latency,
    }

    print("\n" + "=" * 72)
    print("                     评测总结统计                     ")
    print("=" * 72)
    print(f"总测试用例数:        {total_count}")
    print(f"通过用例数:          {passed_count} / {total_count}")
    print(f"综合通过率 (Pass):   {pass_rate:.1f}%")
    print(f"工具路由准确率:      {tool_accuracy:.1f}%")
    print(f"防幻觉与对客安全率:  {safety_rate:.1f}%")
    print(f"平均响应耗时:        {avg_latency} ms")
    print(f"P95 响应耗时:        {p95_latency} ms")
    print("=" * 72)

    # 5. 生成 Markdown 评测报告
    _generate_markdown_report(summary, case_results, _REPORT_PATH)


if __name__ == "__main__":
    main()
