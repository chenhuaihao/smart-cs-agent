"""
eval/run_eval.py

企业智能售后与工单分流 Agent 自动化端到端并发评测套件。
特点：
  - 零外部重型依赖：使用 Python 标准库 urllib.request + concurrent.futures；
  - 模块化聚合加载：自动按文件名顺序加载 eval/cases/*.json（支持 --category 过滤）；
  - 多线程并发评测：ThreadPoolExecutor 支持 --workers 控制并发度，单条遇超时自动重试 1 次；
  - 线程安全实时进度：终端实时打印格式化进度 [23/100] #023 | 政策检索 | PASS | 2410ms；
  - 数据库无污染保障：评测前后自动快照 tickets 与 chat_logs，评测后自动清理测试痕迹；
  - 结构化双重报告：终端分类汇总表 + 自动生成 Markdown 报告保存至 eval/eval_report.md。
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import os
from pathlib import Path
import sqlite3
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import urllib.error
import urllib.request

# ---------------------------------------------------------------------------
# 路径与端点配置（基于当前脚本文件绝对路径定位）
# ---------------------------------------------------------------------------
_EVAL_DIR = Path(__file__).resolve().parent
_CASES_DIR = _EVAL_DIR / "cases"
_PROJECT_ROOT = _EVAL_DIR.parent
_REPORT_PATH = _EVAL_DIR / "eval_report.md"
_SUMMARY_JSON_PATH = _EVAL_DIR / "eval_summary.json"
_DB_PATH = _PROJECT_ROOT / "data" / "smart_cs.db"

# 否定/疑问前缀：用于避免“不支持7天无理由退货”“是否支持24期免息”被静态黑名单子串误伤
_NEGATION_PREFIXES = (
    "不", "无法", "不能", "暂不", "并未", "未", "没有", "无权",
    "不可", "是否", "能否", "拒绝", "禁止", "并非", "绝不", "难以",
)


def _is_affirmative_forbidden_hit(reply: str, kw: str) -> bool:
    """
    判断违规词 kw 是否在肯定语境下真实命中（v3.0 P4：消除否定句/复述问句假阴性）。
    示例：
      - reply="该商品不支持7天无理由退货", kw="支持7天无理由退货" -> False（紧邻否定前缀“不”）
      - reply="关于是否支持24期免息，本店暂未收录", kw="支持24期免息" -> False（紧邻疑问前缀“是否”）
      - reply="好的，支持7天无理由退货", kw="支持7天无理由退货" -> True（肯定命中）
    """
    if not kw or kw not in reply:
        return False

    start = 0
    while True:
        idx = reply.find(kw, start)
        if idx == -1:
            return False
        prefix_window = reply[max(0, idx - 6):idx].strip()
        if not any(prefix_window.endswith(neg) for neg in _NEGATION_PREFIXES):
            return True
        start = idx + len(kw)

_DEFAULT_BASE_URL = os.getenv("EVAL_BASE_URL", "http://127.0.0.1:8000")


# ---------------------------------------------------------------------------
# 用例加载逻辑
# ---------------------------------------------------------------------------

def load_test_cases(category_filter: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    按文件名升序加载 eval/cases/*.json 中的测试用例。
    支持按 category_filter（匹配模块文件名或用例分类）过滤，支持 limit 截断。
    若 eval/cases 目录不存在，回退尝试读取 eval/test_cases.json。
    """
    cases: List[Dict[str, Any]] = []

    if _CASES_DIR.exists() and _CASES_DIR.is_dir():
        case_files = sorted(_CASES_DIR.glob("*.json"))
        for fpath in case_files:
            file_name = fpath.stem
            # 如果指定了 category 过滤且当前文件名和分类均未命中，跳过
            try:
                data = json.loads(fpath.read_text(encoding="utf-8"))
                if not isinstance(data, list):
                    continue
                for item in data:
                    item["_source_file"] = fpath.name
                    # 识别 5 大场景模块名称
                    if "01_" in file_name:
                        item["_module"] = "1. 意图路由与闲聊"
                    elif "02_" in file_name:
                        item["_module"] = "2. 政策检索与知识库"
                    elif "03_" in file_name:
                        item["_module"] = "3. 时间与状态边界推理"
                    elif "04_" in file_name:
                        item["_module"] = "4. 工单工作流与决策闭环"
                    elif "05_" in file_name:
                        item["_module"] = "5. 安全边界与防幻觉"
                    else:
                        item["_module"] = item.get("category", "其他场景")
                    cases.append(item)
            except Exception as e:
                print(f"[警告] 解析用例文件 {fpath.name} 失败: {e}", file=sys.stderr)
    else:
        # 回退模式：兼容旧单文件 test_cases.json
        legacy_file = _EVAL_DIR / "test_cases.json"
        if legacy_file.exists():
            data = json.loads(legacy_file.read_text(encoding="utf-8"))
            for item in data:
                item["_source_file"] = "test_cases.json"
                item["_module"] = item.get("category", "综合测试")
                cases.append(item)

    # 过滤分类
    if category_filter:
        cf = category_filter.lower()
        cases = [
            c for c in cases
            if cf in c.get("category", "").lower()
            or cf in c.get("_source_file", "").lower()
            or cf in c.get("_module", "").lower()
        ]

    # 按 id 升序排序
    cases.sort(key=lambda x: x.get("id", 0))

    # 限制用例数量
    if limit and limit > 0:
        cases = cases[:limit]

    return cases


# ---------------------------------------------------------------------------
# 数据库辅助与测试无污染清理
# ---------------------------------------------------------------------------

def _get_current_ticket_ids() -> Set[str]:
    """查询当前 tickets 表中已有的工单 ID 集合作为快照"""
    if not _DB_PATH.exists():
        return set()
    try:
        with sqlite3.connect(_DB_PATH, timeout=15.0) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tickets';")
            if not cursor.fetchone():
                return set()
            cursor.execute("SELECT ticket_id FROM tickets;")
            return {str(row[0]) for row in cursor.fetchall()}
    except Exception as e:
        print(f"[警告] 读取初始工单集合失败: {e}", file=sys.stderr)
        return set()


def _cleanup_test_artifacts(initial_ticket_ids: Set[str]) -> None:
    """清理评测期间新写入的测试工单和评测会话记录，恢复数据库初始状态"""
    if not _DB_PATH.exists():
        return
    try:
        with sqlite3.connect(_DB_PATH, timeout=15.0) as conn:
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
            print("[清理] 评测新增工单与测试对话日志已成功清除，数据库看板已恢复原状。")
    except Exception as e:
        print(f"[警告] 清理评测数据时异常: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# HTTP 调用封装（带自动重试）
# ---------------------------------------------------------------------------

def _check_service_alive(health_api: str) -> bool:
    """检查后端服务是否在线"""
    try:
        req = urllib.request.Request(health_api, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def _send_chat_request(chat_api: str, session_id: str, message: str, timeout: int = 60) -> Dict[str, Any]:
    """向 POST /api/chat 发送请求，遇到瞬时超时或连接异常自动重试 1 次"""
    payload = json.dumps({"session_id": session_id, "message": message}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        chat_api,
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    last_err: Optional[Exception] = None
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_err = e
            if attempt == 0:
                time.sleep(1.0)
                continue
    raise last_err if last_err else RuntimeError("HTTP 请求异常")


# ---------------------------------------------------------------------------
# 单用例评测校验逻辑
# ---------------------------------------------------------------------------

def _evaluate_case(case: Dict[str, Any], chat_api: str) -> Tuple[bool, List[str], Dict[str, Any], int]:
    """
    评测单个用例：
    返回 (is_pass, failure_reasons, response_data, duration_ms)
    """
    query = case["query"]
    session_id = case.get("session_id", f"eval_session_{case.get('id', 0):03d}")
    req_tools = set(case.get("required_tools", []))
    forbid_tools = set(case.get("forbidden_tools", []))
    expect_citations = case.get("expect_citations", False)
    must_contain_any = case.get("must_contain_any", [])
    must_not_contain = case.get("must_not_contain", [])

    start_t = time.perf_counter()
    reasons: List[str] = []
    res: Dict[str, Any] = {}

    try:
        res = _send_chat_request(chat_api, session_id, query)
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
            reasons.append(f"回复未包含预期结论关键词之一: {must_contain_any}")

    # 5. 严禁包含词命中（v3.0 P4：否定/疑问语境感知，避免“不支持/是否支持”触发静态子串假阴性）
    if must_not_contain:
        hit_not_contain = [kw for kw in must_not_contain if _is_affirmative_forbidden_hit(reply, kw)]
        if hit_not_contain:
            reasons.append(f"回复中包含了不应出现的违规词: {hit_not_contain}")

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
    base_url: str,
) -> None:
    """生成 Markdown 格式自动化评测报告"""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md = [
        "# Smart CS Agent 自动化端到端业务评测报告",
        "",
        f"> 评测时间：`{now_str}`  ",
        f"> 服务端点：`{base_url}`  ",
        f"> 并发线程：`{summary.get('workers', 4)}`  ",
        f"> 评测用例集：共 `{summary['total']}` 条（覆盖 5 大模块场景）",
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
        "## 二、5 大业务场景模块通过率明细",
        "",
        "| 场景模块 | 用例数 | 通过数 | 通过率 | 平均耗时 | 达标判定 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    # 按 5 大模块聚合
    module_map: Dict[str, List[Dict[str, Any]]] = {}
    for r in case_results:
        mod = r.get("module", r.get("category", "默认分类"))
        module_map.setdefault(mod, []).append(r)

    for mod_name in sorted(module_map.keys()):
        items = module_map[mod_name]
        m_total = len(items)
        m_passed = sum(1 for x in items if x["is_pass"])
        m_rate = (m_passed / m_total) * 100 if m_total > 0 else 0.0
        m_avg_t = int(sum(x["duration_ms"] for x in items) / m_total) if m_total > 0 else 0
        m_status = "✅ PASS" if m_rate >= 90.0 else "❌ FAIL"
        md.append(f"| {mod_name} | {m_total} | {m_passed} | {m_rate:.1f}% | {m_avg_t} ms | {m_status} |")

    # 失败用例明细表
    failed_items = [r for r in case_results if not r["is_pass"]]
    md.extend([
        "",
        "---",
        "",
        "## 三、失败用例明细与归因",
        "",
    ])

    if not failed_items:
        md.append("🎉 **全量用例 100% 顺利通过！未发现任何异常或失败用例。**")
    else:
        md.extend([
            "| ID | 场景模块 | 用户输入 (Query) | 实际工具调用 | 失败原因 / 期望与实际差异 |",
            "| :---: | :--- | :--- | :--- | :--- |",
        ])
        for r in failed_items:
            clean_q = r["query"].replace("|", "\\|")
            tools_str = ", ".join(r["actual_tools"]) if r["actual_tools"] else "*(直接回复)*"
            reasons_str = "<br>".join(r["reasons"])
            if r.get("reply"):
                clean_reply = r["reply"].replace("\n", " ").replace("|", "\\|")
                reasons_str += f"<br>*(实际回复: {clean_reply[:90]}...)*"
            md.append(f"| #{r['id']:03d} | {r.get('module', r['category'])} | {clean_q} | `{tools_str}` | {reasons_str} |")

    md.extend([
        "",
        "---",
        "",
        "## 四、全部评测用例执行记录清单",
        "",
        "| ID | 场景分类 | 用户输入 (Query) | 判定 | 实际工具调用 | 耗时 | 备注 / 状态 |",
        "| :---: | :--- | :--- | :---: | :--- | :---: | :--- |",
    ])

    for r in case_results:
        status_tag = "✅ PASS" if r["is_pass"] else "❌ FAIL"
        tools_str = ", ".join(r["actual_tools"]) if r["actual_tools"] else "*(直接回复)*"
        clean_q = r["query"].replace("|", "\\|")
        note = "符合预期" if r["is_pass"] else "; ".join(r["reasons"])
        md.append(
            f"| #{r['id']:03d} | {r['category']} | {clean_q} | {status_tag} | `{tools_str}` | {r['duration_ms']} ms | {note} |"
        )

    md.append("")
    output_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\n[报告] Markdown 评测报告已成功保存至：{output_path.resolve()}")


# ---------------------------------------------------------------------------
# 主执行入口
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Smart CS Agent 自动化端到端业务评测套件")
    parser.add_argument("--workers", type=int, default=4, help="并发执行线程数，默认为 4")
    parser.add_argument("--limit", type=int, default=None, help="仅运行前 N 条用例（快速冒烟）")
    parser.add_argument("--category", type=str, default=None, help="按模块文件名或分类名过滤（如 01、rag、routing 等）")
    parser.add_argument("--base-url", type=str, default=_DEFAULT_BASE_URL, help="FastAPI 服务根地址")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    chat_api = f"{base_url}/api/chat"
    health_api = f"{base_url}/health"

    print("=" * 76)
    print("      Smart CS Agent 企业智能售后与工单分流 Agent 自动化评测系统      ")
    print("=" * 76)
    print(f"[配置] 服务端点: {base_url} | 并发线程: {args.workers} | 过滤条件: {args.category or '全量'}")

    # 1. 加载用例
    test_cases = load_test_cases(category_filter=args.category, limit=args.limit)
    if not test_cases:
        print("[错误] 未找到任何匹配的评测用例，请检查 eval/cases/ 目录或参数过滤设置！", file=sys.stderr)
        sys.exit(1)
    print(f"[就绪] 成功聚合加载 {len(test_cases)} 条评测用例。\n")

    # 2. 探测服务存活
    print(f"[探测] 正在连接服务健康端点 {health_api} ...")
    if not _check_service_alive(health_api):
        print(
            f"[错误] 无法连接到服务 {health_api}！\n"
            "请确认已在终端启动 FastAPI 后端服务：\n"
            "  .venv\\Scripts\\python.exe -m uvicorn app.main:app --reload\n",
            file=sys.stderr,
        )
        sys.exit(1)
    print("[就绪] 后端服务健康在线，开始执行评测。\n")

    # 3. 记录初始工单快照
    initial_ticket_ids = _get_current_ticket_ids()
    print(f"[快照] 数据库现有工单基数: {len(initial_ticket_ids)} 条（评测完成后将自动恢复并零污染清理）\n")

    # 4. 多线程并发执行
    print_lock = threading.Lock()
    completed_counter = 0
    total_cases = len(test_cases)
    raw_results: List[Dict[str, Any]] = []

    header = f"{'进度':<10} | {'ID':<5} | {'场景分类':<14} | {'状态':<6} | {'耗时':<8} | {'实际工具调用':<28}"
    print(header)
    print("-" * 88)

    def worker_task(case_item: Dict[str, Any]) -> Dict[str, Any]:
        nonlocal completed_counter
        cid = case_item.get("id", 0)
        cat = case_item.get("category", "")
        mod = case_item.get("_module", cat)
        query = case_item.get("query", "")

        is_pass, reasons, res, duration_ms = _evaluate_case(case_item, chat_api)
        actual_tools = res.get("tools_called", [])
        actual_citations = res.get("citations", [])
        reply = res.get("reply", "")

        with print_lock:
            completed_counter += 1
            status_text = "PASS" if is_pass else "FAIL"
            tools_str = ",".join(actual_tools) if actual_tools else "(direct)"
            progress_tag = f"[{completed_counter}/{total_cases}]"
            print(f"{progress_tag:<10} | #{cid:03d}  | {cat:<12} | {status_text:<6} | {duration_ms:>5}ms | {tools_str:<28}")
            if not is_pass:
                for r in reasons:
                    print(f"             ↳ [原因] {r}")
                if actual_citations:
                    print(f"             ↳ [实际引用] {actual_citations}")

        return {
            "id": cid,
            "category": cat,
            "module": mod,
            "query": query,
            "is_pass": is_pass,
            "reasons": reasons,
            "actual_tools": actual_tools,
            "actual_citations": actual_citations,
            "duration_ms": duration_ms,
            "reply": reply,
            "required_tools": case_item.get("required_tools", []),
            "forbidden_tools": case_item.get("forbidden_tools", []),
        }

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(worker_task, case) for case in test_cases]
            for fut in as_completed(futures):
                try:
                    res_item = fut.result()
                    raw_results.append(res_item)
                except Exception as ex:
                    print(f"[严重错误] 评测线程内部异常: {ex}", file=sys.stderr)

    finally:
        # 数据无污染清理保证
        print("\n" + "-" * 76)
        _cleanup_test_artifacts(initial_ticket_ids)

    # 5. 按用例 ID 重新排序聚合
    case_results = sorted(raw_results, key=lambda x: x["id"])

    # 6. 指标统计与分类汇总
    total_count = len(case_results)
    passed_count = sum(1 for c in case_results if c["is_pass"])
    pass_rate = (passed_count / total_count * 100) if total_count > 0 else 0.0

    tool_correct_count = 0
    safety_correct_count = 0

    for c in case_results:
        req_tools = set(c.get("required_tools", []))
        forbid_tools = set(c.get("forbidden_tools", []))
        actual = set(c.get("actual_tools", []))
        if req_tools.issubset(actual) and forbid_tools.isdisjoint(actual):
            tool_correct_count += 1

        reply = c.get("reply", "")
        # 安全断言：无技术泄漏 (.md) 且超纲/假单号/注入场景未违规
        safe_ok = (".md" not in reply) and (
            not (c["id"] in range(81, 101) and not c["is_pass"])
        )
        if safe_ok:
            safety_correct_count += 1

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
        "workers": args.workers,
    }

    # 终端打印总体统计与 5 大模块汇总
    print("\n" + "=" * 76)
    print("                           评测指标总览                           ")
    print("=" * 76)
    print(f"评测用例总数:          {total_count}")
    print(f"通过用例总数:          {passed_count} / {total_count}")
    print(f"综合通过率 (Pass Rate): {pass_rate:.1f}%")
    print(f"工具路由准确率:        {tool_accuracy:.1f}%")
    print(f"防幻觉与对客安全率:    {safety_rate:.1f}%")
    print(f"平均响应耗时:          {avg_latency} ms")
    print(f"P95 响应耗时:          {p95_latency} ms")
    print("-" * 76)

    print("5 大业务场景模块分类通过率：")
    module_groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in case_results:
        m = r.get("module", r.get("category", "默认分类"))
        module_groups.setdefault(m, []).append(r)

    for m_name in sorted(module_groups.keys()):
        m_items = module_groups[m_name]
        m_tot = len(m_items)
        m_pas = sum(1 for x in m_items if x["is_pass"])
        m_pr = (m_pas / m_tot * 100) if m_tot > 0 else 0.0
        m_avg = int(sum(x["duration_ms"] for x in m_items) / m_tot) if m_tot > 0 else 0
        print(f"  - {m_name:<20}: {m_pas:>2}/{m_tot:<2} ({m_pr:>5.1f}%) | 平均耗时: {m_avg}ms")
    print("=" * 76)

    # 7. 写入 Markdown 评测报告
    _generate_markdown_report(summary, case_results, _REPORT_PATH, base_url)

    # 8. 全量 100 题评测时同步刷新 eval/eval_summary.json，确保线上 #eval 页面指标严格同源
    if args.category is None and args.limit is None and total_count >= 100:
        _sync_e2e_summary_json(summary)


def _sync_e2e_summary_json(summary: Dict[str, Any]) -> None:
    """将最新全量端到端评测指标写回 eval/eval_summary.json（保留 RAG 消融与数据集元数据）。"""
    if not _SUMMARY_JSON_PATH.exists():
        return
    try:
        data = json.loads(_SUMMARY_JSON_PATH.read_text(encoding="utf-8"))
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data["e2e_summary"] = {
            "evaluated_at": now_str,
            "total_cases": summary["total"],
            "pass_rate": round(summary["pass_rate"], 1),
            "tool_accuracy": round(summary["tool_accuracy"], 1),
            "safety_rate": round(summary["safety_rate"], 1),
            "avg_latency_ms": summary["avg_latency"],
            "p95_latency_ms": summary["p95_latency"],
        }
        for kpi in data.get("kpis", []):
            if kpi.get("key") == "e2e_tool_accuracy":
                kpi["value"] = f"{summary['tool_accuracy']:.1f}%"
                kpi["detail"] = (
                    f"{summary['total']} 题并发通过率 {summary['pass_rate']:.1f}% · "
                    f"防幻觉安全率 {summary['safety_rate']:.1f}%"
                )
        _SUMMARY_JSON_PATH.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[同步] 已刷新 { _SUMMARY_JSON_PATH.name } 中的端到端评测汇总指标。")
    except Exception as e:
        print(f"[警告] 同步 eval_summary.json 失败: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
