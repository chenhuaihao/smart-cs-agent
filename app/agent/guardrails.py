"""
app/agent/guardrails.py

v2.0 P1 护栏与证据账本模块：
  - EvidenceLedger（Pydantic v2）：记录本轮实际通过工具查到的 orders / citations / tickets 真实证据；
  - verify_reply_against_ledger：代码级硬校验（防捏造未查订单状态、防假称已创建工单）；
  - build_fallback_reply：校验超限、步数超限、超时或异常时的结构化兜底文案（含明确原因并引导转人工）；
  - 运行态预算配置读取（MAX_STEPS / STEP_TIMEOUT_SEC / TURN_TIMEOUT_SEC / MAX_VERIFY_RETRIES）。
"""

import json
import os
import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 1. 运行态预算与超时配置（支持 .env 动态覆盖）
# ---------------------------------------------------------------------------

def get_max_steps() -> int:
    """单轮最大推理/工具步数上限（默认 MAX_STEPS=5）。"""
    val = os.getenv("MAX_STEPS", "5")
    try:
        parsed = int(val.strip())
        return parsed if parsed > 0 else 5
    except ValueError:
        return 5


def get_step_timeout_sec() -> float:
    """单步 LLM / 工具调用超时秒数（默认 STEP_TIMEOUT_SEC=20）。"""
    val = os.getenv("STEP_TIMEOUT_SEC", "20")
    try:
        parsed = float(val.strip())
        return parsed if parsed > 0 else 20.0
    except ValueError:
        return 20.0


def get_turn_timeout_sec() -> float:
    """整轮对话总超时秒数（默认 TURN_TIMEOUT_SEC=45）。"""
    val = os.getenv("TURN_TIMEOUT_SEC", "45")
    try:
        parsed = float(val.strip())
        return parsed if parsed > 0 else 45.0
    except ValueError:
        return 45.0


# 规则校验失败时的最大自动重写次数（retry_count <= 2）
MAX_VERIFY_RETRIES = 2

# 高风险写操作工具集合（流式模式下触发 HITL 人工确认）
HIGH_RISK_TOOLS = {"create_after_sales_ticket"}


# ---------------------------------------------------------------------------
# 2. 证据账本（EvidenceLedger）Pydantic 模型
# ---------------------------------------------------------------------------

class OrderEvidence(BaseModel):
    """单笔已核验订单的关键证据"""
    order_id: str = Field(..., description="已通过工具查询的订单号")
    status: str = Field(default="", description="订单状态（已签收/运输中/待发货/已取消，或 not_found）")
    delivered_at: Optional[str] = Field(default=None, description="签收/下单时间（对应订单 created_at）")
    product_name: str = Field(default="", description="商品名称")
    tracking_no: Optional[str] = Field(default=None, description="物流运单号")
    exists: bool = Field(default=True, description="数据库内是否真实存在该订单")


class TicketEvidence(BaseModel):
    """单笔已创建或已查出的售后工单证据"""
    ticket_id: str = Field(..., description="工单号，如 TKT20261123-ABCDEF")
    order_id: str = Field(default="", description="关联订单号")
    issue_type: str = Field(default="", description="工单类型")
    status: str = Field(default="待处理", description="工单状态")


class CitationDetail(BaseModel):
    """单条知识库引用详情（P2 新增，供前端展示关键词/语义/双路命中标签）"""
    source: str = Field(..., description="条款来源，格式 '文件名#章节名'")
    retrieval_source: Literal["bm25", "vector", "hybrid"] = Field(
        default="bm25",
        description="命中来源：bm25（仅关键词）/ vector（仅语义）/ hybrid（双路融合）",
    )
    score: float = Field(default=0.0, description="相关性得分（BM25 分或 RRF 融合分）")


class EvidenceLedger(BaseModel):
    """
    本轮对话的证据账本：
    仅由 tool_node 根据工具真实返回结构化写入，作为 policy_verifier 校验的唯一事实来源。
    """
    orders: Dict[str, OrderEvidence] = Field(
        default_factory=dict,
        description="已查询的订单字典：order_id -> OrderEvidence",
    )
    citations: List[str] = Field(
        default_factory=list,
        description="命中的知识库条款来源列表，格式 '文件名#章节名'",
    )
    citation_details: List[CitationDetail] = Field(
        default_factory=list,
        description="命中的知识库条款详情（含 source / retrieval_source / score）",
    )
    tickets: Dict[str, TicketEvidence] = Field(
        default_factory=dict,
        description="本轮已创建或查询到的工单字典：ticket_id -> TicketEvidence",
    )


def merge_ledger(left: EvidenceLedger | dict | None, right: EvidenceLedger | dict | None) -> EvidenceLedger:
    """LangGraph State reducer：合并两份 EvidenceLedger（保序去重）。"""
    base = left if isinstance(left, EvidenceLedger) else EvidenceLedger.model_validate(left or {})
    inc = right if isinstance(right, EvidenceLedger) else EvidenceLedger.model_validate(right or {})

    merged_orders = dict(base.orders)
    merged_orders.update(inc.orders)

    merged_citations = list(base.citations)
    for c in inc.citations:
        if c not in merged_citations:
            merged_citations.append(c)

    merged_details = list(base.citation_details)
    seen_detail_sources = {d.source for d in merged_details}
    for d in inc.citation_details:
        if d.source not in seen_detail_sources:
            merged_details.append(d)
            seen_detail_sources.add(d.source)

    merged_tickets = dict(base.tickets)
    merged_tickets.update(inc.tickets)

    return EvidenceLedger(
        orders=merged_orders,
        citations=merged_citations,
        citation_details=merged_details,
        tickets=merged_tickets,
    )


def record_tool_evidence(
    ledger: EvidenceLedger,
    tool_name: str,
    arguments: Any,
    tool_result: str,
    new_citations: list[str],
    new_citation_details: list[dict[str, Any] | CitationDetail] | None = None,
) -> EvidenceLedger:
    """
    根据工具执行结果更新证据账本：
      - query_order：记录订单号及其真实 status / delivered_at(created_at) / product_name；
        若返回 error（如订单不存在），同样标记该 order_id 已核查（exists=False），防止误判为"未查单"。
      - create_after_sales_ticket：当 success=True 时记录新建工单号与状态。
      - query_tickets：记录该订单下已有工单列表。
      - search_knowledge_base：合并命中的 citations 与 citation_details。
    """
    parsed_details = [
        d if isinstance(d, CitationDetail) else CitationDetail.model_validate(d)
        for d in (new_citation_details or [])
    ]
    updated = merge_ledger(
        ledger,
        EvidenceLedger(citations=new_citations, citation_details=parsed_details),
    )
    args_dict = arguments if isinstance(arguments, dict) else _parse_json_dict(arguments)
    res_data = _parse_json_dict(tool_result)

    if tool_name == "query_order":
        req_oid = str(args_dict.get("order_id", "")).strip().upper()
        if "order_id" in res_data:
            oid = str(res_data["order_id"]).strip().upper()
            updated.orders[oid] = OrderEvidence(
                order_id=oid,
                status=str(res_data.get("status", "")),
                delivered_at=res_data.get("created_at"),
                product_name=str(res_data.get("product_name", "")),
                tracking_no=res_data.get("tracking_no"),
                exists=True,
            )
        elif req_oid:
            # 查过了但数据库不存在该单：记录已查过（exists=False），模型如实告知"不存在"时放行
            updated.orders[req_oid] = OrderEvidence(
                order_id=req_oid,
                status="not_found",
                exists=False,
            )

    elif tool_name == "create_after_sales_ticket":
        if res_data.get("success") and res_data.get("ticket_id"):
            tid = str(res_data["ticket_id"]).strip().upper()
            updated.tickets[tid] = TicketEvidence(
                ticket_id=tid,
                order_id=str(res_data.get("order_id", "")).strip().upper(),
                issue_type=str(res_data.get("issue_type", "")),
                status=str(res_data.get("status", "待处理")),
            )

    elif tool_name == "query_tickets":
        for t in res_data.get("tickets") or []:
            if isinstance(t, dict) and t.get("ticket_id"):
                tid = str(t["ticket_id"]).strip().upper()
                updated.tickets[tid] = TicketEvidence(
                    ticket_id=tid,
                    order_id=str(t.get("order_id", "")).strip().upper(),
                    issue_type=str(t.get("issue_type", "")),
                    status=str(t.get("status", "")),
                )

    return updated


def _parse_json_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


# ---------------------------------------------------------------------------
# 3. 规则校验器（policy_verifier 核心判断）与降级文案
# ---------------------------------------------------------------------------

# 匹配订单号（如 ORD20261101、ORD20261121A、ORD2026_FAKE_NO）与工单号（TKT20261123-ABCDEF）
_ORDER_ID_RE = re.compile(r"\bORD[0-9A-Z_]{4,20}\b", re.IGNORECASE)
_TICKET_ID_RE = re.compile(r"\bTKT\d{8}-[0-9A-F]{6}\b", re.IGNORECASE)
_TRACKING_NO_RE = re.compile(r"\b(?:SF|JD|YTO|ZT)\d{8,14}\b", re.IGNORECASE)

# 断言具体订单物流/签收状态的关键词（不含通用政策说明如"签收后7天内"）
_ORDER_STATUS_CLAIMS = (
    "已签收", "已经签收", "已发货", "已经发货", "运输中", "派送中", "在途",
    "待发货", "尚未发货", "还未发货", "已取消", "订单已关闭", "运单号", "快递单号",
)

# 声称已成功创建/提交工单的明确话术
_TICKET_CREATION_CLAIMS = (
    "已为您创建", "已成功创建", "工单创建成功", "已为您提交", "已成功提交",
    "已帮您创建", "已帮您提交", "已为您建立", "已帮您建好", "已经为您创建",
    "已经为您提交", "创建了售后工单", "创建了退货工单", "创建了换货工单",
)

# 否定前缀：若句中是"无法/不能/未为您创建"，不应误判为谎称已建单
_NEGATION_MARKERS = ("无法", "不能", "未", "没有", "没能", "暂不", "不可", "拒绝", "需要您确认")


def verify_reply_against_ledger(reply: str, ledger: EvidenceLedger) -> list[str]:
    """
    对 agent_node 产出的最终回复做代码级硬校验。
    返回违规原因列表；空列表表示校验通过。

    规则 1（防订单/物流幻觉）：
      a) 回复提及了具体订单号（ORD...），但 evidence_ledger.orders 中未查过该订单；
      b) 回复在未查任何真实订单（或所查订单不存在）时，捏造具体运单号或断言"您的订单已签收/已发货/运输中"。
    规则 2（防虚假建单幻觉）：
      a) 回复声称"已成功创建工单/已为您提交退货工单"等，但 evidence_ledger.tickets 为空；
      b) 回复给出了具体工单号（TKT...），但该工单号不在 evidence_ledger.tickets 中。
    """
    reasons: list[str] = []
    if not reply or not reply.strip():
        return reasons

    # --- 规则 1a：提及了具体订单号，但账本未查询过 ---
    mentioned_orders = {m.upper() for m in _ORDER_ID_RE.findall(reply)}
    unchecked_orders = [oid for oid in sorted(mentioned_orders) if oid not in ledger.orders]
    if unchecked_orders:
        reasons.append(
            f"回复中提及了订单号 {', '.join(unchecked_orders)}，但未调用 query_order 工具核实真实订单数据。"
        )

    # --- 规则 1b：未查过任何有效订单，却捏造具体运单号或断言当前订单物流/签收状态 ---
    valid_orders = [o for o in ledger.orders.values() if o.exists]
    mentioned_trackings = {m.upper() for m in _TRACKING_NO_RE.findall(reply)}
    known_trackings = {
        (o.tracking_no or "").upper() for o in valid_orders if o.tracking_no
    }
    unverified_trackings = [t for t in sorted(mentioned_trackings) if t not in known_trackings]
    if unverified_trackings:
        reasons.append(
            f"回复中给出了运单号 {', '.join(unverified_trackings)}，但证据账本中无此运单记录。"
        )

    if not valid_orders and _claims_concrete_order_status(reply):
        reasons.append(
            "证据账本中没有任何已核实的订单记录，但回复直接断言了订单的物流或签收状态。"
        )

    # --- 规则 2a/2b：声称已创建工单或给出工单号，但账本无对应工单 ---
    mentioned_tickets = {m.upper() for m in _TICKET_ID_RE.findall(reply)}
    unknown_tickets = [tid for tid in sorted(mentioned_tickets) if tid not in ledger.tickets]
    if unknown_tickets:
        reasons.append(
            f"回复中提及了工单号 {', '.join(unknown_tickets)}，但证据账本中未创建或查询到该工单。"
        )

    if not ledger.tickets and _claims_ticket_created(reply):
        reasons.append(
            "回复声称已创建/已提交售后工单，但证据账本中没有任何已创建的工单记录（严禁未调用 create_after_sales_ticket 就谎称建单成功）。"
        )

    return reasons


def _claims_concrete_order_status(reply: str) -> bool:
    """
    判断回复是否在未查单情况下直接对用户的具体订单下物流/签收定论。
    注意区分：
      - 通用政策解答（如"自签收起7天内无理由退货"、"待发货状态下可取消"）→ 不拦截；
      - 针对具体订单断言（如"您的订单已签收"、"该订单目前运输中"）→ 拦截。
    """
    # 将常见政策条款语境抹除后再匹配具体订单状态断言
    cleaned = re.sub(r"签收[后之]?[起内]?\s*\d+\s*[天个小]", "", reply)
    cleaned = re.sub(r"待发货(?:状态|订单|时|期间|阶段)", "", cleaned)
    cleaned = re.sub(r"(?:已签收|已发货|运输中)(?:状态|的订单|商品|后)", "", cleaned)

    order_subject = ("您的订单", "该订单", "这笔订单", "此订单", "您的包裹", "您的商品目前", "查询到订单", "经核实订单")
    has_subject = any(s in cleaned for s in order_subject)
    has_status = any(k in cleaned for k in _ORDER_STATUS_CLAIMS)
    return has_subject and has_status


def _claims_ticket_created(reply: str) -> bool:
    """判断回复是否在肯定语境下声称已创建/已提交工单。"""
    for sentence in re.split(r"[。！？!\n；;]", reply):
        if any(claim in sentence for claim in _TICKET_CREATION_CLAIMS) and "工单" in sentence:
            if not any(neg in sentence for neg in _NEGATION_MARKERS):
                return True
    return False


def build_fallback_reply(reason: str) -> str:
    """
    统一构造降级兜底回复：
    明确说明降级原因（绝不胡编订单或工单结果），并引导用户转接人工客服处理。
    """
    return (
        f"非常抱歉，为了确保售后信息准确无误，当前请求触发了安全护栏（原因：{reason}），"
        "小智暂时无法直接给出结论。建议您核对订单号后重试，或联系人工客服专员为您优先跟进处理。"
    )
