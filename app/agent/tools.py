"""
app/agent/tools.py

供大模型 Function Calling 使用的底层工具函数模块。
包含：
  - 4 个业务工具函数（query_order / create_after_sales_ticket /
    query_tickets / search_knowledge_base）
  - OpenAI Function Calling 标准格式的 TOOLS_SCHEMA
  - 安全工具分发器 execute_tool（捕获所有边界异常，永不直接崩溃）
"""

import json
import logging
import secrets
from datetime import datetime
from typing import Any

from app.db.database import get_db_connection
# 知识库检索器（模块级单例，首次调用时初始化）
from app.agent.knowledge import get_retriever

logger = logging.getLogger(__name__)

# issue_type 合法枚举值，工具层再次校验（双保险：Schema 层 + 代码层）
VALID_ISSUE_TYPES = {"退货", "换货", "物流催单", "售后咨询", "转人工"}


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def query_order(order_id: str) -> str:
    """
    根据订单号查询订单详情。
    返回 JSON 字符串；订单不存在时返回带 error 字段的 JSON，
    让大模型能直接读懂并告知用户，而不是抛出异常中断流程。
    """
    sql = """
    SELECT order_id, user_name, product_name, amount, status, tracking_no, created_at
    FROM orders
    WHERE order_id = ?;
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (order_id.strip(),))
            row = cursor.fetchone()
    except Exception as e:
        # 数据库层异常：记录日志后返回可读错误，不向上抛出
        logger.error("query_order 数据库异常 order_id=%s: %s", order_id, e)
        return json.dumps({"error": f"数据库查询失败：{str(e)}"}, ensure_ascii=False)

    if not row:
        logger.warning("query_order: 订单 %s 不存在", order_id)
        return json.dumps({"error": f"订单号 {order_id} 不存在，请核实后重试"}, ensure_ascii=False)

    return json.dumps(dict(row), ensure_ascii=False)


def create_after_sales_ticket(
    order_id: str,
    issue_type: str,
    reason: str,
    summary: str,
) -> str:
    """
    创建售后工单。
    流程：
      1. 代码层校验 issue_type 枚举合法性（防止大模型传入非法值）
      2. 校验订单是否存在（不存在则返回错误，不写库）
      3. 生成 TKT{YYYYMMDD}-{6位大写随机十六进制} 格式工单号
      4. INSERT tickets 表，默认状态"待处理"
      5. 返回新工单完整信息 JSON
    """
    # 1. 枚举合法性校验
    if issue_type not in VALID_ISSUE_TYPES:
        return json.dumps(
            {"error": f"issue_type '{issue_type}' 非法，必须是以下之一：{sorted(VALID_ISSUE_TYPES)}"},
            ensure_ascii=False,
        )

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # 2. 校验订单存在
            cursor.execute("SELECT order_id FROM orders WHERE order_id = ?;", (order_id.strip(),))
            if not cursor.fetchone():
                logger.warning("create_after_sales_ticket: 订单 %s 不存在，拒绝建单", order_id)
                return json.dumps(
                    {"error": f"订单号 {order_id} 不存在，无法创建售后工单"},
                    ensure_ascii=False,
                )

            # 3. 生成唯一工单号：TKT{YYYYMMDD}-{6位大写十六进制}
            date_str = datetime.now().strftime("%Y%m%d")
            # secrets.token_hex(3) 生成 3 字节 = 6 个十六进制字符，安全随机
            hex_suffix = secrets.token_hex(3).upper()
            ticket_id = f"TKT{date_str}-{hex_suffix}"

            created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 4. 写入 tickets 表
            insert_sql = """
            INSERT INTO tickets (ticket_id, order_id, issue_type, reason, status, summary, created_at)
            VALUES (?, ?, ?, ?, '待处理', ?, ?);
            """
            cursor.execute(
                insert_sql,
                (ticket_id, order_id.strip(), issue_type, reason, summary, created_at),
            )
            conn.commit()

    except Exception as e:
        logger.error(
            "create_after_sales_ticket 数据库异常 order_id=%s issue_type=%s: %s",
            order_id, issue_type, e,
        )
        return json.dumps({"error": f"创建工单失败：{str(e)}"}, ensure_ascii=False)

    logger.info("工单创建成功 ticket_id=%s order_id=%s issue_type=%s", ticket_id, order_id, issue_type)
    return json.dumps(
        {
            "success": True,
            "ticket_id": ticket_id,
            "order_id": order_id,
            "issue_type": issue_type,
            "reason": reason,
            "summary": summary,
            "status": "待处理",
            "created_at": created_at,
        },
        ensure_ascii=False,
    )


def query_tickets(order_id: str) -> str:
    """
    查询某订单下所有售后工单列表。
    返回工单数组 JSON；无工单时返回空列表，不报错。
    """
    sql = """
    SELECT ticket_id, order_id, issue_type, reason, status, summary, created_at
    FROM tickets
    WHERE order_id = ?
    ORDER BY created_at DESC;
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (order_id.strip(),))
            rows = cursor.fetchall()
    except Exception as e:
        logger.error("query_tickets 数据库异常 order_id=%s: %s", order_id, e)
        return json.dumps({"error": f"查询工单失败：{str(e)}"}, ensure_ascii=False)

    tickets = [dict(row) for row in rows]
    return json.dumps({"order_id": order_id, "tickets": tickets, "count": len(tickets)}, ensure_ascii=False)


def search_knowledge_base(query: str) -> str:
    """
    检索本地售后规则知识库，返回命中条款的 JSON 列表。
    每条包含：source（来源文件）、section（章节）、content（条款原文）、score（相关性分）。
    若无命中返回空列表 JSON，不报错，让大模型诚实告知用户知识库未覆盖该问题。

    边界处理：
      - 检索器未初始化（知识库目录为空）→ 返回空列表 JSON
      - query 为空或全空白 → 返回空列表 JSON（短路，避免无意义检索）
      - 检索过程异常 → 记录日志，返回 error JSON（大模型兜底处理）
    """
    if not query or not query.strip():
        return json.dumps([], ensure_ascii=False)

    try:
        retriever = get_retriever()
        # top_k=3：每次最多返回 3 条最相关条款，避免 context 过长
        results = retriever.search(query.strip(), top_k=3)
        logger.info(
            "search_knowledge_base: query='%s' 命中 %d 条",
            query[:50], len(results),
        )
        return json.dumps(results, ensure_ascii=False)
    except Exception as e:
        # 检索异常不应中断 Agent 对话，记录日志后返回可读错误
        logger.error("search_knowledge_base 检索异常 query='%s': %s", query[:50], e)
        return json.dumps({"error": f"知识库检索失败：{str(e)}"}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# OpenAI Function Calling 标准 TOOLS_SCHEMA
# ---------------------------------------------------------------------------

TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "query_order",
            "description": (
                "根据订单号查询订单详情，包括商品名称、金额、物流状态、运单号等信息。"
                "当用户询问订单状态、物流进度、商品信息时调用此工具。"
                "调用前必须确认用户已提供具体订单号，不可编造。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "订单唯一编号，例如 ORD20261101001，由用户提供",
                    }
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_after_sales_ticket",
            "description": (
                "为已存在的订单创建售后工单。新建工单初始状态为'待处理'，等待售后专员跟进处理。"
                "仅当用户明确提出退货、换货、物流催单、售后咨询或转人工诉求，"
                "且已提供订单号和具体原因时才调用此工具，不可提前或主动调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "关联的订单编号，由用户提供",
                    },
                    "issue_type": {
                        "type": "string",
                        "description": "工单类型，必须严格从以下 5 个选项中选择",
                        "enum": ["退货", "换货", "物流催单", "售后咨询", "转人工"],
                    },
                    "reason": {
                        "type": "string",
                        "description": "用户申请原因的详细描述，如：商品破损、长时间未收到货等",
                    },
                    "summary": {
                        "type": "string",
                        "description": "对本次售后诉求的简短概述，不超过 50 字",
                    },
                },
                "required": ["order_id", "issue_type", "reason", "summary"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_tickets",
            "description": (
                "查询某订单下已提交的所有售后工单列表及其处理状态。"
                "当用户询问'我的工单进展如何'、'之前提交的申请有没有处理'等问题时调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "要查询工单的订单编号",
                    }
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "检索本店本地售后规则、店铺政策与故障排查知识库。只要涉及核实任何店铺政策、售后规则、"
                "退换货条件（如拆封退货/运费承担/15天换新）、保修质保政策（含配件质保期）、发货与物流时效、"
                "退款到账、发票开具与7天价保政策、优惠券退款分摊与定制刻字商品售后规则，"
                "以及蓝牙耳机（单耳无声/断连/充电异常）、机械键盘（连击/失灵/固件重置）等硬件常见故障自助排查步骤与指导时，"
                "都必须调用此工具进行查询。严禁凭借预训练知识自行臆测或回答具体的政策条款与排查步骤。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "用户的政策咨询问题，尽量保留关键词，如"
                            "'耳机拆封后能否退货'、'退款多久到账'、'质保期多长'等"
                        ),
                    }
                },
                "required": ["query"],
            },
        },
    },
]

# 工具名 → 函数 映射表，用于分发器路由
_TOOL_REGISTRY: dict[str, Any] = {
    "query_order": query_order,
    "create_after_sales_ticket": create_after_sales_ticket,
    "query_tickets": query_tickets,
    "search_knowledge_base": search_knowledge_base,
}


# ---------------------------------------------------------------------------
# 安全工具分发器
# ---------------------------------------------------------------------------

def execute_tool(tool_name: str, arguments: str | dict) -> str:
    """
    安全分发大模型请求的工具调用。

    参数：
      tool_name  - 大模型返回的工具名称字符串
      arguments  - 工具参数，可能是 JSON 字符串或已解析的 dict

    边界处理：
      - 未知工具名 → 返回友好错误 JSON
      - JSON 解析失败 → 返回错误 JSON（不崩溃）
      - 参数缺失（TypeError/KeyError）→ 返回错误 JSON
      - 工具函数内部异常 → 由各工具函数自行捕获并返回错误 JSON

    严禁在此函数内直接 raise，确保大模型循环不因工具错误中断。
    """
    # 1. 检查工具名合法性
    if tool_name not in _TOOL_REGISTRY:
        logger.warning("execute_tool: 未知工具名 '%s'", tool_name)
        return json.dumps(
            {"error": f"未知工具 '{tool_name}'，可用工具：{list(_TOOL_REGISTRY.keys())}"},
            ensure_ascii=False,
        )

    # 2. 解析 arguments（大模型传来的可能是 JSON 字符串或 dict）
    if isinstance(arguments, str):
        try:
            kwargs = json.loads(arguments)
        except json.JSONDecodeError as e:
            logger.error("execute_tool: arguments JSON 解析失败 tool=%s err=%s", tool_name, e)
            return json.dumps(
                {"error": f"工具参数 JSON 格式错误：{str(e)}"},
                ensure_ascii=False,
            )
    else:
        kwargs = arguments  # 已是 dict，直接使用

    # 3. 调用对应工具函数，捕获缺参或类型错误
    try:
        result: str = _TOOL_REGISTRY[tool_name](**kwargs)
        return result
    except TypeError as e:
        # 通常是缺少必填参数或参数名错误
        logger.error("execute_tool: 参数错误 tool=%s kwargs=%s err=%s", tool_name, kwargs, e)
        return json.dumps(
            {"error": f"工具 '{tool_name}' 参数错误：{str(e)}"},
            ensure_ascii=False,
        )
    except Exception as e:
        logger.error("execute_tool: 未预期异常 tool=%s err=%s", tool_name, e)
        return json.dumps(
            {"error": f"工具 '{tool_name}' 执行异常：{str(e)}"},
            ensure_ascii=False,
        )
