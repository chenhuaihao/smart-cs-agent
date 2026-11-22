"""
app/agent/core.py

Agent 核心调度模块：实现多轮对话 + Function Calling 循环。

职责：
  - 从 chat_logs 加载最近 6 轮历史作为上下文
  - 调用 DeepSeek（OpenAI 兼容接口）进行推理
  - 根据 tool_calls 分发工具执行，回填 tool 角色消息，循环直到自然语言回复
  - 解析 search_knowledge_base 工具结果，提取保序去重的 citations 溯源引用
  - 统计耗时，将本轮对话持久化到 chat_logs 表
"""

import json
import logging
import os
import time
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI

from app.agent.tools import TOOLS_SCHEMA, execute_tool
from app.db.database import get_db_connection

# 在模块加载时加载 .env，确保环境变量就绪
load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 客户端初始化（模块级单例，避免每次请求重复创建）
# ---------------------------------------------------------------------------

_API_KEY = os.getenv("OPENAI_API_KEY", "")
_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "deepseek-chat")

if not _API_KEY:
    # 缺少 API Key 时立刻报错，避免在实际请求时才触发隐晦的 401
    raise EnvironmentError(
        "缺少环境变量 OPENAI_API_KEY！"
        "请在项目根目录的 .env 文件中配置真实的 API Key，参考 .env.example。"
    )

# 初始化 OpenAI 兼容客户端，timeout=30s 防止长时间挂起
_client = OpenAI(
    api_key=_API_KEY,
    base_url=_BASE_URL,
    timeout=30.0,
)

# ---------------------------------------------------------------------------
# System Prompt：严格中文客服角色约束（含 RAG 强制检索规则）
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """你是"小智"，一名专业、友善的企业智能售后客服助理。

【行为准则】
1. **订单核验与政策咨询边界**：
   - 只要用户在咨询中提及了具体订单号（无论其是否自述了签收日期、商品状态或时间差），**必须先调用 query_order 工具核实数据库真实订单与物流信息**，严禁仅凭用户单方自述直接给出售后结论。
   - **政策咨询优先检索**：当用户咨询政策/规则（如7天价保、退换货条件、发票开具、保修质保、运费承担等）但**未提供订单号**时，**必须先调用 search_knowledge_base 检索并准确解答通用规则**！解答完毕后，可根据需要礼貌提示“如需核查具体订单的处理情况，请提供您的订单号”。**绝不能因为用户未提供订单号就直接拒答或跳过 search_knowledge_base 工具检索！**（仅在用户明确要求查询特定订单物流、特定工单或发起建单却未给单号时，才直接追问订单号）。
2. **严禁捏造**：绝对不允许虚构不存在的订单信息、物流轨迹或工单记录。所有信息必须来源于工具查询结果。
3. **异常物流检索与处置**：当查询订单后发现存在**超时未发货、物流停滞无更新、或用户怀疑丢件破损**等异常情况时，**必须主动调用 search_knowledge_base 检索物流异常处理与赔付规则**，严格依据店铺规则给出处置建议或协助建单，严禁仅凭常识口头安抚。
4. **建单门槛**：只有在用户**明确提出退货/换货/物流催单/售后咨询/转人工诉求**，并且**订单号和具体原因均已齐全**时，才可调用 create_after_sales_ticket 工具创建工单。条件不满足时，继续向用户收集信息。
5. **结果解读**：收到工具返回的 JSON 后，将关键信息翻译为自然、易懂的中文回复给用户，不要直接暴露 JSON 原文。
6. **工具错误处理**：若工具返回包含 "error" 字段，向用户说明情况并引导其提供正确信息，不要重复调用相同参数。
7. **语气风格**：专业、简洁、有温度，使用礼貌敬语，避免过度啰嗦。
8. **知识库优先与严禁虚假检索声明**：
   - 凡是用户询问任何店铺规则、政策、服务权益、活动、分期/支付、换购、退换货条件、拆封退货、运费承担、质保期限（含各类配件质保）、保修流程、发货时效、退款到账时间、发票开具与7天价保、优惠券退款分摊、定制商品退换，以及遇到耳机（单耳无声/充电异常）或键盘等硬件故障排查时，**无论用户是否提供了订单号，都必须先实际调用 search_knowledge_base 工具检索，严禁凭借自身记忆直接作答，更严禁因未提供订单号而放弃知识库检索**。
   - 遇到设备硬件故障（如耳机不响、键盘按键失灵）时，**先耐心引导用户按知识库指引进行自助排查与重置操作**；仅在用户反馈排查重置依然无效、且订单号与原因齐全时，再按政策时效协助发起换新或维修工单。
   - 检索后，严格依据工具返回的条款原文作答。对客回复时，严禁暴露内部英文文件名或 `.md` 后缀（例如严禁输出 `return_policy.md`、`coupon_and_custom_rules.md` 等技术路径），应转化为自然友好的中文规则名称向用户说明（如"根据本店的定制商品售后规则"、"按照店铺7天无理由退货规定"）；精确的文件与章节溯源由系统后端自动记录，无需在聊天正文中展示。
   - **绝对红线**：严禁在**未实际调用 search_knowledge_base** 的情况下，在回复中擅自声称"我检索了知识库"、"知识库中暂未找到"或"经查询知识库无此政策"等虚假表述！只有在确实发起了工具调用且返回结果为空或无匹配时，才可如实告知"本店知识库暂未收录相关说明"，并引导用户转接人工客服处理。

【能力边界】
- 你只处理售后相关问题（查订单、查工单、退换货、物流催单、转人工、政策咨询）。
- 对于无关话题，礼貌说明你的职责范围并引导用户回到售后问题。
"""

# Function Calling 最大循环步数，防止模型异常导致死循环
_MAX_STEPS = 6  # 知识库检索可能多消耗一步，从 5 调整到 6

# 历史消息条数限制（最近 12 条 = 6 轮对话）
_HISTORY_LIMIT = 12


# ---------------------------------------------------------------------------
# 历史记录读取
# ---------------------------------------------------------------------------

def _load_history(session_id: str) -> list[dict]:
    """
    从 chat_logs 表读取指定 session 最近 _HISTORY_LIMIT 条对话记录，
    按 id DESC 取最新的，然后反转为时间正序，作为多轮上下文拼入 messages。

    只取 role 和 content 字段，tool_called / latency_ms 是内部统计，
    不需要传给大模型。
    """
    sql = """
    SELECT role, content
    FROM chat_logs
    WHERE session_id = ?
    ORDER BY id DESC
    LIMIT ?;
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (session_id, _HISTORY_LIMIT))
            rows = cursor.fetchall()
    except Exception as e:
        # 读取历史失败不影响主流程，降级为空历史继续处理
        logger.error("_load_history 数据库异常 session_id=%s: %s", session_id, e)
        return []

    # 查出是倒序的（最新在前），反转后得到时间正序的历史消息
    messages = [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]
    return messages


# ---------------------------------------------------------------------------
# 对话记录持久化
# ---------------------------------------------------------------------------

def _save_message(
    session_id: str,
    role: str,
    content: str,
    tool_called: str | None = None,
    latency_ms: int | None = None,
) -> None:
    """
    将单条消息写入 chat_logs 表。
    tool_called 仅在 assistant 消息时有值（逗号拼接的工具名列表）。
    latency_ms 仅在 assistant 消息时有值（端到端耗时，毫秒）。
    """
    sql = """
    INSERT INTO chat_logs (session_id, role, content, tool_called, latency_ms, created_at)
    VALUES (?, ?, ?, ?, ?, ?);
    """
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (session_id, role, content, tool_called, latency_ms, created_at))
            conn.commit()
    except Exception as e:
        # 持久化失败记录日志，不阻断接口响应（已生成的回复照常返回）
        logger.error(
            "_save_message 写入失败 session_id=%s role=%s: %s",
            session_id, role, e,
        )


# ---------------------------------------------------------------------------
# citations 提取辅助函数
# ---------------------------------------------------------------------------

def _extract_citations(tool_name: str, tool_result: str) -> list[str]:
    """
    从 search_knowledge_base 工具的返回 JSON 中提取溯源引用列表。
    格式：["文件名#章节名", ...]，保序去重（先出现的优先）。

    之所以在此直接解析结构化 JSON 而非依赖大模型在回复文本中输出：
    - 结构化解析可靠性 100%，不受大模型措辞影响
    - 不需要在 Prompt 中要求大模型格式化输出，减少 Prompt 复杂度
    """
    # 仅对知识库检索工具提取 citations，其他工具忽略
    if tool_name != "search_knowledge_base":
        return []

    try:
        data = json.loads(tool_result)
    except json.JSONDecodeError:
        # 工具返回格式异常（如包含 error 字段的非列表 JSON），直接跳过
        return []

    if not isinstance(data, list):
        # 返回 {"error": ...} 或其他非列表结构时，无可提取的 citations
        return []

    # 保序去重：用 seen set 判断重复，用 list 保序
    seen: set[str] = set()
    citations: list[str] = []
    for chunk in data:
        source = chunk.get("source", "")
        section = chunk.get("section", "")
        if source and section:
            key = f"{source}#{section}"
            if key not in seen:
                seen.add(key)
                citations.append(key)

    return citations


# ---------------------------------------------------------------------------
# 核心 Agent 入口
# ---------------------------------------------------------------------------

def run_agent(session_id: str, user_message: str) -> tuple[str, list[str], int, list[str]]:
    """
    执行一轮完整的 Agent 推理（含多步 Function Calling 循环）。

    流程：
      1. 读取历史消息组装 messages
      2. 进入循环（最多 _MAX_STEPS 步）：
         a. 调用大模型
         b. 若返回 tool_calls → 执行工具 → 回填 tool 角色消息 → 继续下一轮
            如果工具是 search_knowledge_base，同步解析返回结果提取 citations
         c. 若 finish_reason == "stop" → 输出最终回复，退出循环
      3. 计算端到端耗时
      4. 将用户输入和最终回复持久化到 chat_logs
      5. 返回 (reply, tools_called, latency_ms, citations)

    参数：
      session_id   - 会话唯一标识，用于读取/写入历史记录
      user_message - 用户本轮输入内容

    返回：
      reply        - 助手最终自然语言回复
      tools_called - 本轮实际调用的工具名列表（可能为空）
      latency_ms   - 端到端耗时（毫秒）
      citations    - 本轮命中知识库条款的去重来源列表（格式: "文件名#章节名"）
    """
    start_time = time.monotonic()

    # 1. 加载历史 + 构建 messages（动态注入当前系统日期 + system + history + 本轮用户输入）
    today_str = datetime.now().strftime("%Y-%m-%d")
    dynamic_system_prompt = (
        f"{_SYSTEM_PROMPT}\n"
        f"【系统运行环境】\n"
        f"- 当前系统日期：{today_str}（用于精准核算订单签收或购买距今经过的天数，准确判断是否在 7天无理由退货、15天质量换新或官方质保期限内）。"
    )

    history = _load_history(session_id)
    messages: list[dict] = [{"role": "system", "content": dynamic_system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    tools_called: list[str] = []  # 记录本轮调用的工具名
    citations: list[str] = []     # 记录知识库命中条款来源（保序去重）
    final_reply: str | None = None

    logger.info(
        "run_agent 开始 session_id=%s history_len=%d user_message='%s'",
        session_id, len(history), user_message[:50],
    )

    # 2. Function Calling 循环
    for step in range(_MAX_STEPS):
        logger.info("run_agent step=%d messages_count=%d", step, len(messages))

        try:
            response = _client.chat.completions.create(
                model=_MODEL_NAME,
                messages=messages,          # type: ignore[arg-type]
                tools=TOOLS_SCHEMA,         # type: ignore[arg-type]
                tool_choice="auto",         # 由模型自主决定是否调用工具
            )
        except Exception as e:
            # 大模型 API 调用失败（网络超时、鉴权失败等），立即返回友好提示
            logger.error("run_agent: 大模型 API 调用失败 step=%d: %s", step, e)
            final_reply = "抱歉，我暂时无法连接到后端服务，请稍后重试。如有紧急问题，可联系人工客服。"
            break

        assistant_message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        logger.info("run_agent step=%d finish_reason=%s", step, finish_reason)

        # 2a. 大模型请求调用工具（通过 message.tool_calls 判断，兼容 DeepSeek 格式）
        if assistant_message.tool_calls:
            # 将 assistant 的完整消息（含 tool_calls）回填到 messages
            messages.append(assistant_message)  # type: ignore[arg-type]

            # 逐个执行工具调用，每个调用结果以 tool 角色追加到 messages
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                arguments = tool_call.function.arguments  # JSON 字符串

                logger.info(
                    "run_agent: 调用工具 tool_name=%s call_id=%s",
                    tool_name, tool_call.id,
                )

                # 执行工具，execute_tool 内部保证不抛异常
                tool_result = execute_tool(tool_name, arguments)
                tools_called.append(tool_name)

                # 若工具为知识库检索，直接解析结构化结果提取 citations（不依赖大模型输出）
                # 保序去重：已在 _extract_citations 内部处理，这里做跨调用去重
                new_citations = _extract_citations(tool_name, tool_result)
                for c in new_citations:
                    if c not in citations:
                        citations.append(c)

                # 将工具返回结果以 tool 角色追加（DeepSeek 兼容 OpenAI tool 消息格式）
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                })

                logger.info("run_agent: 工具结果 tool_name=%s result='%s'", tool_name, tool_result[:100])

            # 工具执行完毕，继续循环让模型处理工具结果
            continue

        # 2b. 模型返回自然语言回复（finish_reason == "stop"）
        if finish_reason == "stop":
            final_reply = assistant_message.content or ""
            break

        # 2c. 其他意外 finish_reason（如 "length" 截断），取当前 content 兜底
        logger.warning("run_agent: 意外 finish_reason=%s，取当前 content 兜底", finish_reason)
        final_reply = assistant_message.content or ""
        break

    # 3. 达到最大步数仍未生成最终回复：友好兜底，避免返回 None
    if final_reply is None:
        logger.warning(
            "run_agent: 达到最大步数 %d 仍未生成最终回复 session_id=%s",
            _MAX_STEPS, session_id,
        )
        final_reply = (
            "非常抱歉，我在处理您的请求时遇到了一点复杂情况，未能及时给出回复。"
            "建议您稍后重试，或联系人工客服为您进一步处理。"
        )

    # 4. 计算端到端耗时（毫秒）
    latency_ms = int((time.monotonic() - start_time) * 1000)

    # 5. 持久化：先写用户输入，再写助手回复（含工具调用列表和耗时）
    _save_message(session_id, "user", user_message)
    _save_message(
        session_id,
        "assistant",
        final_reply,
        tool_called=",".join(tools_called) if tools_called else None,
        latency_ms=latency_ms,
    )

    logger.info(
        "run_agent 完成 session_id=%s tools_called=%s citations=%s latency_ms=%d",
        session_id, tools_called, citations, latency_ms,
    )

    return final_reply, tools_called, latency_ms, citations
