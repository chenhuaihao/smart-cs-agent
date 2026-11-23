"""
app/llm/deepseek.py

DeepSeek（及任意 OpenAI 兼容厂商）Provider 适配器，基于官方 openai.AsyncOpenAI。

环境变量与 v1.0 完全兼容（优先 DEEPSEEK_*，回退 OPENAI_*）：
  DEEPSEEK_API_KEY  / OPENAI_API_KEY
  DEEPSEEK_BASE_URL / OPENAI_BASE_URL
  DEEPSEEK_MODEL    / OPENAI_MODEL_NAME
"""

import asyncio
import logging
import os
import weakref
from typing import Any, AsyncIterator

from dotenv import load_dotenv
from openai import AsyncOpenAI

from app.llm.base import BaseLLMProvider, LLMStreamChunk, ToolCallRequest

load_dotenv()

logger = logging.getLogger(__name__)

# 单次 HTTP 请求超时（秒）：流式场景下作用于连接与每次读取，防止上游挂起拖死请求
_REQUEST_TIMEOUT = 30.0


class DeepSeekProvider(BaseLLMProvider):
    """OpenAI 兼容协议的流式 Provider。"""

    name = "deepseek"

    def __init__(self) -> None:
        self.api_key: str = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY", "")
        self.base_url: str = (
            os.getenv("DEEPSEEK_BASE_URL") or os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
        )
        self.model: str = os.getenv("DEEPSEEK_MODEL") or os.getenv("OPENAI_MODEL_NAME", "deepseek-chat")

        if not self.api_key:
            # 缺 Key 时在构造阶段立刻失败，避免请求时才收到隐晦的 401
            raise EnvironmentError(
                "缺少大模型 API Key 环境变量（DEEPSEEK_API_KEY 或 OPENAI_API_KEY）！"
                "请在项目根目录的 .env 文件中配置真实的 API Key，参考 .env.example。"
            )

        # 按事件循环缓存 AsyncOpenAI 客户端：
        # 底层 httpx 连接池绑定创建它的事件循环，同步兼容入口 run_agent 会在独立循环里运行，
        # 若跨循环复用同一客户端会触发 "Event loop is closed"。WeakKey 保证循环销毁后自动回收。
        self._clients: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, AsyncOpenAI]" = (
            weakref.WeakKeyDictionary()
        )

    def _get_client(self) -> AsyncOpenAI:
        """获取当前事件循环专属的 AsyncOpenAI 客户端（同一循环内复用连接池）。"""
        loop = asyncio.get_running_loop()
        client = self._clients.get(loop)
        if client is None:
            client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, timeout=_REQUEST_TIMEOUT)
            self._clients[loop] = client
        return client

    async def astream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        """
        流式调用 chat.completions：
          - content 增量 → 立即产出 text_delta；
          - tool_calls 增量 → 按 index 累积拼装，流结束后一次性产出 tool_calls。
        网络 / 鉴权异常不在此吞掉，直接上抛给编排层统一处理。
        """
        params: dict[str, Any] = {"model": self.model, "messages": messages, "stream": True}
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"  # 由模型自主决定是否调用工具

        stream = await self._get_client().chat.completions.create(**params)

        # index → {"id", "name", "arguments"}：OpenAI 协议中同一工具调用的碎片共享 index
        pending: dict[int, dict[str, str]] = {}
        finish_reason: str | None = None

        async for chunk in stream:
            if not chunk.choices:
                # 部分厂商最后会推一个只含 usage 的空 choices 包，直接跳过
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            if delta is not None:
                if delta.content:
                    yield LLMStreamChunk(type="text_delta", text=delta.content)
                for tc in delta.tool_calls or []:
                    slot = pending.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function is not None:
                        # name 一般只在首个碎片出现；arguments 需要逐段拼接
                        if tc.function.name:
                            slot["name"] = tc.function.name
                        if tc.function.arguments:
                            slot["arguments"] += tc.function.arguments
            if choice.finish_reason:
                finish_reason = choice.finish_reason

        if pending:
            calls = [
                ToolCallRequest(
                    id=slot["id"] or f"call_{idx}",  # 个别兼容厂商不下发 id，兜底生成保证 tool 消息可回填
                    name=slot["name"],
                    arguments=slot["arguments"] or "{}",
                )
                for idx, slot in sorted(pending.items())
            ]
            yield LLMStreamChunk(type="tool_calls", tool_calls=calls)

        logger.info(
            "DeepSeekProvider 流式推理结束 model=%s finish_reason=%s tool_calls=%d",
            self.model, finish_reason, len(pending),
        )
