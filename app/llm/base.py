"""
app/llm/base.py

模型端口定义：Agent 编排层（LangGraph 节点）只依赖这里的抽象，不感知具体厂商 SDK。

为什么要抽这一层：
  - v1.0 直接在 core.py 里 new OpenAI 客户端，换模型 / 写单测都必须改业务代码；
  - 抽象后，切换厂商只需新增一个 Provider 子类并在 factory 注册；
  - 单元测试可注入 Mock Provider，不消耗真实 API 额度。
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Literal

from pydantic import BaseModel, Field


class ToolCallRequest(BaseModel):
    """模型请求调用的单个工具（已在 Provider 内部把流式碎片拼装完整）。"""

    id: str = Field(..., description="工具调用 ID，回填 tool 消息时必须原样带回")
    name: str = Field(..., description="工具名")
    arguments: str = Field(default="{}", description="JSON 字符串形式的参数（保持与 OpenAI 协议一致）")


class LLMStreamChunk(BaseModel):
    """
    统一流式输出结构，只有两种类型：
      - text_delta：文本增量，text 字段有值，可直接推给前端做打字机效果；
      - tool_calls：本次推理请求的完整工具调用列表（流结束前一次性给出）。

    之所以把工具调用"攒齐再吐"：OpenAI 协议下 arguments 是按字符碎片下发的，
    半截 JSON 对上层毫无意义，拼装细节属于厂商协议，应封装在适配器内部。
    """

    type: Literal["text_delta", "tool_calls"]
    text: str = ""
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)


class BaseLLMProvider(ABC):
    """大模型 Provider 抽象基类。"""

    #: Provider 名称，用于日志与排障
    name: str = "base"

    @abstractmethod
    def astream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        """
        以流式方式执行一次对话推理。

        参数：
          messages - OpenAI 格式的消息列表（system / user / assistant / tool）
          tools    - OpenAI Function Calling 格式的工具 Schema，None 表示不开放工具

        返回：
          异步迭代器，依次产出 LLMStreamChunk。
          实现方遇到网络 / 鉴权等不可恢复错误时应直接抛出异常，由编排层统一转成 error 事件。
        """
        raise NotImplementedError
