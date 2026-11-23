"""
app/llm/factory.py

Provider 工厂：根据环境变量 LLM_PROVIDER（默认 deepseek）返回进程级单例。

扩展新厂商：实现 BaseLLMProvider 子类后在 _REGISTRY 注册即可，业务代码零改动。
通义 / 豆包等 OpenAI 兼容厂商无需新类，直接复用 deepseek 适配器并修改 BASE_URL / MODEL。
"""

import logging
import os
import threading
from typing import Callable

from app.llm.base import BaseLLMProvider
from app.llm.deepseek import DeepSeekProvider

logger = logging.getLogger(__name__)

# 名称 → 构造函数。键统一小写，读取环境变量时同样 lower() 后匹配
_REGISTRY: dict[str, Callable[[], BaseLLMProvider]] = {
    "deepseek": DeepSeekProvider,
}

_instance: BaseLLMProvider | None = None
# 并发首次请求时防止重复构造 Provider（构造里会读取环境变量、校验 Key）
_lock = threading.Lock()


def get_llm_provider() -> BaseLLMProvider:
    """返回当前进程的 Provider 单例（懒加载，首次调用时构造）。"""
    global _instance
    if _instance is not None:
        return _instance

    with _lock:
        # 双重检查：拿到锁后可能已被其他线程初始化
        if _instance is None:
            provider_name = os.getenv("LLM_PROVIDER", "deepseek").strip().lower()
            builder = _REGISTRY.get(provider_name)
            if builder is None:
                raise ValueError(
                    f"不支持的 LLM_PROVIDER='{provider_name}'，可选值：{sorted(_REGISTRY)}"
                )
            _instance = builder()
            logger.info("LLM Provider 初始化完成 provider=%s", provider_name)
    return _instance


def set_llm_provider(provider: BaseLLMProvider | None) -> None:
    """
    显式注入 / 重置 Provider 单例。
    主要用于单元测试注入 Mock Provider；传 None 表示清空，下次调用时按环境变量重新构造。
    """
    global _instance
    with _lock:
        _instance = provider
