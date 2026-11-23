"""
app/llm/embedding.py

在线文本向量化（Embedding）适配器：
  - 沿用端口-适配器模式定义 BaseEmbeddingProvider 抽象基类；
  - 实现 OpenAICompatibleEmbeddingProvider（复用官方 openai SDK，兼容硅基流动 BAAI/bge-m3、
    阿里云百炼 text-embedding-v3 及 OpenAI 官方 Embedding 接口）；
  - 严格遵守低内存红线：绝不引入 torch / sentence-transformers 等本地大模型依赖；
  - 优雅降级：未配置 EMBEDDING_API_KEY 时返回 None 并打 WARNING 日志，调用方自动降级为纯 BM25。
"""

from abc import ABC, abstractmethod
import logging
import os
import threading
from typing import Sequence

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = logging.getLogger(__name__)

# 默认指向硅基流动兼容端点与 BAAI/bge-m3（均可通过 .env 覆盖为百炼或 OpenAI）
DEFAULT_EMBEDDING_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_EMBEDDING_TIMEOUT_SEC = 15.0
# 阿里云百炼 text-embedding-v3 OpenAI 兼容接口单次最多 10 条，保守取 10 兼顾所有主流厂商
DEFAULT_EMBEDDING_BATCH_SIZE = 10

_PLACEHOLDER_KEYS = {
    "your_embedding_api_key_here",
    "your_openai_api_key_here",
    "your_api_key_here",
}


class BaseEmbeddingProvider(ABC):
    """文本向量化 Provider 抽象基类（单元测试可注入确定性 Mock 实现）。"""

    name: str = "base"
    model: str = "unknown"

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        将一批文本转换为等长的浮点向量列表。

        参数：
          texts - 待向量化的文本列表

        返回：
          与输入顺序严格一一对应的二维浮点向量列表；
          网络超时、鉴权失败等异常由调用方统一捕获并降级为纯 BM25。
        """
        raise NotImplementedError


class OpenAICompatibleEmbeddingProvider(BaseEmbeddingProvider):
    """基于官方 openai SDK 的 OpenAI 兼容协议在线 Embedding 适配器。"""

    name = "openai_compatible"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_sec: float | None = None,
        batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    ) -> None:
        self.api_key: str = (api_key if api_key is not None else os.getenv("EMBEDDING_API_KEY", "")).strip()
        self.base_url: str = (
            base_url if base_url is not None else os.getenv("EMBEDDING_BASE_URL", DEFAULT_EMBEDDING_BASE_URL)
        ).strip() or DEFAULT_EMBEDDING_BASE_URL
        self.model: str = (
            model if model is not None else os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        ).strip() or DEFAULT_EMBEDDING_MODEL

        raw_timeout = os.getenv("EMBEDDING_TIMEOUT_SEC", str(DEFAULT_EMBEDDING_TIMEOUT_SEC))
        try:
            env_timeout = float(raw_timeout)
        except ValueError:
            env_timeout = DEFAULT_EMBEDDING_TIMEOUT_SEC
        self.timeout_sec: float = timeout_sec if timeout_sec is not None else max(env_timeout, 1.0)
        self.batch_size: int = max(batch_size, 1)

        if not self.api_key:
            raise EnvironmentError("缺少 EMBEDDING_API_KEY 环境变量，无法初始化在线 Embedding Provider。")

        # 显式传入 timeout 与 max_retries=1，防止上游抖动拖垮检索链路，超时后迅速触发 BM25 降级
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout_sec,
            max_retries=1,
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """分批调用 /embeddings 接口并按 index 排序还原严格对应关系。"""
        if not texts:
            return []

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch: Sequence[str] = texts[start : start + self.batch_size]
            response = self._client.embeddings.create(
                model=self.model,
                input=list(batch),
            )
            # 某些兼容厂商不保证返回顺序，按 item.index 排序确保与输入文本严格对齐
            ordered = sorted(response.data, key=lambda item: item.index)
            if len(ordered) != len(batch):
                raise RuntimeError(
                    f"Embedding 接口返回条数异常：期望 {len(batch)} 条，实际 {len(ordered)} 条"
                )
            for item in ordered:
                vectors.append([float(v) for v in item.embedding])

        return vectors


# ---------------------------------------------------------------------------
# 进程级懒加载工厂（支持未配置时优雅返回 None，以及单测显式注入 Mock）
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_cached_key: tuple[str, str, str] | None = None
_cached_instance: BaseEmbeddingProvider | None = None
_override_active: bool = False
_override_provider: BaseEmbeddingProvider | None = None
_warned_missing_key: bool = False


def get_embedding_provider() -> BaseEmbeddingProvider | None:
    """
    获取当前进程的在线 Embedding Provider。
    若未配置 EMBEDDING_API_KEY（或仍为占位符），记录 WARNING 并返回 None，
    上层 KnowledgeRetriever 将自动平滑降级为纯 BM25 检索。
    """
    global _cached_key, _cached_instance, _warned_missing_key

    with _lock:
        if _override_active:
            return _override_provider

        api_key = os.getenv("EMBEDDING_API_KEY", "").strip()
        if not api_key or api_key.lower() in _PLACEHOLDER_KEYS:
            if not _warned_missing_key:
                logger.warning(
                    "embedding: 未配置 EMBEDDING_API_KEY，向量检索已关闭，知识库检索将自动降级为纯 BM25 模式。"
                )
                _warned_missing_key = True
            return None

        base_url = os.getenv("EMBEDDING_BASE_URL", DEFAULT_EMBEDDING_BASE_URL).strip() or DEFAULT_EMBEDDING_BASE_URL
        model = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL).strip() or DEFAULT_EMBEDDING_MODEL
        env_key = (api_key, base_url, model)

        if _cached_instance is None or _cached_key != env_key:
            _cached_instance = OpenAICompatibleEmbeddingProvider(
                api_key=api_key,
                base_url=base_url,
                model=model,
            )
            _cached_key = env_key
            _warned_missing_key = False
            logger.info(
                "embedding: 在线 Embedding Provider 初始化完成 model=%s base_url=%s",
                model,
                base_url,
            )

        return _cached_instance


def set_embedding_provider(
    provider: BaseEmbeddingProvider | None,
    *,
    force_none: bool = False,
) -> None:
    """
    显式注入或重置 Embedding Provider（主要用于单元测试）。
      - 传入具体 Provider：锁定使用该实例（如 MockEmbeddingProvider）；
      - 传入 None 且 force_none=True：强制返回 None（模拟无 Key 降级）；
      - 传入 None 且 force_none=False：清除覆盖与缓存，恢复按环境变量读取。
    """
    global _cached_key, _cached_instance, _override_active, _override_provider, _warned_missing_key
    with _lock:
        if provider is not None:
            _override_active = True
            _override_provider = provider
        elif force_none:
            _override_active = True
            _override_provider = None
        else:
            _override_active = False
            _override_provider = None
            _cached_key = None
            _cached_instance = None
            _warned_missing_key = False
