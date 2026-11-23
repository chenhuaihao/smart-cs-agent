"""
tests/test_p2_hybrid_rag.py

v2.0 P2 阶段验收测试（使用 Mock BaseEmbeddingProvider + 临时 Chroma 目录，不依赖真实外网与 API Key）：
  1. 无 EMBEDDING_API_KEY 或在线 Embedding 超时/报错时平滑降级为纯 BM25，检索正常且 retrieval_source == "bm25"；
  2. 启用 Mock Embedding 时：
     - 口语化查询（BM25 无字面命中）可通过向量路召回并标注 retrieval_source == "vector"；
     - 双路同时命中的文档经 RRF (k=60) 融合后排在最前且标注 retrieval_source == "hybrid"；
     - SSE citation 事件、evidence_ledger 与 /api/knowledge/search 正确透传 citation_details，
       同时保留原有 citations: list[str]（'文件名#章节名'）契约 100% 兼容；
  3. 知识库内容未变时二次加载直接复用本地 Chroma 持久化索引，0 次重复调用 Embedding API；
     修改单个文档时仅增量向量化变更切片。
"""

import json
import logging
import math
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi.testclient import TestClient
import pytest

from app.agent.knowledge import KnowledgeRetriever, set_retriever
from app.db import database
from app.llm.base import BaseLLMProvider, LLMStreamChunk, ToolCallRequest
from app.llm.embedding import BaseEmbeddingProvider, get_embedding_provider, set_embedding_provider
from app.llm.factory import set_llm_provider
from app.main import app


# ---------------------------------------------------------------------------
# 确定性语义 Mock Embedding Provider（零外网、零本地模型）
# ---------------------------------------------------------------------------

# 6 维主题词表：用于把文档与口语化查询映射到可解释的语义空间
_TOPIC_KEYWORDS: list[tuple[str, ...]] = (
    # dim 0: 耳机单侧无声 / 一边听不见（含口语化表达，避免使用父级标题词导致同文件切片同分）
    ("单侧无声", "左耳", "右耳", "听不见", "没声", "没动静", "一边响"),
    # dim 1: 拆封耳机退货
    ("拆封", "无质量问题", "外包装", "拆开", "退货", "不想买"),
    # dim 2: 质保维修
    ("质保", "保修", "维修", "寄修", "365天"),
    # dim 3: 发票开具
    ("发票", "普票", "专票", "抬头", "税号"),
    # dim 4: 价保降价
    ("价保", "降价", "差价", "买贵了"),
    # dim 5: 机械键盘故障
    ("机械键盘", "连击", "串键", "轴体", "按键"),
)


class MockSemanticEmbeddingProvider(BaseEmbeddingProvider):
    """
    确定性语义向量 Mock：
      - 将文本按主题词表映射为归一化 7 维向量（前 6 维主题 + 第 7 维极小兜底维度避免零向量）；
      - 记录 embed_texts 调用次数与每次传入的文本列表，用于验证持久化索引复用与增量更新。
    """

    name = "mock_semantic"
    model = "mock-bge-m3"

    def __init__(self, fail_after_calls: int | None = None) -> None:
        self.call_count: int = 0
        self.embedded_batches: list[list[str]] = []
        self._fail_after_calls = fail_after_calls

    def _encode_one(self, text: str) -> list[float]:
        vec = [0.0] * (len(_TOPIC_KEYWORDS) + 1)
        for idx, keywords in enumerate(_TOPIC_KEYWORDS):
            hits = sum(1 for kw in keywords if kw in text)
            if hits > 0:
                vec[idx] = float(hits)
        if all(v == 0.0 for v in vec[:-1]):
            # 无任何主题命中时落到独立正交维度，保证与主题查询的余弦相似度为 0
            vec[-1] = 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.call_count += 1
        self.embedded_batches.append(list(texts))
        if self._fail_after_calls is not None and self.call_count > self._fail_after_calls:
            raise TimeoutError("模拟在线 Embedding API 请求超时")
        return [self._encode_one(t) for t in texts]


class _KnowledgeToolLLMProvider(BaseLLMProvider):
    """用于端到端 SSE 验收的 Mock LLM：首轮调 search_knowledge_base，次轮输出最终回复。"""

    name = "mock-rag-llm"

    async def astream_chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[LLMStreamChunk]:
        last = messages[-1]
        if last["role"] == "tool":
            yield LLMStreamChunk(type="text_delta", text="根据知识库条款，已为您核实相关售后规则。")
            return
        yield LLMStreamChunk(
            type="tool_calls",
            tool_calls=[
                ToolCallRequest(
                    id="call_rag_1",
                    name="search_knowledge_base",
                    arguments=json.dumps({"query": last["content"]}, ensure_ascii=False),
                )
            ],
        )


def _parse_sse(body: str) -> list[dict[str, Any]]:
    frames = []
    for block in body.replace("\r\n", "\n").strip().split("\n\n"):
        if not block.strip():
            continue
        fields: dict[str, str] = {}
        for line in block.split("\n"):
            key, sep, value = line.partition(": ")
            assert sep, f"非法 SSE 行：{line!r}"
            fields[key] = value
        frames.append({"id": int(fields["id"]), "event": fields["event"], "data": json.loads(fields["data"])})
    return frames


@pytest.fixture(autouse=True)
def _cleanup_singletons():
    """每个用例结束后清理模块级单例，防止测试间相互污染。"""
    yield
    set_retriever(None)
    set_embedding_provider(None)
    set_llm_provider(None)


# ---------------------------------------------------------------------------
# 1. 无 EMBEDDING_API_KEY / 在线超时异常时平滑降级为纯 BM25
# ---------------------------------------------------------------------------

def test_degrades_to_bm25_when_api_key_missing(tmp_path: Path, monkeypatch, caplog) -> None:
    """未配置 EMBEDDING_API_KEY 时：打 WARNING 日志、降级为纯 BM25、正常返回且 retrieval_source == 'bm25'。"""
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    set_embedding_provider(None)

    with caplog.at_level(logging.WARNING):
        provider = get_embedding_provider()
        assert provider is None
        retriever = KnowledgeRetriever(chroma_dir=tmp_path / "chroma_no_key")

    assert retriever.vector_enabled is False
    assert any("EMBEDDING_API_KEY" in rec.message for rec in caplog.records)

    hits = retriever.search("耳机拆封后还能退货吗", top_k=3)
    assert len(hits) > 0
    assert all(h["retrieval_source"] == "bm25" for h in hits)
    assert all("#" in h["chunk_id"] for h in hits)
    assert any(h["source"] == "return_policy.md" for h in hits)


def test_degrades_to_bm25_when_embedding_api_fails_at_init_or_query(tmp_path: Path, caplog) -> None:
    """在线 Embedding 在建索引或查询阶段超时抛错时，绝不阻塞或崩溃，平滑降级为纯 BM25。"""
    # 场景 A：启动建索引时即超时失败
    broken_init_provider = MockSemanticEmbeddingProvider(fail_after_calls=0)
    with caplog.at_level(logging.WARNING):
        retriever_a = KnowledgeRetriever(
            chroma_dir=tmp_path / "chroma_broken_init",
            embedding_provider=broken_init_provider,
        )
    assert retriever_a.vector_enabled is False
    hits_a = retriever_a.search("耳机拆封后还能退货吗", top_k=3)
    assert len(hits_a) > 0
    assert all(h["retrieval_source"] == "bm25" for h in hits_a)

    # 场景 B：建索引成功，但在线查询时超时
    flaky_query_provider = MockSemanticEmbeddingProvider(fail_after_calls=1)
    retriever_b = KnowledgeRetriever(
        chroma_dir=tmp_path / "chroma_flaky_query",
        embedding_provider=flaky_query_provider,
    )
    assert retriever_b.vector_enabled is True
    with caplog.at_level(logging.WARNING):
        hits_b = retriever_b.search("耳机拆封后还能退货吗", top_k=3)
    assert len(hits_b) > 0
    assert all(h["retrieval_source"] == "bm25" for h in hits_b)
    assert any("降级为纯 BM25" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# 2. Hybrid RAG：口语化向量召回 + RRF 双路融合排序 + 上游 citation_details 透传
# ---------------------------------------------------------------------------

def test_colloquial_query_recalled_by_vector_and_dual_hit_ranked_top_as_hybrid(tmp_path: Path) -> None:
    """
    验证双路 Hybrid RAG 核心效果：
      - 口语化查询（纯 BM25 无法命中）由向量路召回并标注 retrieval_source == 'vector'；
      - 同时命中 BM25 与向量路的文档经 RRF (k=60) 融合后排在最前且标注 retrieval_source == 'hybrid'。
    """
    provider = MockSemanticEmbeddingProvider()
    chroma_dir = tmp_path / "chroma_hybrid"

    bm25_only = KnowledgeRetriever(chroma_dir=chroma_dir, enable_vector=False)
    hybrid = KnowledgeRetriever(chroma_dir=chroma_dir, embedding_provider=provider)
    assert hybrid.vector_enabled is True

    # 2a. 口语化查询：不含"蓝牙耳机/单侧无声/排查"等书面词，纯 BM25 召回为空，Hybrid 通过向量路命中
    colloquial_query = "戴着一边完全没动静听不见啦咋弄"
    assert bm25_only.search(colloquial_query, top_k=3) == []

    vec_hits = hybrid.search(colloquial_query, top_k=3)
    assert len(vec_hits) > 0
    assert vec_hits[0]["retrieval_source"] == "vector"
    assert vec_hits[0]["source"] == "troubleshooting_guide.md"
    assert "单侧无声" in vec_hits[0]["section"]

    # 2b. 双路同时命中：关键词与语义同时命中的条款经 RRF 叠加（>= 2/(60+rank) > 1/61），稳居首位且标记 hybrid
    dual_query = "蓝牙耳机单侧无声排查步骤"
    hybrid_hits = hybrid.search(dual_query, top_k=3)
    assert len(hybrid_hits) > 0
    top_hit = hybrid_hits[0]
    assert top_hit["retrieval_source"] == "hybrid"
    assert top_hit["source"] == "troubleshooting_guide.md"
    assert "单侧无声" in top_hit["section"]
    # 单路第 1 名最高分为 1/61 ≈ 0.016393；双路融合分必然严格大于 1/61
    assert top_hit["score"] > round(1.0 / 61.0, 6)


def test_sse_citation_and_evidence_ledger_include_citation_details(tmp_path: Path, monkeypatch) -> None:
    """
    端到端验证：
      - SSE citation 事件 data 同时包含 citations（'文件名#章节名'）与 citation_details；
      - evidence_updated stage 事件的 evidence_ledger 包含 citation_details；
      - /api/knowledge/search 返回 retrieval_source 字段；
      - POST /api/chat 响应契约 100% 保持不变。
    """
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "test_p2.db"))
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "1000")
    monkeypatch.setenv("DAILY_CHAT_LIMIT", "100000")

    provider = MockSemanticEmbeddingProvider()
    retriever = KnowledgeRetriever(chroma_dir=tmp_path / "chroma_e2e", embedding_provider=provider)
    set_retriever(retriever)
    set_llm_provider(_KnowledgeToolLLMProvider())

    with TestClient(app) as client:
        # 1) 验证调试检索接口暴露 retrieval_source
        search_resp = client.get("/api/knowledge/search", params={"q": "蓝牙耳机单侧无声排查步骤", "top_k": 3})
        assert search_resp.status_code == 200
        search_body = search_resp.json()
        assert search_body["total_hits"] > 0
        assert search_body["results"][0]["retrieval_source"] == "hybrid"

        # 2) 验证 SSE 流式接口透传 citation_details 至 citation 事件与 evidence_ledger
        stream_resp = client.post(
            "/api/chat/stream",
            json={"session_id": "s-p2-sse", "message": "蓝牙耳机单侧无声排查步骤"},
        )
        assert stream_resp.status_code == 200
        frames = _parse_sse(stream_resp.text)

        citation_events = [f["data"]["data"] for f in frames if f["event"] == "citation"]
        assert len(citation_events) == 1
        cit_data = citation_events[0]
        assert len(cit_data["citations"]) > 0
        assert all("#" in c for c in cit_data["citations"])
        assert "citation_details" in cit_data
        assert len(cit_data["citation_details"]) == len(cit_data["citations"])
        assert cit_data["citation_details"][0]["source"] == cit_data["citations"][0]
        assert cit_data["citation_details"][0]["retrieval_source"] == "hybrid"
        assert cit_data["citation_details"][0]["score"] > 0

        ledger_events = [
            f["data"]["data"]["evidence_ledger"]
            for f in frames
            if f["event"] == "stage" and f["data"]["data"].get("stage") == "evidence_updated"
        ]
        assert len(ledger_events) == 1
        ledger = ledger_events[0]
        assert ledger["citations"] == cit_data["citations"]
        assert len(ledger["citation_details"]) == len(cit_data["citations"])
        assert ledger["citation_details"][0]["retrieval_source"] == "hybrid"

        # 3) 验证同步接口 POST /api/chat 契约 100% 向后兼容
        sync_resp = client.post(
            "/api/chat",
            json={"session_id": "s-p2-sync", "message": "蓝牙耳机单侧无声排查步骤"},
        )
        assert sync_resp.status_code == 200
        sync_body = sync_resp.json()
        assert set(sync_body) == {"session_id", "reply", "tools_called", "latency_ms", "citations"}
        assert sync_body["citations"] == cit_data["citations"]


# ---------------------------------------------------------------------------
# 3. 持久化索引复用与内容变更增量更新
# ---------------------------------------------------------------------------

def test_persisted_chroma_index_reused_without_reembedding_when_unchanged(tmp_path: Path) -> None:
    """
    验证内容哈希比对机制：
      - 首次加载：调用 Embedding API 对全部切片建索引并落盘；
      - 内容未变二次加载：直接复用 Chroma 持久化索引，0 次 Embedding API 调用；
      - 修改其中一个文件后再次加载：仅对变更切片增量调用 Embedding API，而非全量重算。
    """
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    doc_a = kb_dir / "policy_a.md"
    doc_b = kb_dir / "policy_b.md"
    doc_a.write_text("## 耳机拆封退货\n\n拆封后无质量问题不支持退货。\n", encoding="utf-8")
    doc_b.write_text("## 电子发票开具\n\n确认收货后24小时内自动开具电子普票。\n", encoding="utf-8")

    chroma_dir = tmp_path / "chroma_persist"

    # 第 1 次加载：索引为空，全量向量化 2 块
    provider_1 = MockSemanticEmbeddingProvider()
    r1 = KnowledgeRetriever(knowledge_dir=kb_dir, chroma_dir=chroma_dir, embedding_provider=provider_1)
    assert r1.vector_enabled is True
    assert provider_1.call_count == 1
    assert len(provider_1.embedded_batches[0]) == 2

    # 第 2 次加载（文档未变）：直接复用本地持久化索引，0 次 Embedding API 调用
    provider_2 = MockSemanticEmbeddingProvider()
    r2 = KnowledgeRetriever(knowledge_dir=kb_dir, chroma_dir=chroma_dir, embedding_provider=provider_2)
    assert r2.vector_enabled is True
    assert provider_2.call_count == 0

    # 查询仅向量化 1 条 query，不重复向量化语料
    res = r2.search("买贵了发票抬头写错了怎么办", top_k=2)
    assert len(res) > 0
    assert provider_2.call_count == 1
    assert provider_2.embedded_batches[0] == ["买贵了发票抬头写错了怎么办"]

    # 第 3 次加载（仅修改 policy_b.md）：增量向量化 1 块，而非全量 2 块
    doc_b.write_text("## 电子发票开具\n\n确认收货后24小时内自动开具增值税电子普通发票，支持重开一次。\n", encoding="utf-8")
    provider_3 = MockSemanticEmbeddingProvider()
    r3 = KnowledgeRetriever(knowledge_dir=kb_dir, chroma_dir=chroma_dir, embedding_provider=provider_3)
    assert r3.vector_enabled is True
    assert provider_3.call_count == 1
    assert len(provider_3.embedded_batches[0]) == 1
