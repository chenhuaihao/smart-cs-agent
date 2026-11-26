"""
app/agent/knowledge.py

本地知识库 Hybrid RAG 检索模块（v2.0 P4-B）：
  - 保留原有 jieba.cut_for_search + rank_bm25 (BM25Okapi) 切块、分词与关键词门控检索；
  - Pre-RRF 双路原生门控：
    * BM25 路：绝对下限 BM25_MIN_SCORE（默认 3.0）+ 相对衰减 BM25_MIN_RATIO（默认 0.5）；
    * Vector 路：绝对余弦相似度 VECTOR_MIN_SIMILARITY（默认 0.48）+ 相对落差 VECTOR_MAX_DROP（默认 0.10）；
  - RRF 融合 + 单路尾部裁剪（RRF_SINGLE_PATH_RATIO=0.6）：当存在 hybrid 双路命中时，裁剪弱相关单路尾部；
  - 进程级常驻单例 + 线程安全 Query Embedding LRU 缓存（maxsize=256）+ 全链路分段耗时日志。
"""

from collections import OrderedDict
import logging
import os
from pathlib import Path
import re
import string
import threading
import time
from typing import Any, Literal, TypedDict

import jieba
from rank_bm25 import BM25Okapi

from app.agent.vector_store import ChromaVectorStore
from app.llm.embedding import BaseEmbeddingProvider, get_embedding_provider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量配置
# ---------------------------------------------------------------------------

_PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
_KNOWLEDGE_DIR: Path = _PROJECT_ROOT / "data" / "knowledge"
_CHROMA_DIR: Path = _PROJECT_ROOT / "data" / "chroma_db"

# RRF 平滑常数与 Pre-RRF 双路原生相关性门控默认值：
# 阈值基于 40 题开发校准集（eval/rag_eval_cases.json，域内最小 0.619 / 域外最大 0.568）定型，
# 并已在 18 题未参与调参的独立留出集（eval/rag_holdout_cases.json）上零改参复核通过（详见 eval/rag_holdout_report.md）。
DEFAULT_RRF_K: int = 60
DEFAULT_BM25_MIN_SCORE: float = 3.0
DEFAULT_BM25_MIN_RATIO: float = 0.50
DEFAULT_VECTOR_MIN_SIMILARITY: float = 0.58
DEFAULT_VECTOR_MAX_DROP: float = 0.10
DEFAULT_RRF_SINGLE_PATH_RATIO: float = 0.60
DEFAULT_QUERY_EMBED_CACHE_SIZE: int = 256


def _read_float_env(name: str, default: float, min_val: float = 0.0) -> float:
    """读取浮点型环境变量，未设置或非法时回退默认值。"""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        parsed = float(raw.strip())
        return parsed if parsed >= min_val else default
    except ValueError:
        return default


_STOPWORDS: set[str] = {
    "的", "了", "是", "在", "我", "您", "他", "她", "它",
    "们", "这", "那", "有", "和", "也", "就", "都", "不",
    "与", "或", "及", "以", "但", "而", "如", "对", "对于",
    "什么", "怎么", "如何", "可以", "能", "会", "要", "吗",
    "呢", "啊", "吧", "哦", "嗯", "请问", "期", "办",
    "你们", "我们", "店", "本店", "问下", "帮我", "查下", "一下", "支持",
}

_HEADING_RE = re.compile(r"^(#{2,3})\s+(.+)$", re.MULTILINE)

RetrievalSource = Literal["bm25", "vector", "hybrid"]


# ---------------------------------------------------------------------------
# 类型定义
# ---------------------------------------------------------------------------

class KnowledgeChunk(TypedDict):
    """单个知识库文档块结构（兼容 v1.0 字段，P2 新增 chunk_id 与 retrieval_source）。"""
    chunk_id: str
    source: str
    section: str
    content: str
    score: float
    retrieval_source: RetrievalSource


# ---------------------------------------------------------------------------
# 核心检索类
# ---------------------------------------------------------------------------

class KnowledgeRetriever:
    """
    Hybrid RAG 知识库检索器：
      - 词法路：jieba 中文分词 + BM25Okapi + 标题相关性/覆盖度/相对衰减门控；
      - 语义路：在线 Embedding API（带 LRU 缓存）+ Chroma 本地持久化向量检索（余弦相似度）；
      - 融合层：RRF 倒数排名融合 + 单路尾部裁剪，无 Key/异常自动降级纯 BM25。
    """

    def __init__(
        self,
        knowledge_dir: Path = _KNOWLEDGE_DIR,
        chroma_dir: Path = _CHROMA_DIR,
        embedding_provider: BaseEmbeddingProvider | None = None,
        *,
        enable_vector: bool = True,
        enable_pre_rrf_gate: bool = True,
        rrf_k: int = DEFAULT_RRF_K,
        bm25_min_score: float | None = None,
        bm25_min_ratio: float | None = None,
        vector_min_similarity: float | None = None,
        vector_max_drop: float | None = None,
        rrf_single_path_ratio: float | None = None,
        query_cache_size: int = DEFAULT_QUERY_EMBED_CACHE_SIZE,
    ) -> None:
        self._dir = knowledge_dir
        self._chroma_dir = chroma_dir
        self._rrf_k = max(rrf_k, 1)
        self._enable_pre_rrf_gate = enable_pre_rrf_gate
        self._bm25_min_score = (
            bm25_min_score
            if bm25_min_score is not None
            else _read_float_env("BM25_MIN_SCORE", DEFAULT_BM25_MIN_SCORE)
        )
        self._bm25_min_ratio = (
            bm25_min_ratio
            if bm25_min_ratio is not None
            else _read_float_env("BM25_MIN_RATIO", DEFAULT_BM25_MIN_RATIO)
        )
        self._vector_min_similarity = (
            vector_min_similarity
            if vector_min_similarity is not None
            else _read_float_env("VECTOR_MIN_SIMILARITY", DEFAULT_VECTOR_MIN_SIMILARITY)
        )
        self._vector_max_drop = (
            vector_max_drop
            if vector_max_drop is not None
            else _read_float_env("VECTOR_MAX_DROP", DEFAULT_VECTOR_MAX_DROP)
        )
        self._rrf_single_path_ratio = (
            rrf_single_path_ratio
            if rrf_single_path_ratio is not None
            else _read_float_env("RRF_SINGLE_PATH_RATIO", DEFAULT_RRF_SINGLE_PATH_RATIO)
        )
        self._query_cache_size = max(query_cache_size, 0)
        self._query_embed_cache: OrderedDict[tuple[str, str], list[float]] = OrderedDict()
        self._query_cache_lock = threading.Lock()
        self.last_search_metrics: dict[str, Any] = {}

        self._chunks: list[KnowledgeChunk] = []
        self._chunk_by_id: dict[str, KnowledgeChunk] = {}
        self._tokenized_corpus: list[list[str]] = []
        self._section_tokens: list[set[str]] = []
        self._bm25: BM25Okapi | None = None

        self._embedding_provider: BaseEmbeddingProvider | None = (
            embedding_provider if embedding_provider is not None else (get_embedding_provider() if enable_vector else None)
        )
        self._vector_store: ChromaVectorStore | None = None
        self._vector_ready: bool = False

        self._load_and_index()

    @property
    def vector_enabled(self) -> bool:
        """当前实例是否处于 Hybrid（BM25 + 向量）可用状态。"""
        return self._vector_ready and self._vector_store is not None and self._embedding_provider is not None

    # ------------------------------------------------------------------
    # 私有方法：分词、切块与双路索引构建
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> list[str]:
        """对中文文本做 cut_for_search 分词，过滤停用词和标点，保留数字（如"7天"、"15天"）。"""
        raw_tokens = jieba.cut_for_search(text)
        tokens: list[str] = []
        for tok in raw_tokens:
            tok = tok.strip()
            if not tok:
                continue
            if all(c in string.punctuation + "，。！？；：\"\"''、【】《》（）…—" for c in tok):
                continue
            if tok in _STOPWORDS:
                continue
            tokens.append(tok)
        return tokens

    def _parse_md_chunks(self, md_path: Path) -> list[KnowledgeChunk]:
        """按 ## / ### 标题切分 Markdown 文件，为每块生成稳定 chunk_id（文件名#章节名）。"""
        try:
            text = md_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error("knowledge: 读取文件 %s 失败: %s", md_path, e)
            return []

        source = md_path.name
        chunks: list[KnowledgeChunk] = []
        seen_ids: dict[str, int] = {}

        def _make_chunk(section: str, content: str) -> KnowledgeChunk:
            base_id = f"{source}#{section}"
            count = seen_ids.get(base_id, 0) + 1
            seen_ids[base_id] = count
            chunk_id = base_id if count == 1 else f"{base_id}#{count}"
            return KnowledgeChunk(
                chunk_id=chunk_id,
                source=source,
                section=section,
                content=content,
                score=0.0,
                retrieval_source="bm25",
            )

        headings = [
            (m.start(), len(m.group(1)), m.group(2).strip())
            for m in _HEADING_RE.finditer(text)
        ]

        if not headings:
            content = text.strip()
            if content:
                chunks.append(_make_chunk("全文", content))
            return chunks

        headings.append((len(text), 0, ""))
        current_h2: str = ""

        for i, (start, level, title) in enumerate(headings[:-1]):
            next_start = headings[i + 1][0]
            if level == 2:
                current_h2 = title
                section = title
            else:
                section = f"{current_h2} > {title}" if current_h2 else title

            heading_line_end = text.index("\n", start) + 1 if "\n" in text[start:] else len(text)
            content = text[heading_line_end:next_start].strip()
            if not content:
                continue
            chunks.append(_make_chunk(section, content))

        logger.debug("knowledge: 文件 %s 切分为 %d 块", source, len(chunks))
        return chunks

    def _load_and_index(self) -> None:
        """扫描知识库并构建 BM25 索引；若在线 Embedding 可用，同步构建或复用 Chroma 持久化索引。"""
        if not self._dir.exists():
            logger.warning("knowledge: 知识库目录不存在: %s", self._dir)
            return

        md_files = sorted(self._dir.glob("*.md"))
        if not md_files:
            logger.warning("knowledge: 知识库目录 %s 中未找到 .md 文件", self._dir)
            return

        for md_path in md_files:
            self._chunks.extend(self._parse_md_chunks(md_path))

        if not self._chunks:
            logger.warning("knowledge: 切分后无有效文档块，检索将返回空结果")
            return

        self._tokenized_corpus = [
            self._tokenize(chunk["section"] + " " + chunk["content"])
            for chunk in self._chunks
        ]

        valid_indices = [i for i, toks in enumerate(self._tokenized_corpus) if toks]
        if len(valid_indices) < len(self._chunks):
            logger.warning("knowledge: %d 个块分词后为空，已跳过", len(self._chunks) - len(valid_indices))
            self._chunks = [self._chunks[i] for i in valid_indices]
            self._tokenized_corpus = [self._tokenized_corpus[i] for i in valid_indices]

        self._chunk_by_id = {c["chunk_id"]: c for c in self._chunks}
        self._section_tokens = [set(self._tokenize(chunk["section"])) for chunk in self._chunks]
        self._bm25 = BM25Okapi(self._tokenized_corpus, k1=1.5, b=0.75)

        logger.info(
            "knowledge: BM25 索引构建完成，共 %d 块，来源文件: %s",
            len(self._chunks),
            [f.name for f in md_files],
        )

        if self._embedding_provider is not None:
            self._vector_store = ChromaVectorStore(persist_dir=self._chroma_dir)
            self._vector_ready = self._vector_store.sync_chunks(
                [dict(c) for c in self._chunks],
                self._embedding_provider,
            )

    # ------------------------------------------------------------------
    # 私有方法：BM25 检索、Query 向量缓存与 RRF 融合
    # ------------------------------------------------------------------

    def _search_bm25(
        self,
        query: str,
        top_k: int,
        min_score: float,
        *,
        apply_gating: bool = True,
    ) -> list[KnowledgeChunk]:
        """BM25 关键词检索 + 标题/覆盖度/相对衰减门控；支持关闭门控用于消融评测。"""
        if self._bm25 is None or not self._chunks:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            logger.debug("knowledge: query 分词后为空，BM25 返回空结果")
            return []

        scores = self._bm25.get_scores(query_tokens)
        query_token_set = set(query_tokens)
        query_token_count = len(query_token_set)

        scored_chunks: list[KnowledgeChunk] = []
        for i in range(len(self._chunks)):
            sc = float(scores[i])
            if apply_gating:
                if sc < min_score:
                    continue

                doc_tokens = set(self._tokenized_corpus[i])
                matched_tokens = query_token_set & doc_tokens
                coverage = len(matched_tokens) / query_token_count
                matched_in_section = query_token_set & self._section_tokens[i]

                if query_token_count >= 3 and len(matched_tokens) < 2:
                    continue
                if len(matched_in_section) == 0 and (len(matched_tokens) < 3 or sc < 7.0):
                    continue
                if query_token_count >= 5:
                    if len(matched_tokens) < 3 and coverage < 0.35:
                        continue
                    if len(matched_tokens) == 2 and sc < 7.0:
                        continue
                if query_token_count >= 8:
                    if len(matched_tokens) < 3 or (len(matched_tokens) < 4 and coverage < 0.30):
                        continue
            else:
                if sc <= 0.0:
                    continue

            scored_chunks.append({**self._chunks[i], "score": sc, "retrieval_source": "bm25"})

        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        # BM25 相对衰减门控：剔除得分低于 Top-1 50% 的边缘凑数切片
        if apply_gating and scored_chunks and self._bm25_min_ratio > 0:
            top1_bm25 = scored_chunks[0]["score"]
            min_allowed = top1_bm25 * self._bm25_min_ratio
            scored_chunks = [c for c in scored_chunks if c["score"] >= min_allowed]
        return scored_chunks[:top_k]

    def _embed_query_cached(self, query_text: str) -> tuple[list[float], bool]:
        """线程安全 Query Embedding 内存 LRU 缓存（maxsize=256），高频问法毫秒级命中。"""
        assert self._embedding_provider is not None
        cache_key = (f"{self._embedding_provider.name}:{self._embedding_provider.model}", query_text)
        if self._query_cache_size > 0:
            with self._query_cache_lock:
                cached = self._query_embed_cache.get(cache_key)
                if cached is not None:
                    self._query_embed_cache.move_to_end(cache_key)
                    return cached, True

        vectors = self._embedding_provider.embed_texts([query_text])
        if not vectors:
            return [], False
        vec = vectors[0]

        if self._query_cache_size > 0:
            with self._query_cache_lock:
                self._query_embed_cache[cache_key] = vec
                self._query_embed_cache.move_to_end(cache_key)
                while len(self._query_embed_cache) > self._query_cache_size:
                    self._query_embed_cache.popitem(last=False)
        return vec, False

    def _rrf_fuse(
        self,
        bm25_hits: list[KnowledgeChunk],
        vector_hits: list[tuple[str, float]],
        top_k: int,
        *,
        apply_tail_prune: bool = True,
        rrf_min_score: float = 0.0,
    ) -> list[KnowledgeChunk]:
        """
        Reciprocal Rank Fusion（倒数排名融合，k=60）+ 单路尾部裁剪：
          - 当存在 hybrid 双路命中切片时，剔除 RRF 分数低于 Top-1 * 0.6 的单路尾部切片；
          - 若传入 rrf_min_score > 0，再按最低融合分数过滤。
        """
        bm25_ranks: dict[str, int] = {
            chunk["chunk_id"]: rank for rank, chunk in enumerate(bm25_hits, start=1)
        }
        vector_ranks: dict[str, int] = {}
        for rank, (cid, _sim) in enumerate(vector_hits, start=1):
            if cid in self._chunk_by_id and cid not in vector_ranks:
                vector_ranks[cid] = rank

        all_ids = set(bm25_ranks) | set(vector_ranks)
        if not all_ids:
            return []

        fused: list[tuple[float, int, str, KnowledgeChunk]] = []
        for cid in all_ids:
            in_bm25 = cid in bm25_ranks
            in_vec = cid in vector_ranks
            rrf_score = 0.0
            if in_bm25:
                rrf_score += 1.0 / (self._rrf_k + bm25_ranks[cid])
            if in_vec:
                rrf_score += 1.0 / (self._rrf_k + vector_ranks[cid])

            if in_bm25 and in_vec:
                retrieval_source: RetrievalSource = "hybrid"
                best_rank = min(bm25_ranks[cid], vector_ranks[cid])
            elif in_bm25:
                retrieval_source = "bm25"
                best_rank = bm25_ranks[cid]
            else:
                retrieval_source = "vector"
                best_rank = vector_ranks[cid]

            base_chunk = self._chunk_by_id[cid]
            merged: KnowledgeChunk = {
                **base_chunk,
                "score": round(rrf_score, 6),
                "retrieval_source": retrieval_source,
            }
            fused.append((rrf_score, best_rank, cid, merged))

        fused.sort(key=lambda item: (-item[0], item[1], item[2]))

        # 单路尾部裁剪：存在高质量 hybrid 命中时，过滤落后显著的单路蹭分项
        has_hybrid = any(item[3]["retrieval_source"] == "hybrid" for item in fused)
        if apply_tail_prune and has_hybrid and fused and self._rrf_single_path_ratio > 0:
            top1_rrf = fused[0][0]
            tail_floor = top1_rrf * self._rrf_single_path_ratio
            fused = [
                item
                for item in fused
                if item[3]["retrieval_source"] == "hybrid" or item[0] >= tail_floor
            ]

        if rrf_min_score > 0:
            fused = [item for item in fused if item[0] >= rrf_min_score]

        return [item[3] for item in fused[:top_k]]

    # ------------------------------------------------------------------
    # 公共方法：混合检索入口（带分段耗时打点）
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 3.0,
        *,
        enable_pre_rrf_gate: bool | None = None,
        rrf_min_score: float = 0.0,
    ) -> list[KnowledgeChunk]:
        """
        对 query 执行检索并记录各阶段分段耗时（bm25_ms / embed_api_ms / chroma_ms / rrf_ms / total_ms）。
        """
        total_start = time.perf_counter()
        if self._bm25 is None or not self._chunks:
            logger.warning("knowledge: 索引未初始化，返回空结果")
            return []

        clean_query = query.strip() if query else ""
        if not clean_query:
            return []

        use_gate = self._enable_pre_rrf_gate if enable_pre_rrf_gate is None else enable_pre_rrf_gate

        if not self.vector_enabled:
            t_bm25 = time.perf_counter()
            results = self._search_bm25(clean_query, top_k=top_k, min_score=min_score, apply_gating=use_gate)
            bm25_ms = round((time.perf_counter() - t_bm25) * 1000, 2)
            total_ms = round((time.perf_counter() - total_start) * 1000, 2)
            self.last_search_metrics = {
                "mode": "bm25",
                "bm25_ms": bm25_ms,
                "embed_api_ms": 0.0,
                "chroma_ms": 0.0,
                "rrf_ms": 0.0,
                "total_ms": total_ms,
                "cache_hit": False,
            }
            logger.info(
                "knowledge[bm25]: query='%s' hits=%d bm25_ms=%.2f total_ms=%.2f",
                clean_query[:50], len(results), bm25_ms, total_ms,
            )
            return results

        candidate_k = max(top_k * 2, 6)
        bm25_floor = max(min_score, self._bm25_min_score) if use_gate else 0.0

        t_bm25 = time.perf_counter()
        bm25_hits = self._search_bm25(clean_query, top_k=candidate_k, min_score=bm25_floor, apply_gating=use_gate)
        bm25_ms = round((time.perf_counter() - t_bm25) * 1000, 2)

        embed_api_ms = 0.0
        chroma_ms = 0.0
        cache_hit = False
        try:
            assert self._vector_store is not None and self._embedding_provider is not None
            t_embed = time.perf_counter()
            query_vec, cache_hit = self._embed_query_cached(clean_query)
            embed_api_ms = round((time.perf_counter() - t_embed) * 1000, 2)

            t_chroma = time.perf_counter()
            vector_hits = self._vector_store.query_by_vector(
                query_vector=query_vec,
                top_k=candidate_k,
                min_similarity=self._vector_min_similarity if use_gate else 0.0,
                max_drop=self._vector_max_drop if use_gate else None,
            )
            chroma_ms = round((time.perf_counter() - t_chroma) * 1000, 2)
        except Exception as e:
            logger.warning("knowledge: 在线向量检索异常，本次请求平滑降级为纯 BM25: %s", e)
            total_ms = round((time.perf_counter() - total_start) * 1000, 2)
            self.last_search_metrics = {
                "mode": "bm25_fallback",
                "bm25_ms": bm25_ms,
                "embed_api_ms": embed_api_ms,
                "chroma_ms": chroma_ms,
                "rrf_ms": 0.0,
                "total_ms": total_ms,
                "cache_hit": cache_hit,
            }
            return bm25_hits[:top_k]

        t_rrf = time.perf_counter()
        results = self._rrf_fuse(
            bm25_hits,
            vector_hits,
            top_k=top_k,
            apply_tail_prune=use_gate,
            rrf_min_score=rrf_min_score,
        )
        rrf_ms = round((time.perf_counter() - t_rrf) * 1000, 2)
        total_ms = round((time.perf_counter() - total_start) * 1000, 2)
        self.last_search_metrics = {
            "mode": "hybrid",
            "bm25_ms": bm25_ms,
            "embed_api_ms": embed_api_ms,
            "chroma_ms": chroma_ms,
            "rrf_ms": rrf_ms,
            "total_ms": total_ms,
            "cache_hit": cache_hit,
        }

        logger.info(
            "knowledge[hybrid]: query='%s' hits=%d (gate=%s, bm25=%d, vec=%d) "
            "bm25_ms=%.2f embed_api_ms=%.2f(cache=%s) chroma_ms=%.2f rrf_ms=%.2f total_ms=%.2f",
            clean_query[:50],
            len(results),
            use_gate,
            len(bm25_hits),
            len(vector_hits),
            bm25_ms,
            embed_api_ms,
            cache_hit,
            chroma_ms,
            rrf_ms,
            total_ms,
        )
        return results


# ---------------------------------------------------------------------------
# 模块级单例（进程常驻，支持单测显式注入与重置）
# ---------------------------------------------------------------------------

_retriever_instance: KnowledgeRetriever | None = None
_retriever_lock = threading.Lock()


def get_retriever() -> KnowledgeRetriever:
    """获取 KnowledgeRetriever 进程级常驻单例（线程安全懒加载 / 启动预热）。"""
    global _retriever_instance
    if _retriever_instance is not None:
        return _retriever_instance
    with _retriever_lock:
        if _retriever_instance is None:
            logger.info("knowledge: 首次初始化 KnowledgeRetriever 单例...")
            _retriever_instance = KnowledgeRetriever()
    return _retriever_instance


def set_retriever(retriever: KnowledgeRetriever | None) -> None:
    """显式注入或重置 KnowledgeRetriever 单例（主要用于单元测试）。"""
    global _retriever_instance
    with _retriever_lock:
        _retriever_instance = retriever
