"""
app/agent/knowledge.py

本地知识库 Hybrid RAG 检索模块（v2.0 P2）：
  - 保留原有 jieba.cut_for_search + rank_bm25 (BM25Okapi) 切块、分词与关键词门控检索；
  - 接入 Chroma 轻量本地向量持久化（data/chroma_db）与在线 Embedding 适配器；
  - 实现 BM25 + 向量双路召回与 RRF（Reciprocal Rank Fusion，k=60）融合排序，
    并为每条结果标注 retrieval_source（bm25 / vector / hybrid）；
  - 优雅降级：未配置 EMBEDDING_API_KEY 或在线请求超时/失败时自动降级为纯 BM25，绝不阻塞服务。
"""

import logging
from pathlib import Path
import re
import string
from typing import Literal, TypedDict

import jieba
from rank_bm25 import BM25Okapi

from app.agent.vector_store import ChromaVectorStore
from app.llm.embedding import BaseEmbeddingProvider, get_embedding_provider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量配置
# ---------------------------------------------------------------------------

_PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
# 知识库目录与 Chroma 持久化目录（不依赖运行目录 CWD）
_KNOWLEDGE_DIR: Path = _PROJECT_ROOT / "data" / "knowledge"
_CHROMA_DIR: Path = _PROJECT_ROOT / "data" / "chroma_db"

# RRF 平滑常数与向量路余弦相似度最低阈值
DEFAULT_RRF_K: int = 60
DEFAULT_VECTOR_MIN_SIMILARITY: float = 0.35

# 中文常见停用词（高频虚词与对客客套词，对语义无贡献，过滤后提升检索精度）
_STOPWORDS: set[str] = {
    "的", "了", "是", "在", "我", "您", "他", "她", "它",
    "们", "这", "那", "有", "和", "也", "就", "都", "不",
    "与", "或", "及", "以", "但", "而", "如", "对", "对于",
    "什么", "怎么", "如何", "可以", "能", "会", "要", "吗",
    "呢", "啊", "吧", "哦", "嗯", "请问", "期", "办",
    "你们", "我们", "店", "本店", "问下", "帮我", "查下", "一下", "支持",
}

# 匹配 ## 或 ### 级标题的正则
_HEADING_RE = re.compile(r"^(#{2,3})\s+(.+)$", re.MULTILINE)

RetrievalSource = Literal["bm25", "vector", "hybrid"]


# ---------------------------------------------------------------------------
# 类型定义
# ---------------------------------------------------------------------------

class KnowledgeChunk(TypedDict):
    """单个知识库文档块结构（兼容 v1.0 字段，P2 新增 chunk_id 与 retrieval_source）。"""
    chunk_id: str                      # 稳定唯一标识，格式 "文件名#章节名"
    source: str                        # 来源文件名，如 "return_policy.md"
    section: str                       # 章节标题，如 "耳机类退货规则 > 拆封耳机——无质量问题"
    content: str                       # 该章节正文内容
    score: float                       # 相关性得分（纯 BM25 模式为 BM25 分，Hybrid 模式为 RRF 分）
    retrieval_source: RetrievalSource  # 命中来源：bm25 / vector / hybrid


# ---------------------------------------------------------------------------
# 核心检索类
# ---------------------------------------------------------------------------

class KnowledgeRetriever:
    """
    Hybrid RAG 知识库检索器：
      - 词法路：jieba 中文分词 + BM25Okapi + 标题相关性/覆盖度门控；
      - 语义路：在线 Embedding API + Chroma 本地持久化向量检索（余弦相似度）；
      - 融合层：RRF 倒数排名融合（score = sum(1 / (k + rank))），无 Key/异常自动降级纯 BM25。
    """

    def __init__(
        self,
        knowledge_dir: Path = _KNOWLEDGE_DIR,
        chroma_dir: Path = _CHROMA_DIR,
        embedding_provider: BaseEmbeddingProvider | None = None,
        *,
        enable_vector: bool = True,
        rrf_k: int = DEFAULT_RRF_K,
        vector_min_similarity: float = DEFAULT_VECTOR_MIN_SIMILARITY,
    ) -> None:
        self._dir = knowledge_dir
        self._chroma_dir = chroma_dir
        self._rrf_k = max(rrf_k, 1)
        self._vector_min_similarity = vector_min_similarity

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
            # 防止同文件极端情况出现重名章节导致 Chroma 主键冲突
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

        # 按需初始化/复用 Chroma 向量索引；失败时记录 WARNING 并保持纯 BM25 可用
        if self._embedding_provider is not None:
            self._vector_store = ChromaVectorStore(persist_dir=self._chroma_dir)
            self._vector_ready = self._vector_store.sync_chunks(
                [dict(c) for c in self._chunks],
                self._embedding_provider,
            )

    # ------------------------------------------------------------------
    # 私有方法：BM25 检索与 RRF 融合
    # ------------------------------------------------------------------

    def _search_bm25(self, query: str, top_k: int, min_score: float) -> list[KnowledgeChunk]:
        """保留原有 BM25 + 标题相关性/词元覆盖度门控检索逻辑不变。"""
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
            if sc < min_score:
                continue

            doc_tokens = set(self._tokenized_corpus[i])
            matched_tokens = query_token_set & doc_tokens
            coverage = len(matched_tokens) / query_token_count
            matched_in_section = query_token_set & self._section_tokens[i]

            # 1. 基础命中门槛：query >= 3 个词元时，至少命中 2 个不同词元
            if query_token_count >= 3 and len(matched_tokens) < 2:
                continue
            # 2. 标题相关性门控：章节标题与 query 无交集时，要求命中 >= 3 词且高分 (>= 7.0)
            if len(matched_in_section) == 0 and (len(matched_tokens) < 3 or sc < 7.0):
                continue
            # 3. 中长句词元覆盖度门控：防虚假共现高分
            if query_token_count >= 5:
                if len(matched_tokens) < 3 and coverage < 0.35:
                    continue
                if len(matched_tokens) == 2 and sc < 7.0:
                    continue
            if query_token_count >= 8:
                if len(matched_tokens) < 3 or (len(matched_tokens) < 4 and coverage < 0.30):
                    continue

            scored_chunks.append({**self._chunks[i], "score": sc, "retrieval_source": "bm25"})

        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return scored_chunks[:top_k]

    def _rrf_fuse(
        self,
        bm25_hits: list[KnowledgeChunk],
        vector_hits: list[tuple[str, float]],
        top_k: int,
    ) -> list[KnowledgeChunk]:
        """
        Reciprocal Rank Fusion（倒数排名融合）：
          score(d) = sum(1 / (k + rank(d)))，rank 从 1 开始计数，默认 k=60。
        同时依据两路命中情况为每条结果打标：bm25 / vector / hybrid。
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

        # 按 RRF 分降序；同分时按单路最优名次升序、chunk_id 稳定排序
        fused.sort(key=lambda item: (-item[0], item[1], item[2]))
        return [item[3] for item in fused[:top_k]]

    # ------------------------------------------------------------------
    # 公共方法：混合检索入口
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 3.0,
    ) -> list[KnowledgeChunk]:
        """
        对 query 执行检索，返回按相关性降序排列的命中条款列表。
          - 在线 Embedding + Chroma 就绪时：执行 BM25 + 向量双路召回 + RRF 融合排序；
          - 无 Key 或线上请求超时/失败时：自动平滑降级为纯 BM25 检索并打 WARNING 日志。
        """
        if self._bm25 is None or not self._chunks:
            logger.warning("knowledge: 索引未初始化，返回空结果")
            return []

        if not query or not query.strip():
            return []

        # 降级路径：未启用向量检索时，保持与 v1.0 完全一致的纯 BM25 行为
        if not self.vector_enabled:
            results = self._search_bm25(query, top_k=top_k, min_score=min_score)
            if not results:
                logger.info("knowledge[bm25]: query='%s' 在阈值 min_score=%.1f 下无命中", query[:50], min_score)
            else:
                logger.info(
                    "knowledge[bm25]: query='%s' 命中 %d 块，top1=%s score=%.3f",
                    query[:50], len(results), results[0]["chunk_id"], results[0]["score"],
                )
            return results

        # Hybrid 路径：两路各自扩大候选池后进行 RRF 融合
        candidate_k = max(top_k * 2, 6)
        bm25_hits = self._search_bm25(query, top_k=candidate_k, min_score=min_score)

        try:
            assert self._vector_store is not None and self._embedding_provider is not None
            vector_hits = self._vector_store.query(
                query_text=query.strip(),
                provider=self._embedding_provider,
                top_k=candidate_k,
                min_similarity=self._vector_min_similarity,
            )
        except Exception as e:
            logger.warning("knowledge: 在线向量检索异常，本次请求平滑降级为纯 BM25: %s", e)
            return bm25_hits[:top_k]

        results = self._rrf_fuse(bm25_hits, vector_hits, top_k=top_k)
        if not results:
            logger.info("knowledge[hybrid]: query='%s' 双路均无命中", query[:50])
        else:
            logger.info(
                "knowledge[hybrid]: query='%s' 命中 %d 块（bm25=%d, vector=%d），top1=%s [%s] score=%.6f",
                query[:50],
                len(results),
                len(bm25_hits),
                len(vector_hits),
                results[0]["chunk_id"],
                results[0]["retrieval_source"],
                results[0]["score"],
            )
        return results


# ---------------------------------------------------------------------------
# 模块级单例（支持单测显式注入与重置）
# ---------------------------------------------------------------------------

_retriever_instance: KnowledgeRetriever | None = None


def get_retriever() -> KnowledgeRetriever:
    """获取 KnowledgeRetriever 单例（懒加载，首次调用时构建/复用索引）。"""
    global _retriever_instance
    if _retriever_instance is None:
        logger.info("knowledge: 首次初始化 KnowledgeRetriever 单例...")
        _retriever_instance = KnowledgeRetriever()
    return _retriever_instance


def set_retriever(retriever: KnowledgeRetriever | None) -> None:
    """显式注入或重置 KnowledgeRetriever 单例（主要用于单元测试）。"""
    global _retriever_instance
    _retriever_instance = retriever
