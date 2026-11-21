"""
app/agent/knowledge.py

本地知识库 BM25 检索模块。
职责：
  - 扫描 data/knowledge/*.md，按 ## / ### 标题切块
  - 使用 jieba.cut_for_search + rank_bm25 (BM25Okapi) 建立索引
  - 提供 search(query, top_k) 方法返回按相关性排序的命中条款列表
  - 模块级单例，避免每次请求重复读盘和构建索引
"""

import logging
import re
import string
from pathlib import Path
from typing import TypedDict

import jieba
import jieba.analyse
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量配置
# ---------------------------------------------------------------------------

# 知识库目录：基于本文件位置向上两级到项目根，再进入 data/knowledge
# 路径解析不依赖当前运行目录 (CWD)，适应任意启动方式
_KNOWLEDGE_DIR: Path = Path(__file__).resolve().parents[2] / "data" / "knowledge"

# 中文常见停用词（高频虚词，对语义无贡献，过滤后提升检索精度）
_STOPWORDS: set[str] = {
    "的", "了", "是", "在", "我", "您", "他", "她", "它",
    "们", "这", "那", "有", "和", "也", "就", "都", "不",
    "与", "或", "及", "以", "但", "而", "如", "对", "对于",
    "什么", "怎么", "如何", "可以", "能", "会", "要", "吗",
    "呢", "啊", "吧", "哦", "嗯",
}

# 匹配 ## 或 ### 级标题的正则
_HEADING_RE = re.compile(r"^(#{2,3})\s+(.+)$", re.MULTILINE)


# ---------------------------------------------------------------------------
# 类型定义
# ---------------------------------------------------------------------------

class KnowledgeChunk(TypedDict):
    """单个文档块结构"""
    source: str    # 来源文件名，如 "return_policy.md"
    section: str   # 章节标题，如 "耳机类退货规则"
    content: str   # 该章节正文内容
    score: float   # BM25 相关性得分（search 时填入，加载时为 0.0）


# ---------------------------------------------------------------------------
# 核心检索类
# ---------------------------------------------------------------------------

class KnowledgeRetriever:
    """
    本地 Markdown 知识库检索器。
    使用 jieba 中文分词 + BM25Okapi 实现轻量级全文检索，
    无需向量数据库或深度学习模型。
    """

    def __init__(self, knowledge_dir: Path = _KNOWLEDGE_DIR) -> None:
        self._dir = knowledge_dir
        self._chunks: list[KnowledgeChunk] = []  # 切分后的文档块
        self._tokenized_corpus: list[list[str]] = []  # 分词后的语料
        self._bm25: BM25Okapi | None = None          # 延迟初始化

        self._load_and_index()

    # ------------------------------------------------------------------
    # 私有方法：加载与索引构建
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> list[str]:
        """
        对中文文本进行分词，过滤停用词和标点。
        使用 cut_for_search 模式：在精确模式基础上对长词再细分，
        有利于命中短关键词（如"退货"、"换新"等）。
        """
        # 过滤 ASCII 标点和中文标点，保留汉字、字母、数字
        # 注意：不过滤数字（"7天"、"15天"等数字是重要检索词）
        raw_tokens = jieba.cut_for_search(text)
        tokens = []
        for tok in raw_tokens:
            tok = tok.strip()
            if not tok:
                continue
            # 过滤纯标点或纯空白词元
            if all(c in string.punctuation + "，。！？；：""''、【】《》（）…—" for c in tok):
                continue
            # 过滤停用词
            if tok in _STOPWORDS:
                continue
            tokens.append(tok)
        return tokens

    def _parse_md_chunks(self, md_path: Path) -> list[KnowledgeChunk]:
        """
        按 ## / ### 标题切分 Markdown 文件为独立文档块。
        切块逻辑：
          - 找到所有标题行的位置
          - 两个相邻标题之间的正文作为前一个标题的 content
          - 保留父级 ## 标题作为上下文前缀，拼入 section 字段
            方便检索时标题关键词也纳入索引
        """
        try:
            text = md_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error("knowledge: 读取文件 %s 失败: %s", md_path, e)
            return []

        source = md_path.name
        chunks: list[KnowledgeChunk] = []

        # 找出所有标题的 (match_start, level, title_text)
        headings = [
            (m.start(), len(m.group(1)), m.group(2).strip())
            for m in _HEADING_RE.finditer(text)
        ]

        if not headings:
            # 整个文件没有标题时，作为一个整块处理（兜底）
            chunks.append(KnowledgeChunk(
                source=source,
                section="全文",
                content=text.strip(),
                score=0.0,
            ))
            return chunks

        # 追加哨兵，方便取最后一个标题的正文
        headings.append((len(text), 0, ""))

        current_h2: str = ""  # 记录最近一个 ## 级别标题，用于给 ### 添加上下文

        for i, (start, level, title) in enumerate(headings[:-1]):
            next_start = headings[i + 1][0]

            if level == 2:
                current_h2 = title

            # section 字段：## 标题直接用 title，### 标题拼上父级 ##
            # 格式："父标题 > 子标题"，方便溯源展示
            if level == 2:
                section = title
            else:
                section = f"{current_h2} > {title}" if current_h2 else title

            # 提取正文（去掉标题行本身）
            heading_line_end = text.index("\n", start) + 1 if "\n" in text[start:] else len(text)
            content = text[heading_line_end:next_start].strip()

            # 过滤空白块（例如连续两个标题之间无内容）
            if not content:
                continue

            chunks.append(KnowledgeChunk(
                source=source,
                section=section,
                content=content,
                score=0.0,
            ))

        logger.debug("knowledge: 文件 %s 切分为 %d 块", source, len(chunks))
        return chunks

    def _load_and_index(self) -> None:
        """
        扫描知识库目录下所有 .md 文件，切块后构建 BM25 索引。
        在实例初始化时调用一次，之后复用（模块级单例保证不重复初始化）。
        """
        if not self._dir.exists():
            # 知识库目录不存在时打印警告，不抛出异常（服务仍可启动）
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

        # 将 section（章节标题）和 content（正文）一并纳入索引词袋，
        # 提高标题关键词的命中率（如用户问"耳机退货"时能命中标题含"耳机"的块）
        self._tokenized_corpus = [
            self._tokenize(chunk["section"] + " " + chunk["content"])
            for chunk in self._chunks
        ]

        # 过滤掉分词结果为空的块（极端情况：纯标点内容）
        valid_indices = [i for i, toks in enumerate(self._tokenized_corpus) if toks]
        if len(valid_indices) < len(self._chunks):
            logger.warning(
                "knowledge: %d 个块分词后为空，已跳过",
                len(self._chunks) - len(valid_indices),
            )
            self._chunks = [self._chunks[i] for i in valid_indices]
            self._tokenized_corpus = [self._tokenized_corpus[i] for i in valid_indices]

        # 构建 BM25 索引，k1=1.5 / b=0.75 为通用中文文档推荐参数
        self._bm25 = BM25Okapi(self._tokenized_corpus, k1=1.5, b=0.75)
        logger.info(
            "knowledge: 索引构建完成，共 %d 块，来源文件: %s",
            len(self._chunks),
            [f.name for f in md_files],
        )

    # ------------------------------------------------------------------
    # 公共方法：检索
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 3) -> list[KnowledgeChunk]:
        """
        对 query 进行 BM25 检索，返回按相关性降序排列的命中条款列表。

        参数：
            query  - 用户查询文本（中文自然语言）
            top_k  - 最多返回条款数，默认 3

        返回：
            命中块列表，每项包含 source / section / content / score；
            无命中时返回空列表（score=0 的块被过滤掉）。
        """
        # 知识库为空时直接返回空列表（不抛异常，调用方兜底）
        if self._bm25 is None or not self._chunks:
            logger.warning("knowledge: BM25 索引未初始化，返回空结果")
            return []

        # 对 query 分词
        query_tokens = self._tokenize(query)
        if not query_tokens:
            # 分词后无有效词元（如用户只输入标点），直接返回空
            logger.debug("knowledge: query 分词后为空，返回空结果")
            return []

        # 计算每个文档块的 BM25 得分
        scores = self._bm25.get_scores(query_tokens)

        # 过滤 score <= 0 的无关块，只保留有实际命中的结果
        # 边界原因：BM25 对完全不相关文档返回 0，保留 0 分块会误导大模型
        scored_chunks = [
            {**self._chunks[i], "score": float(scores[i])}
            for i in range(len(self._chunks))
            if scores[i] > 0
        ]

        if not scored_chunks:
            logger.info("knowledge: query='%s' 无命中结果", query[:50])
            return []

        # 按得分降序排序，取 top_k
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        results = scored_chunks[:top_k]

        logger.info(
            "knowledge: query='%s' 命中 %d 块（总正分块 %d），top1 score=%.3f",
            query[:50], len(results), len(scored_chunks), results[0]["score"],
        )
        return results  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# 模块级单例：只在第一次导入时初始化一次，后续所有请求复用同一实例
# ---------------------------------------------------------------------------

_retriever_instance: KnowledgeRetriever | None = None


def get_retriever() -> KnowledgeRetriever:
    """
    获取 KnowledgeRetriever 单例。
    使用懒加载模式：首次调用时初始化并缓存，之后直接返回缓存实例。
    线程安全说明：FastAPI 默认单进程多协程，CPython GIL 保证此处赋值原子性，
    极端并发下最多初始化两次（代价极低，不引入 Lock 保持简洁）。
    """
    global _retriever_instance
    if _retriever_instance is None:
        logger.info("knowledge: 首次初始化 KnowledgeRetriever 单例...")
        _retriever_instance = KnowledgeRetriever()
    return _retriever_instance
