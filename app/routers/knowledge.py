"""
app/routers/knowledge.py

知识库检索调试接口：GET /api/knowledge/search
用于在 /docs 中单独验证切块质量和 BM25 检索效果，不依赖大模型。
"""

import logging
from typing import List

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from app.agent.knowledge import get_retriever
from app.core.limiter import check_api_read_rate_limit

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])


# ---------------------------------------------------------------------------
# 响应 Schema
# ---------------------------------------------------------------------------

class KnowledgeHit(BaseModel):
    """单条知识库命中结果"""
    source: str = Field(..., description="来源文件名，如 return_policy.md")
    section: str = Field(..., description="命中的章节标题")
    content: str = Field(..., description="章节正文内容")
    score: float = Field(..., description="相关性得分（纯 BM25 分或 Hybrid RRF 融合分），越高越相关")
    retrieval_source: str = Field(
        default="bm25",
        description="命中来源：bm25（仅关键词）/ vector（仅语义）/ hybrid（双路融合）",
    )


class KnowledgeSearchResponse(BaseModel):
    """检索接口响应体"""
    query: str = Field(..., description="原始查询文本")
    top_k: int = Field(..., description="请求的最大返回条数")
    min_score: float = Field(..., description="使用的最低得分过滤阈值")
    total_hits: int = Field(..., description="实际命中条数（score >= min_score 的结果数，可能小于 top_k）")
    results: List[KnowledgeHit] = Field(default_factory=list, description="按相关性降序排列的命中列表")


# ---------------------------------------------------------------------------
# 检索接口
# ---------------------------------------------------------------------------

@router.get(
    "/search",
    response_model=KnowledgeSearchResponse,
    dependencies=[Depends(check_api_read_rate_limit)],
    summary="检索本地知识库（调试用）",
    description=(
        "对本地 data/knowledge/*.md 中的售后规则文档执行 BM25 检索，"
        "返回按相关性排序的命中条款列表。"
        "该接口直接调用检索模块，不经过大模型，方便验证切块和检索质量。"
        "低于 min_score 阈值的块已被过滤，仅返回高置信度相关的结果。"
    ),
)
def search_knowledge(
    q: str = Query(..., min_length=1, description="查询关键词，如'耳机拆封退货'"),
    top_k: int = Query(default=3, ge=1, le=10, description="最多返回条数，默认 3，最大 10"),
    min_score: float = Query(default=3.0, ge=0.0, description="纯 BM25 最低得分阈值，默认 3.0"),
    enable_pre_rrf_gate: bool = Query(default=True, description="是否开启 Pre-RRF 双路原生相关性门控与尾部裁剪"),
    rrf_min_score: float = Query(default=0.0, ge=0.0, le=1.0, description="Hybrid RRF 融合最小分数阈值（默认 0.0，推荐 0.016）"),
) -> KnowledgeSearchResponse:
    """
    知识库检索调试接口。
    直接调用 KnowledgeRetriever.search()，完全绕过大模型，
    方便验证切块粒度、Pre-RRF 门控开关、RRF 阈值与双路命中来源。
    """
    logger.info(
        "GET /api/knowledge/search q='%s' top_k=%d min_score=%.2f gate=%s rrf_min=%.4f",
        q[:50], top_k, min_score, enable_pre_rrf_gate, rrf_min_score,
    )

    retriever = get_retriever()
    raw_results = retriever.search(
        query=q,
        top_k=top_k,
        min_score=min_score,
        enable_pre_rrf_gate=enable_pre_rrf_gate,
        rrf_min_score=rrf_min_score,
    )

    hits = [
        KnowledgeHit(
            source=chunk["source"],
            section=chunk["section"],
            content=chunk["content"],
            score=chunk["score"],
            retrieval_source=chunk.get("retrieval_source", "bm25"),
        )
        for chunk in raw_results
    ]

    return KnowledgeSearchResponse(
        query=q,
        top_k=top_k,
        min_score=min_score,
        total_hits=len(hits),
        results=hits,
    )
