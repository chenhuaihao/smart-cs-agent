"""
app/routers/knowledge.py

知识库检索调试接口：GET /api/knowledge/search
用于在 /docs 中单独验证切块质量和 BM25 检索效果，不依赖大模型。
"""

import logging
from typing import List

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from app.agent.knowledge import get_retriever

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
    score: float = Field(..., description="BM25 相关性得分，越高越相关")


class KnowledgeSearchResponse(BaseModel):
    """检索接口响应体"""
    query: str = Field(..., description="原始查询文本")
    top_k: int = Field(..., description="请求的最大返回条数")
    total_hits: int = Field(..., description="实际命中条数（score > 0 的结果数，可能小于 top_k）")
    results: List[KnowledgeHit] = Field(default_factory=list, description="按相关性降序排列的命中列表")


# ---------------------------------------------------------------------------
# 检索接口
# ---------------------------------------------------------------------------

@router.get(
    "/search",
    response_model=KnowledgeSearchResponse,
    summary="检索本地知识库（调试用）",
    description=(
        "对本地 data/knowledge/*.md 中的售后规则文档执行 BM25 检索，"
        "返回按相关性排序的命中条款列表。"
        "该接口直接调用检索模块，不经过大模型，方便验证切块和检索质量。"
        "score=0 的块已被过滤，仅返回真正相关的结果。"
    ),
)
def search_knowledge(
    q: str = Query(..., min_length=1, description="查询关键词，如'耳机拆封退货'"),
    top_k: int = Query(default=3, ge=1, le=10, description="最多返回条数，默认 3，最大 10"),
) -> KnowledgeSearchResponse:
    """
    知识库检索调试接口。
    直接调用 KnowledgeRetriever.search()，完全绕过大模型，
    方便在 /docs 中逐条验证切块粒度和检索命中质量。
    """
    logger.info("GET /api/knowledge/search q='%s' top_k=%d", q[:50], top_k)

    retriever = get_retriever()
    # 单例已在首次请求（或服务启动）时初始化，此处直接使用
    raw_results = retriever.search(query=q, top_k=top_k)

    hits = [
        KnowledgeHit(
            source=chunk["source"],
            section=chunk["section"],
            content=chunk["content"],
            score=chunk["score"],
        )
        for chunk in raw_results
    ]

    return KnowledgeSearchResponse(
        query=q,
        top_k=top_k,
        total_hits=len(hits),
        results=hits,
    )
