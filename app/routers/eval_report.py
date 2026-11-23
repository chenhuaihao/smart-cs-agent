"""
app/routers/eval_report.py

只读评测汇总接口：GET /api/eval/summary
直接读取由 eval/run_rag_eval.py 自动产出的 eval/eval_summary.json，
确保前端「评测报告」页展示的数字与评测脚本/报告 100% 同源一致，杜绝手写硬编码。
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.core.limiter import check_api_read_rate_limit

logger = logging.getLogger(__name__)

_SUMMARY_PATH = Path(__file__).resolve().parents[2] / "eval" / "eval_summary.json"

router = APIRouter(
    prefix="/api/eval",
    tags=["eval"],
    dependencies=[Depends(check_api_read_rate_limit)],
)


class EvalKpiCard(BaseModel):
    key: str
    label: str
    value: str
    detail: str
    tone: str = "brand"


class RagConfigMetric(BaseModel):
    name: str
    is_production: bool = False
    hit_at_3: float
    formal_hit_at_3: float
    colloquial_hit_at_3: float
    mrr: float
    precision_at_3: float
    ood_fpr: float
    ood_fp_label: str
    avg_ms: float
    p95_ms: float


class EvalFailureCase(BaseModel):
    id: str
    source_eval: str
    category: str
    query: str
    actual: str
    reason: str
    analysis: str


class EvalDatasetInfo(BaseModel):
    rag_total: int
    rag_breakdown: str
    e2e_total: int
    e2e_breakdown: str
    construction_note: str
    reproduce_commands: List[str] = Field(default_factory=list)


class EvalSummaryResponse(BaseModel):
    generated_at: str
    embedding_model: str
    gate_params: Dict[str, Any] = Field(default_factory=dict)
    kpis: List[EvalKpiCard] = Field(default_factory=list)
    rag_configs: List[RagConfigMetric] = Field(default_factory=list)
    e2e_summary: Dict[str, Any] = Field(default_factory=dict)
    failure_cases: List[EvalFailureCase] = Field(default_factory=list)
    dataset: EvalDatasetInfo


@router.get(
    "/summary",
    response_model=EvalSummaryResponse,
    summary="获取最新评测汇总数据（由评测脚本自动生成）",
    description="直接读取 eval/eval_summary.json，为前端「评测报告」标签页提供单一事实数据源。",
)
def get_eval_summary() -> EvalSummaryResponse:
    if not _SUMMARY_PATH.exists():
        logger.warning("eval_report: 未找到评测汇总文件 %s", _SUMMARY_PATH)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="尚未生成 eval/eval_summary.json，请先运行 python eval/run_rag_eval.py。",
        )

    try:
        raw = json.loads(_SUMMARY_PATH.read_text(encoding="utf-8"))
        return EvalSummaryResponse.model_validate(raw)
    except Exception as e:
        logger.error("eval_report: 解析 %s 失败: %s", _SUMMARY_PATH, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"评测汇总文件解析失败：{str(e)}",
        )
