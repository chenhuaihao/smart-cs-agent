/**
 * frontend/src/types/chat.ts
 *
 * 与后端 FastAPI Pydantic v2 Schema 严格对齐的 TypeScript 类型定义：
 *   - app/schemas/chat.py（SSE 事件协议、断线补发、HITL 授权）
 *   - app/agent/guardrails.py（EvidenceLedger、CitationDetail）
 *   - app/schemas/ticket.py（工单模型）
 *   - app/schemas/order.py（订单模型）
 *   - app/routers/knowledge.py（Hybrid RAG 混合检索模型）
 */

export type EventType =
  | 'stage'
  | 'tool_call'
  | 'citation'
  | 'token'
  | 'approval_required'
  | 'done'
  | 'error';

export type TurnStatus = 'running' | 'waiting_approval' | 'completed' | 'failed';

export type RetrievalSource = 'bm25' | 'vector' | 'hybrid';

export interface CitationDetail {
  /** 条款来源，格式 "文件名#章节名" */
  source: string;
  /** 命中来源：bm25（仅关键词）/ vector（仅语义）/ hybrid（双路融合） */
  retrieval_source: RetrievalSource;
  /** 相关性得分（纯 BM25 分或 RRF 融合分） */
  score: number;
}

export interface OrderEvidence {
  order_id: string;
  status: string;
  delivered_at?: string | null;
  product_name: string;
  tracking_no?: string | null;
  exists: boolean;
}

export interface TicketEvidence {
  ticket_id: string;
  order_id: string;
  issue_type: string;
  status: string;
}

export interface EvidenceLedger {
  orders: Record<string, OrderEvidence>;
  citations: string[];
  citation_details: CitationDetail[];
  tickets: Record<string, TicketEvidence>;
}

// ---------------------------------------------------------------------------
// SSE 事件负载（按 type 区分）
// ---------------------------------------------------------------------------

export interface StageEventData {
  stage:
    | 'received'
    | 'thinking'
    | 'tool_running'
    | 'evidence_updated'
    | 'verifying'
    | 'fallback'
    | string;
  session_id?: string;
  step?: number;
  tools?: string[];
  tool?: string;
  evidence_ledger?: EvidenceLedger;
  passed?: boolean;
  reasons?: string[];
  reason?: string;
}

export interface ToolCallEventData {
  call_id: string;
  name: string;
  arguments: Record<string, unknown> | string;
  status: 'success' | 'error' | 'cancelled';
  duration_ms: number;
}

export interface CitationEventData {
  call_id: string;
  citations: string[];
  citation_details?: CitationDetail[];
}

export interface TokenEventData {
  delta: string;
  step: number;
  fallback?: boolean;
}

export interface ApprovalRequiredEventData {
  turn_id: string;
  tool: string;
  call_id: string;
  args: Record<string, unknown>;
  summary: string;
}

export interface DoneEventData {
  reply: string;
  tools_called: string[];
  citations: string[];
  ttft_ms: number | null;
  latency_ms: number;
}

export interface ErrorEventData {
  message?: string;
  reply?: string;
  tools_called?: string[];
  citations?: string[];
  ttft_ms?: number | null;
  latency_ms?: number;
}

export interface ChatEvent<T = Record<string, unknown>> {
  turn_id: string;
  seq: number;
  type: EventType;
  data: T;
  created_at: string;
}

export interface TurnEventsResponse {
  turn_id: string;
  status: TurnStatus;
  events: ChatEvent[];
  last_seq: number | null;
}

// ---------------------------------------------------------------------------
// 业务实体：工单 / 订单 / 知识库检索
// ---------------------------------------------------------------------------

export type TicketIssueType = '退货' | '换货' | '物流催单' | '售后咨询' | '转人工';

export interface Ticket {
  ticket_id: string;
  order_id: string;
  issue_type: string;
  reason: string | null;
  status: string;
  summary: string | null;
  created_at: string;
}

export interface Order {
  order_id: string;
  user_name: string;
  product_name: string;
  amount: number;
  status: string;
  tracking_no: string | null;
  created_at: string;
}

export interface KnowledgeHit {
  source: string;
  section: string;
  content: string;
  score: number;
  retrieval_source: RetrievalSource | string;
}

export interface KnowledgeSearchResponse {
  query: string;
  top_k: number;
  min_score: number;
  total_hits: number;
  results: KnowledgeHit[];
}

// ---------------------------------------------------------------------------
// 评测报告汇总（源自 eval/eval_summary.json，由评测脚本自动生成）
// ---------------------------------------------------------------------------

export interface EvalKpiCard {
  key: string;
  label: string;
  value: string;
  detail: string;
  tone: string;
}

export interface RagConfigMetric {
  name: string;
  is_production: boolean;
  hit_at_3: number;
  formal_hit_at_3: number;
  colloquial_hit_at_3: number;
  mrr: number;
  precision_at_3: number;
  ood_fpr: number;
  ood_fp_label: string;
  avg_ms: number;
  p95_ms: number;
}

export interface EvalFailureCase {
  id: string;
  source_eval: string;
  category: string;
  query: string;
  actual: string;
  reason: string;
  analysis: string;
}

export interface EvalDatasetInfo {
  rag_total: number;
  rag_breakdown: string;
  e2e_total: number;
  e2e_breakdown: string;
  construction_note: string;
  reproduce_commands: string[];
}

export interface EvalSummary {
  generated_at: string;
  embedding_model: string;
  gate_params: Record<string, number | string>;
  kpis: EvalKpiCard[];
  rag_configs: RagConfigMetric[];
  e2e_summary: Record<string, number | string | null>;
  failure_cases: EvalFailureCase[];
  dataset: EvalDatasetInfo;
}

// ---------------------------------------------------------------------------
// 前端对话流展示模型（一条助手消息对应一个完整 turn 的可观测状态）
// ---------------------------------------------------------------------------

export type ApprovalUiState = 'pending' | 'submitting' | 'approved' | 'rejected';

export interface ApprovalState {
  data: ApprovalRequiredEventData;
  state: ApprovalUiState;
  error?: string;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  createdAt: string;
  turnId?: string;
  status?: TurnStatus;
  /** 当前最新阶段文案，如 "正在思考（第 1 步）" / "正在调用工具..." */
  stageLabel?: string;
  stageCode?: string;
  /** 本轮已消费的最大 seq（用于 UI 展示与断线补发） */
  lastSeq?: number;
  /** 本轮是否触发过断线补发 */
  replayedCount?: number;
  tools: ToolCallEventData[];
  citations: string[];
  citationDetails: CitationDetail[];
  evidenceLedger?: EvidenceLedger;
  approval?: ApprovalState;
  ttftMs?: number | null;
  latencyMs?: number;
  isFallback?: boolean;
  errorMessage?: string;
}
