/**
 * frontend/src/api/client.ts
 *
 * 后端 REST + SSE 流式 API 客户端：
 *   - 支持 POST /api/chat/stream 与 POST /api/chat/turns/{turn_id}/approve 的 SSE 逐帧解析；
 *   - 正确处理 TCP 分包截断（按 \n\n 切帧并保留半截缓冲）、解析 id / event / data 与 X-Turn-Id 响应头；
 *   - 封装 GET /api/chat/turns/{turn_id}/events、工单、订单与 Hybrid RAG 知识库检索接口。
 */

import type {
  ChatEvent,
  EvalSummary,
  EventType,
  KnowledgeSearchResponse,
  Order,
  Ticket,
  TurnEventsResponse,
} from '../types/chat';

export class ApiError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(detail || `HTTP ${status}`);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
}

async function parseErrorDetail(resp: Response): Promise<string> {
  try {
    const text = await resp.text();
    if (!text) return `请求失败（HTTP ${resp.status}）`;
    const parsed = JSON.parse(text) as { detail?: unknown; message?: string };
    if (typeof parsed.detail === 'string') return parsed.detail;
    if (Array.isArray(parsed.detail)) {
      return parsed.detail.map((d) => JSON.stringify(d)).join('；');
    }
    return parsed.message || text;
  } catch {
    return `请求失败（HTTP ${resp.status}）`;
  }
}

async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(url, init);
  if (!resp.ok) {
    throw new ApiError(resp.status, await parseErrorDetail(resp));
  }
  return (await resp.json()) as T;
}

// ---------------------------------------------------------------------------
// SSE 流式解析器（供 /api/chat/stream 与 /api/chat/turns/{turn_id}/approve 共用）
// ---------------------------------------------------------------------------

export interface StreamHandlers {
  onTurnId?: (turnId: string) => void;
  onEvent: (event: ChatEvent) => void;
}

/**
 * 解析单个 SSE 文本块（以空行分隔），严格读取 id / event / data。
 * 后端格式：
 *   id: 1
 *   event: stage
 *   data: {"turn_id":"...","seq":1,"type":"stage","data":{...},"created_at":"..."}
 */
function parseSseBlock(rawBlock: string): ChatEvent | null {
  const lines = rawBlock.split('\n');
  let idStr = '';
  let eventName = '';
  const dataLines: string[] = [];

  for (const rawLine of lines) {
    const line = rawLine.trimEnd();
    if (!line || line.startsWith(':')) continue;
    const colonIdx = line.indexOf(':');
    if (colonIdx === -1) continue;
    const field = line.slice(0, colonIdx);
    const value = line.slice(colonIdx + 1).replace(/^ /, '');
    if (field === 'id') idStr = value;
    else if (field === 'event') eventName = value;
    else if (field === 'data') dataLines.push(value);
  }

  if (dataLines.length === 0) return null;
  const payloadStr = dataLines.join('\n');
  const parsed = JSON.parse(payloadStr) as Partial<ChatEvent>;

  const seq = typeof parsed.seq === 'number' ? parsed.seq : Number.parseInt(idStr, 10);
  if (!Number.isFinite(seq) || seq < 1) return null;

  return {
    turn_id: parsed.turn_id ?? '',
    seq,
    type: (parsed.type ?? eventName) as EventType,
    data: (parsed.data ?? {}) as Record<string, unknown>,
    created_at: parsed.created_at ?? '',
  };
}

async function consumeSseResponse(
  resp: Response,
  handlers: StreamHandlers,
): Promise<string | null> {
  if (!resp.ok) {
    throw new ApiError(resp.status, await parseErrorDetail(resp));
  }

  const headerTurnId = resp.headers.get('X-Turn-Id') || resp.headers.get('x-turn-id');
  if (headerTurnId && handlers.onTurnId) {
    handlers.onTurnId(headerTurnId);
  }

  if (!resp.body) {
    throw new Error('响应体不支持流式读取（ReadableStream 为空）');
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';
  let resolvedTurnId = headerTurnId;

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');

      let sepIndex = buffer.indexOf('\n\n');
      while (sepIndex !== -1) {
        const block = buffer.slice(0, sepIndex).trim();
        buffer = buffer.slice(sepIndex + 2);
        if (block) {
          const evt = parseSseBlock(block);
          if (evt) {
            if (!resolvedTurnId && evt.turn_id) {
              resolvedTurnId = evt.turn_id;
              handlers.onTurnId?.(evt.turn_id);
            }
            handlers.onEvent(evt);
          }
        }
        sepIndex = buffer.indexOf('\n\n');
      }
    }

    // 处理流关闭前缓冲区中可能残留的最后一帧
    const tail = (buffer + decoder.decode()).replace(/\r\n/g, '\n').trim();
    if (tail) {
      for (const block of tail.split('\n\n')) {
        const clean = block.trim();
        if (!clean) continue;
        const evt = parseSseBlock(clean);
        if (evt) {
          if (!resolvedTurnId && evt.turn_id) {
            resolvedTurnId = evt.turn_id;
            handlers.onTurnId?.(evt.turn_id);
          }
          handlers.onEvent(evt);
        }
      }
    }
  } finally {
    reader.releaseLock();
  }

  return resolvedTurnId;
}

/** 发起一轮 SSE 流式对话（POST /api/chat/stream） */
export async function streamChat(
  sessionId: string,
  message: string,
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<string | null> {
  const resp = await fetch('/api/chat/stream', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
    },
    body: JSON.stringify({ session_id: sessionId, message }),
    signal,
  });
  return consumeSseResponse(resp, handlers);
}

/** 断线补发：拉取 turn 内 seq > afterSeq 的事件（GET /api/chat/turns/{turn_id}/events） */
export async function getTurnEvents(
  turnId: string,
  afterSeq: number,
  signal?: AbortSignal,
): Promise<TurnEventsResponse> {
  const params = new URLSearchParams({ after_seq: String(Math.max(0, afterSeq)) });
  return requestJson<TurnEventsResponse>(
    `/api/chat/turns/${encodeURIComponent(turnId)}/events?${params.toString()}`,
    { signal },
  );
}

/** 高风险写操作人工授权续流（POST /api/chat/turns/{turn_id}/approve） */
export async function approveTurnStream(
  turnId: string,
  approved: boolean,
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<string | null> {
  const resp = await fetch(`/api/chat/turns/${encodeURIComponent(turnId)}/approve`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
    },
    body: JSON.stringify({ approved }),
    signal,
  });
  return consumeSseResponse(resp, handlers);
}

// ---------------------------------------------------------------------------
// 工单 / 订单 / 知识库联调接口
// ---------------------------------------------------------------------------

export async function listTickets(filters?: {
  orderId?: string;
  status?: string;
}): Promise<Ticket[]> {
  const params = new URLSearchParams();
  if (filters?.orderId?.trim()) params.set('order_id', filters.orderId.trim());
  if (filters?.status?.trim()) params.set('status', filters.status.trim());
  const qs = params.toString();
  return requestJson<Ticket[]>(`/api/tickets${qs ? `?${qs}` : ''}`);
}

export async function updateTicketStatus(ticketId: string, status: string): Promise<Ticket> {
  return requestJson<Ticket>(`/api/tickets/${encodeURIComponent(ticketId)}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ status }),
  });
}

export async function resetDemoTickets(
  adminToken?: string,
): Promise<{ message: string; orders_count?: number; count: number }> {
  const headers: Record<string, string> = {};
  if (adminToken && adminToken.trim()) {
    headers['X-Demo-Admin-Token'] = adminToken.trim();
  }
  return requestJson<{ message: string; orders_count?: number; count: number }>(
    '/api/tickets/reset',
    {
      method: 'POST',
      headers,
    },
  );
}

export async function listOrders(status?: string): Promise<Order[]> {
  const params = new URLSearchParams();
  if (status?.trim()) params.set('status', status.trim());
  const qs = params.toString();
  return requestJson<Order[]>(`/api/orders${qs ? `?${qs}` : ''}`);
}

export async function getOrderById(orderId: string): Promise<Order> {
  return requestJson<Order>(`/api/orders/${encodeURIComponent(orderId.trim())}`);
}

export async function searchKnowledge(
  query: string,
  topK: number,
  options?: {
    minScore?: number;
    enablePreRrfGate?: boolean;
    rrfMinScore?: number;
  },
): Promise<KnowledgeSearchResponse> {
  const params = new URLSearchParams({
    q: query.trim(),
    top_k: String(topK),
    min_score: String(options?.minScore ?? 3.0),
    enable_pre_rrf_gate: String(options?.enablePreRrfGate ?? true),
    rrf_min_score: String(options?.rrfMinScore ?? 0.0),
  });
  return requestJson<KnowledgeSearchResponse>(`/api/knowledge/search?${params.toString()}`);
}

export async function checkHealth(): Promise<{ status: string; service: string }> {
  return requestJson<{ status: string; service: string }>('/health');
}

export async function getEvalSummary(): Promise<EvalSummary> {
  return requestJson<EvalSummary>('/api/eval/summary');
}
