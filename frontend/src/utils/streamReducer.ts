/**
 * frontend/src/utils/streamReducer.ts
 *
 * v3.0 P3 纯函数化 SSE 帧解析器与流式事件状态机（无 DOM/React 副作用，可直接单元测试）：
 *   - parseSseBlock / feedSseChunk：处理 TCP 分包、粘包、CRLF 与注释心跳行；
 *   - applyEventToMessage：严格按 seq 幂等去重、阶段转换、工具前草稿清理、HITL 挂起与终态收敛。
 */

import type {
  ApprovalRequiredEventData,
  ChatEvent,
  ChatMessage,
  CitationDetail,
  CitationEventData,
  DoneEventData,
  ErrorEventData,
  EventType,
  StageEventData,
  TokenEventData,
  ToolCallEventData,
} from '../types/chat';

export interface TurnRuntimeMeta {
  messageId: string;
  processedSeqs: Set<number>;
  lastSeq: number;
  terminal: boolean;
  waitingApproval: boolean;
}

export function createTurnRuntimeMeta(messageId: string): TurnRuntimeMeta {
  return {
    messageId,
    processedSeqs: new Set<number>(),
    lastSeq: 0,
    terminal: false,
    waitingApproval: false,
  };
}

/**
 * 解析单个以空行分隔的 SSE 文本块，提取 id / event / data。
 */
export function parseSseBlock(rawBlock: string): ChatEvent | null {
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

/**
 * 增量向 SSE 缓冲区追加分片文本，返回本轮切出的完整 ChatEvent 列表与剩余半截缓冲。
 */
export function feedSseChunk(
  prevBuffer: string,
  incomingText: string,
): { events: ChatEvent[]; remainder: string } {
  let buffer = (prevBuffer + incomingText).replace(/\r\n/g, '\n');
  const events: ChatEvent[] = [];

  let sepIndex = buffer.indexOf('\n\n');
  while (sepIndex !== -1) {
    const block = buffer.slice(0, sepIndex).trim();
    buffer = buffer.slice(sepIndex + 2);
    if (block) {
      const evt = parseSseBlock(block);
      if (evt) events.push(evt);
    }
    sepIndex = buffer.indexOf('\n\n');
  }
  return { events, remainder: buffer };
}

export function formatStageLabel(data: StageEventData): string {
  switch (data.stage) {
    case 'received':
      return '已接收请求，准备进入推理...';
    case 'thinking':
      return `Agent 思考中（第 ${data.step ?? 1} 步）`;
    case 'tool_running':
      return `正在调用工具：${(data.tools ?? []).join('、') || '业务工具'}`;
    case 'evidence_updated':
      return `证据账本已更新（${data.tool ?? '工具结果'}）`;
    case 'verifying':
      return data.passed ? '规则护栏校验通过' : '规则校验拦截，触发安全重写...';
    case 'fallback':
      return `触发安全降级：${data.reason ?? '已转入安全兜底'}`;
    case 'approval_expired':
      return `审批已超时取消：${data.reason ?? '未创建任何工单'}`;
    case 'input_guard_blocked':
      return '输入安全护栏已拦截高风险指令';
    default:
      return `处理阶段：${data.stage}`;
  }
}

export function mergeCitationDetails(
  existing: CitationDetail[],
  incoming: CitationDetail[],
): CitationDetail[] {
  const merged = [...existing];
  const seen = new Set(merged.map((d) => d.source));
  for (const item of incoming) {
    if (!item.source || seen.has(item.source)) continue;
    merged.push(item);
    seen.add(item.source);
  }
  return merged;
}

export function mergeCitations(existing: string[], incoming: string[]): string[] {
  const merged = [...existing];
  for (const c of incoming) {
    if (c && !merged.includes(c)) merged.push(c);
  }
  return merged;
}

/**
 * 纯函数消息归约器：将单条已通过 seq 去重校验的事件应用到指定助手消息上，返回新消息快照。
 */
export function reduceChatEvent(
  msg: ChatMessage,
  event: ChatEvent,
  lastSeq: number,
  fromReplay = false,
): ChatMessage {
  const next: ChatMessage = {
    ...msg,
    turnId: event.turn_id,
    lastSeq,
    replayedCount: fromReplay ? (msg.replayedCount ?? 0) + 1 : msg.replayedCount,
  };

  switch (event.type) {
    case 'stage': {
      const data = event.data as unknown as StageEventData;
      next.stageCode = data.stage;
      next.stageLabel = formatStageLabel(data);
      if (data.evidence_ledger) {
        next.evidenceLedger = data.evidence_ledger;
        if (data.evidence_ledger.citation_details?.length) {
          next.citationDetails = mergeCitationDetails(
            next.citationDetails,
            data.evidence_ledger.citation_details,
          );
        }
      }
      if (data.stage === 'tool_running') {
        next.content = '';
      }
      if (data.stage === 'verifying' && data.passed === false) {
        next.content = '';
      }
      if (data.stage === 'fallback') {
        next.isFallback = true;
      }
      return next;
    }

    case 'tool_call': {
      const data = event.data as unknown as ToolCallEventData;
      next.tools = [...next.tools, data];
      return next;
    }

    case 'citation': {
      const data = event.data as unknown as CitationEventData;
      const freshCitations = Array.isArray(data.citations) ? data.citations : [];
      next.citations = mergeCitations(next.citations, freshCitations);
      const incomingDetails: CitationDetail[] =
        Array.isArray(data.citation_details) && data.citation_details.length > 0
          ? data.citation_details
          : freshCitations.map((src) => ({
              source: src,
              retrieval_source: 'bm25',
              score: 0,
            }));
      next.citationDetails = mergeCitationDetails(next.citationDetails, incomingDetails);
      return next;
    }

    case 'token': {
      const data = event.data as unknown as TokenEventData;
      if (data.fallback) {
        next.isFallback = true;
        next.content = data.delta ?? '';
      } else {
        next.content = `${next.content}${data.delta ?? ''}`;
      }
      return next;
    }

    case 'approval_required': {
      const data = event.data as unknown as ApprovalRequiredEventData;
      next.status = 'waiting_approval';
      next.stageCode = 'waiting_approval';
      next.stageLabel = '等待人工授权确认建单';
      next.content = '';
      next.approval = { data, state: 'pending' };
      return next;
    }

    case 'done': {
      const data = event.data as unknown as DoneEventData;
      next.status = 'completed';
      next.stageCode = 'completed';
      next.stageLabel = '回复已完成';
      if (typeof data.reply === 'string' && data.reply.length > 0) {
        next.content = data.reply;
      }
      if (Array.isArray(data.citations) && data.citations.length > 0) {
        next.citations = mergeCitations(next.citations, data.citations);
      }
      next.ttftMs = data.ttft_ms;
      next.latencyMs = data.latency_ms;
      return next;
    }

    case 'error': {
      const data = event.data as unknown as ErrorEventData;
      next.status = 'failed';
      next.stageCode = 'failed';
      next.stageLabel = '本轮处理异常';
      next.errorMessage = data.message || '后端返回错误事件';
      if (!next.content) {
        next.content = data.reply || data.message || '抱歉，处理请求时发生异常。';
      }
      next.latencyMs = data.latency_ms;
      return next;
    }

    default:
      return next;
  }
}

/**
 * 幂等消费入口：校验 meta.processedSeqs 防重，更新 meta 状态并返回归约后的消息。
 */
export function applyEventIdempotent(
  meta: TurnRuntimeMeta,
  msg: ChatMessage,
  event: ChatEvent,
  fromReplay = false,
): { applied: boolean; message: ChatMessage; completedData: DoneEventData | null } {
  if (!event.turn_id || meta.processedSeqs.has(event.seq)) {
    return { applied: false, message: msg, completedData: null };
  }

  meta.processedSeqs.add(event.seq);
  if (event.seq > meta.lastSeq) {
    meta.lastSeq = event.seq;
  }

  let completedData: DoneEventData | null = null;
  if (event.type === 'approval_required') {
    meta.waitingApproval = true;
  } else if (event.type === 'done') {
    meta.terminal = true;
    meta.waitingApproval = false;
    completedData = event.data as unknown as DoneEventData;
  } else if (event.type === 'error') {
    meta.terminal = true;
    meta.waitingApproval = false;
  }

  const updated = reduceChatEvent(msg, event, meta.lastSeq, fromReplay);
  return { applied: true, message: updated, completedData };
}
