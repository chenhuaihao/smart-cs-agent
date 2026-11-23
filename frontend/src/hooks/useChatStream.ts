/**
 * frontend/src/hooks/useChatStream.ts
 *
 * 智能客服流式状态机 Hook：
 *   - 维护当前 turnId、lastSeq 与每个 turn 的 processedSeqs 集合，确保按 seq 严格幂等去重；
 *   - 统一消费 SSE 实时流与 GET /api/chat/turns/{turn_id}/events?after_seq=... 补发事件；
 *   - 当流意外中断或未收到终态即断开时，自动发起增量补发直到 done / error / waiting_approval；
 *   - 支持 HITL 人工授权（approveTurnStream）将续流无缝接回当前轮次。
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import {
  ApiError,
  approveTurnStream,
  getTurnEvents,
  streamChat,
} from '../api/client';
import type {
  ApprovalRequiredEventData,
  ChatEvent,
  ChatMessage,
  CitationDetail,
  CitationEventData,
  DoneEventData,
  ErrorEventData,
  StageEventData,
  TokenEventData,
  ToolCallEventData,
} from '../types/chat';

interface TurnRuntimeMeta {
  messageId: string;
  processedSeqs: Set<number>;
  lastSeq: number;
  terminal: boolean;
  waitingApproval: boolean;
}

interface UseChatStreamOptions {
  /** 当一轮对话完成时回调（例如检测调用了建单工具后自动刷新工单看板） */
  onTurnComplete?: (done: DoneEventData) => void;
}

function createSessionId(): string {
  const rand = Math.random().toString(16).slice(2, 8);
  return `web-${Date.now().toString(36)}-${rand}`;
}

function nowTimeLabel(): string {
  return new Date().toLocaleTimeString('zh-CN', { hour12: false });
}

function formatStageLabel(data: StageEventData): string {
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
    default:
      return `处理阶段：${data.stage}`;
  }
}

function mergeCitationDetails(
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

function mergeCitations(existing: string[], incoming: string[]): string[] {
  const merged = [...existing];
  for (const c of incoming) {
    if (c && !merged.includes(c)) merged.push(c);
  }
  return merged;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export function useChatStream(options: UseChatStreamOptions = {}) {
  const [sessionId, setSessionId] = useState<string>(() => createSessionId());
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [isReplaying, setIsReplaying] = useState<boolean>(false);
  const [activeTurnId, setActiveTurnId] = useState<string | null>(null);
  const [lastSeq, setLastSeq] = useState<number>(0);
  const [streamError, setStreamError] = useState<string | null>(null);

  const turnMetaRef = useRef<Map<string, TurnRuntimeMeta>>(new Map());
  const abortRef = useRef<AbortController | null>(null);
  const onTurnCompleteRef = useRef(options.onTurnComplete);
  onTurnCompleteRef.current = options.onTurnComplete;

  useEffect(() => {
    return () => {
      abortRef.current?.abort();
    };
  }, []);

  const ensureTurnMeta = useCallback((turnId: string, messageId: string): TurnRuntimeMeta => {
    let meta = turnMetaRef.current.get(turnId);
    if (!meta) {
      meta = {
        messageId,
        processedSeqs: new Set<number>(),
        lastSeq: 0,
        terminal: false,
        waitingApproval: false,
      };
      turnMetaRef.current.set(turnId, meta);
    }
    return meta;
  }, []);

  /**
   * 幂等事件消费器：所有实时 SSE 事件与 after_seq 补发事件均经过此函数，
   * 严格按 turnId + seq 去重，保证已处理过的 seq 绝不重复拼接或重复触发副作用。
   */
  const applyEvent = useCallback(
    (messageId: string, event: ChatEvent, fromReplay = false) => {
      const turnId = event.turn_id;
      if (!turnId) return;

      const meta = ensureTurnMeta(turnId, messageId);
      if (meta.processedSeqs.has(event.seq)) {
        return;
      }
      meta.processedSeqs.add(event.seq);
      if (event.seq > meta.lastSeq) {
        meta.lastSeq = event.seq;
      }

      setActiveTurnId(turnId);
      setLastSeq(meta.lastSeq);

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

      setMessages((prev) =>
        prev.map((msg) => {
          if (msg.id !== meta.messageId) return msg;

          const next: ChatMessage = {
            ...msg,
            turnId,
            lastSeq: meta.lastSeq,
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
              if (data.stage === 'verifying' && data.passed === false) {
                // 校验未通过准备重写时清空上一轮未通过草稿，避免两份回答拼在一起
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
        }),
      );

      if (completedData) {
        onTurnCompleteRef.current?.(completedData);
      }
    },
    [ensureTurnMeta],
  );

  /**
   * 断线重连补发循环：按 after_seq={lastSeq} 增量拉取未消费事件，
   * 经过 applyEvent 幂等去重直至 turn 达到 completed / failed / waiting_approval。
   */
  const replayTurn = useCallback(
    async (turnId: string, messageId: string): Promise<void> => {
      const meta = ensureTurnMeta(turnId, messageId);
      if (meta.terminal || meta.waitingApproval) return;

      setIsReplaying(true);
      setMessages((prev) =>
        prev.map((m) =>
          m.id === messageId
            ? { ...m, stageLabel: `连接中断，正在从 seq=${meta.lastSeq} 补发事件...` }
            : m,
        ),
      );

      const maxPolls = 25;
      try {
        for (let attempt = 0; attempt < maxPolls; attempt += 1) {
          const resp = await getTurnEvents(turnId, meta.lastSeq);
          for (const ev of resp.events) {
            applyEvent(messageId, ev, true);
          }

          if (
            meta.terminal ||
            meta.waitingApproval ||
            resp.status === 'completed' ||
            resp.status === 'failed' ||
            resp.status === 'waiting_approval'
          ) {
            if (resp.status === 'completed' && !meta.terminal) {
              meta.terminal = true;
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === messageId ? { ...m, status: 'completed', stageLabel: '补发完成' } : m,
                ),
              );
            }
            return;
          }

          // 后端 turn 仍在运行中，短暂退避后继续拉取增量事件
          await sleep(Math.min(400 + attempt * 150, 1200));
        }

        setStreamError(`turn ${turnId} 补发超时，可点击“补发同步”手动重试。`);
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        setStreamError(`断线补发失败：${msg}`);
      } finally {
        setIsReplaying(false);
      }
    },
    [applyEvent, ensureTurnMeta],
  );

  /** 发送用户消息并开启一轮 SSE 流式对话 */
  const sendMessage = useCallback(
    async (rawText: string): Promise<void> => {
      const text = rawText.trim();
      if (!text || isStreaming) return;

      setStreamError(null);
      const userMsgId = `u-${Date.now()}-${Math.random().toString(16).slice(2, 6)}`;
      const assistantMsgId = `a-${Date.now()}-${Math.random().toString(16).slice(2, 6)}`;
      const createdAt = nowTimeLabel();

      const userMsg: ChatMessage = {
        id: userMsgId,
        role: 'user',
        content: text,
        createdAt,
        tools: [],
        citations: [],
        citationDetails: [],
      };
      const assistantMsg: ChatMessage = {
        id: assistantMsgId,
        role: 'assistant',
        content: '',
        createdAt,
        status: 'running',
        stageCode: 'connecting',
        stageLabel: '正在连接智能客服流...',
        lastSeq: 0,
        tools: [],
        citations: [],
        citationDetails: [],
      };

      setMessages((prev) => [...prev, userMsg, assistantMsg]);
      setIsStreaming(true);

      const controller = new AbortController();
      abortRef.current = controller;

      let resolvedTurnId: string | null = null;
      const bindTurnId = (tid: string) => {
        if (!tid) return;
        resolvedTurnId = tid;
        ensureTurnMeta(tid, assistantMsgId);
        setActiveTurnId(tid);
        setMessages((prev) =>
          prev.map((m) => (m.id === assistantMsgId ? { ...m, turnId: tid } : m)),
        );
      };

      try {
        await streamChat(
          sessionId,
          text,
          {
            onTurnId: bindTurnId,
            onEvent: (ev) => {
              if (!resolvedTurnId && ev.turn_id) {
                bindTurnId(ev.turn_id);
              }
              applyEvent(assistantMsgId, ev, false);
            },
          },
          controller.signal,
        );

        // SSE 流已关闭但尚未收到 done / error / approval_required，触发断线补发兜底
        if (resolvedTurnId) {
          const meta = turnMetaRef.current.get(resolvedTurnId);
          if (meta && !meta.terminal && !meta.waitingApproval) {
            await replayTurn(resolvedTurnId, assistantMsgId);
          }
        }
      } catch (err) {
        if (controller.signal.aborted) {
          // 主动中止时若已有 turnId，保留状态允许用户手动补发
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMsgId ? { ...m, stageLabel: '已断开实时流（可点击补发同步）' } : m,
            ),
          );
          return;
        }

        // 若已拿到 turn_id（说明后端已落库该轮），自动走 after_seq 断线补发
        if (resolvedTurnId) {
          await replayTurn(resolvedTurnId, assistantMsgId);
        } else {
          const detail =
            err instanceof ApiError
              ? err.detail
              : err instanceof Error
                ? err.message
                : '网络连接失败，请确认后端 127.0.0.1:8000 已启动';
          setStreamError(detail);
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMsgId
                ? {
                    ...m,
                    status: 'failed',
                    stageLabel: '请求失败',
                    errorMessage: detail,
                    content: `请求失败：${detail}`,
                  }
                : m,
            ),
          );
        }
      } finally {
        if (abortRef.current === controller) {
          abortRef.current = null;
        }
        setIsStreaming(false);
      }
    },
    [applyEvent, ensureTurnMeta, isStreaming, replayTurn, sessionId],
  );

  /** 提交高风险写操作审批决策，并将 /approve 返回的 SSE 续流接回当前轮次 */
  const decideApproval = useCallback(
    async (turnId: string, approved: boolean): Promise<void> => {
      const targetMsg = messages.find((m) => m.turnId === turnId);
      if (!targetMsg) return;

      const meta = ensureTurnMeta(turnId, targetMsg.id);
      meta.waitingApproval = false;

      setStreamError(null);
      setIsStreaming(true);
      setMessages((prev) =>
        prev.map((m) =>
          m.id === targetMsg.id
            ? {
                ...m,
                status: 'running',
                stageLabel: approved ? '已授权建单，正在执行落库...' : '已拒绝建单，正在通知 Agent...',
                approval: m.approval ? { ...m.approval, state: 'submitting', error: undefined } : undefined,
              }
            : m,
        ),
      );

      const controller = new AbortController();
      abortRef.current = controller;

      try {
        await approveTurnStream(
          turnId,
          approved,
          {
            onEvent: (ev) => applyEvent(targetMsg.id, ev, false),
          },
          controller.signal,
        );

        setMessages((prev) =>
          prev.map((m) =>
            m.id === targetMsg.id && m.approval
              ? {
                  ...m,
                  approval: { ...m.approval, state: approved ? 'approved' : 'rejected' },
                }
              : m,
          ),
        );

        // 若续流提前关闭但尚未收到 done，走 after_seq 补发
        if (!meta.terminal && !meta.waitingApproval) {
          await replayTurn(turnId, targetMsg.id);
        }
      } catch (err) {
        if (controller.signal.aborted) return;

        // 409/404 等业务错误直接回显；网络中断则走补发
        if (err instanceof ApiError && err.status >= 400 && err.status < 500) {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === targetMsg.id && m.approval
                ? {
                    ...m,
                    status: 'waiting_approval',
                    approval: { ...m.approval, state: 'pending', error: err.detail },
                  }
                : m,
            ),
          );
          setStreamError(err.detail);
        } else {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === targetMsg.id && m.approval
                ? {
                    ...m,
                    approval: { ...m.approval, state: approved ? 'approved' : 'rejected' },
                  }
                : m,
            ),
          );
          await replayTurn(turnId, targetMsg.id);
        }
      } finally {
        if (abortRef.current === controller) {
          abortRef.current = null;
        }
        setIsStreaming(false);
      }
    },
    [applyEvent, ensureTurnMeta, messages, replayTurn],
  );

  /** 手动触发某轮的 after_seq 断线补发同步 */
  const syncTurn = useCallback(
    async (turnId: string): Promise<void> => {
      const targetMsg = messages.find((m) => m.turnId === turnId);
      if (!targetMsg) return;
      const meta = ensureTurnMeta(turnId, targetMsg.id);
      meta.terminal = false;
      meta.waitingApproval = false;
      await replayTurn(turnId, targetMsg.id);
    },
    [ensureTurnMeta, messages, replayTurn],
  );

  /** 模拟/主动断开当前 SSE 流（可用于演示断线后手动或自动 after_seq 补发） */
  const abortStream = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  /** 重置会话（生成新 session_id 并清空当前对话流） */
  const resetSession = useCallback(() => {
    abortRef.current?.abort();
    turnMetaRef.current.clear();
    setSessionId(createSessionId());
    setMessages([]);
    setActiveTurnId(null);
    setLastSeq(0);
    setStreamError(null);
    setIsStreaming(false);
    setIsReplaying(false);
  }, []);

  return {
    sessionId,
    messages,
    isStreaming,
    isReplaying,
    activeTurnId,
    lastSeq,
    streamError,
    sendMessage,
    decideApproval,
    syncTurn,
    abortStream,
    resetSession,
  };
}
