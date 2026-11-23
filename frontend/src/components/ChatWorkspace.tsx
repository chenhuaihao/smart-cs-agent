/**
 * frontend/src/components/ChatWorkspace.tsx
 *
 * 智能客服对话工作区：
 *   - 支持 SSE 逐字流式输出与实时阶段提示（思考 / 工具执行 / 账本更新 / 规则校验 / 降级）；
 *   - 展示工具调用步骤、证据账本摘要与 Hybrid RAG 引用徽章（bm25 / vector / hybrid + score）；
 *   - 在对话流中内联渲染 ApprovalCard（HITL 高风险建单确认）；
 *   - 支持展示 turn_id / last_seq 并在中断时手动触发 after_seq 补发同步。
 */

import { useEffect, useRef, useState } from 'react';
import type { FormEvent, KeyboardEvent } from 'react';
import type { ChatMessage, CitationDetail, RetrievalSource } from '../types/chat';
import { ApprovalCard } from './ApprovalCard';
import { FormattedText } from './FormattedText';

interface ChatWorkspaceProps {
  sessionId: string;
  messages: ChatMessage[];
  isStreaming: boolean;
  isReplaying: boolean;
  activeTurnId: string | null;
  lastSeq: number;
  streamError: string | null;
  onSend: (text: string) => void;
  onDecideApproval: (turnId: string, approved: boolean) => void;
  onSyncTurn: (turnId: string) => void;
  onAbortStream: () => void;
  onResetSession: () => void;
  onInspectOrder: (orderId: string) => void;
}

const QUICK_PROMPTS = [
  { label: '口语化语义检索', text: '蓝牙耳机戴着一边完全没动静听不见啦咋弄？' },
  { label: '拆封退货政策', text: '耳机拆封后没有质量问题还能 7 天无理由退货吗？' },
  { label: '查询订单详情', text: '帮我查一下订单 ORD20261101 的当前状态和运单号' },
  { label: '触发 HITL 建单', text: '订单 ORD20261121 全新未拆封，请帮我创建一个退货工单，原因：不想要了' },
];

const SOURCE_BADGE_MAP: Record<RetrievalSource, { label: string; className: string }> = {
  hybrid: { label: '双路融合 hybrid', className: 'badge badge--hybrid' },
  vector: { label: '语义召回 vector', className: 'badge badge--vector' },
  bm25: { label: '关键词 bm25', className: 'badge badge--bm25' },
};

function renderCitationBadge(detail: CitationDetail) {
  const meta = SOURCE_BADGE_MAP[detail.retrieval_source] ?? SOURCE_BADGE_MAP.bm25;
  return (
    <div key={detail.source} className="citation-chip">
      <span className="citation-chip__source">{detail.source}</span>
      <span className={meta.className}>{meta.label}</span>
      {detail.score > 0 && (
        <span className="citation-chip__score">
          {detail.retrieval_source === 'bm25' && detail.score >= 1
            ? detail.score.toFixed(2)
            : detail.score.toFixed(4)}
        </span>
      )}
    </div>
  );
}

export function ChatWorkspace({
  sessionId,
  messages,
  isStreaming,
  isReplaying,
  activeTurnId,
  lastSeq,
  streamError,
  onSend,
  onDecideApproval,
  onSyncTurn,
  onAbortStream,
  onResetSession,
  onInspectOrder,
}: ChatWorkspaceProps) {
  const [input, setInput] = useState('');
  const scrollAreaRef = useRef<HTMLDivElement | null>(null);
  const endRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  const lastMessage = messages[messages.length - 1];
  const scrollTriggerKey = lastMessage
    ? [
        messages.length,
        lastMessage.content.length,
        lastMessage.tools.length,
        lastMessage.citationDetails.length,
        lastMessage.approval?.state ?? '',
        lastMessage.stageCode ?? '',
      ].join(':')
    : '0';

  // 仅在存在对话消息时，于新增 token、tool_call、引用或弹出 ApprovalCard 后自动平滑滚动到底部；
  // 空欢迎页保持顶部居中，避免首次打开页面被错误滚到底部裁切顶部内容。
  useEffect(() => {
    const container = scrollAreaRef.current;
    if (messages.length === 0) {
      if (container) container.scrollTop = 0;
      return;
    }
    const rafId = window.requestAnimationFrame(() => {
      if (container) {
        container.scrollTo({
          top: container.scrollHeight,
          behavior: 'smooth',
        });
      } else {
        endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
      }
    });
    return () => window.cancelAnimationFrame(rafId);
  }, [scrollTriggerKey, messages.length]);

  // 输入框紧凑自适应高度（默认单行，随输入内容微扩至最多约 4 行）
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, 108)}px`;
  }, [input]);

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isStreaming) return;
    onSend(input);
    setInput('');
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!input.trim() || isStreaming) return;
      onSend(input);
      setInput('');
    }
  };

  return (
    <section className="workspace-panel chat-workspace">
      <header className="workspace-header">
        <div>
          <h2 className="workspace-title">智能售后对话工作台</h2>
          <p className="workspace-subtitle">
            SSE 流式推理 · 证据账本护栏 · Hybrid RAG 溯源 · HITL 高风险建单确认
          </p>
        </div>
        <div className="chat-meta-bar">
          <span className="meta-pill" title="当前会话 ID（保留最近 6 轮上下文）">
            Session: <code>{sessionId}</code>
          </span>
          {activeTurnId && (
            <span className="meta-pill" title="当前轮次与已消费最大 seq">
              Turn: <code>{activeTurnId.slice(0, 8)}</code> · seq={lastSeq}
            </span>
          )}
          <button
            type="button"
            className="btn btn-ghost btn-sm"
            onClick={onResetSession}
            disabled={isStreaming}
          >
            新建会话
          </button>
        </div>
      </header>

      {streamError && (
        <div className="alert-banner alert-banner--error">
          <span>{streamError}</span>
          {activeTurnId && (
            <button
              type="button"
              className="btn btn-ghost btn-sm"
              onClick={() => onSyncTurn(activeTurnId)}
              disabled={isReplaying}
            >
              {isReplaying ? '补发中...' : '断线补发同步'}
            </button>
          )}
        </div>
      )}

      <div
        className={`chat-scroll-area ${messages.length === 0 ? 'chat-scroll-area--empty' : ''}`}
        ref={scrollAreaRef}
      >
        {messages.length === 0 ? (
          <div className="empty-state">
            <div className="empty-state__head">
              <div className="empty-state__icon">AI</div>
              <h3>欢迎使用 Smart CS Agent v2.0 控制台</h3>
            </div>
            <p>
              点击下方快捷场景，体验 SSE 实时流式输出、Hybrid RAG（BM25 / 向量 / RRF）溯源标签与 HITL 建单确认。
            </p>
            <div className="quick-prompts">
              {QUICK_PROMPTS.map((item) => (
                <button
                  key={item.label}
                  type="button"
                  className="quick-prompt-card"
                  onClick={() => onSend(item.text)}
                  disabled={isStreaming}
                >
                  <span className="quick-prompt-card__label">{item.label}</span>
                  <span className="quick-prompt-card__text">{item.text}</span>
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="message-list">
            {messages.map((msg) => {
              if (msg.role === 'user') {
                return (
                  <article key={msg.id} className="message-row message-row--user">
                    <div className="message-bubble message-bubble--user">
                      <div className="message-bubble__meta">
                        <span>客户</span>
                        <span>{msg.createdAt}</span>
                      </div>
                      <FormattedText text={msg.content} />
                    </div>
                  </article>
                );
              }

              const isRunning = msg.status === 'running';
              const orderKeys = Object.keys(msg.evidenceLedger?.orders ?? {});

              return (
                <article key={msg.id} className="message-row message-row--assistant">
                  <div className="message-bubble message-bubble--assistant">
                    <div className="message-bubble__meta">
                      <div className="message-bubble__title">
                        <strong>小智 Agent</strong>
                        {msg.stageLabel && (
                          <span
                            className={`stage-pill ${
                              msg.isFallback
                                ? 'stage-pill--warn'
                                : msg.status === 'waiting_approval'
                                  ? 'stage-pill--approval'
                                  : isRunning
                                    ? 'stage-pill--active'
                                    : 'stage-pill--done'
                            }`}
                          >
                            {isRunning && <span className="pulse-dot" />}
                            {msg.stageLabel}
                          </span>
                        )}
                      </div>
                      <div className="message-bubble__stats">
                        {msg.turnId && (
                          <span className="stat-tag" title={`turn_id=${msg.turnId}`}>
                            seq={msg.lastSeq ?? 0}
                          </span>
                        )}
                        {(msg.replayedCount ?? 0) > 0 && (
                          <span className="stat-tag stat-tag--info">
                            已补发 {msg.replayedCount} 条
                          </span>
                        )}
                        {typeof msg.ttftMs === 'number' && (
                          <span className="stat-tag">TTFT {msg.ttftMs}ms</span>
                        )}
                        {typeof msg.latencyMs === 'number' && (
                          <span className="stat-tag">耗时 {msg.latencyMs}ms</span>
                        )}
                        {msg.turnId && msg.status !== 'completed' && msg.status !== 'failed' && (
                          <button
                            type="button"
                            className="link-btn"
                            onClick={() => onSyncTurn(msg.turnId!)}
                            disabled={isReplaying}
                          >
                            补发同步
                          </button>
                        )}
                      </div>
                    </div>

                    {/* 工具执行步骤列表 */}
                    {msg.tools.length > 0 && (
                      <div className="tool-steps">
                        {msg.tools.map((tool, idx) => (
                          <div key={`${tool.call_id}-${idx}`} className="tool-step">
                            <span
                              className={`tool-step__dot tool-step__dot--${tool.status}`}
                              title={tool.status}
                            />
                            <code className="tool-step__name">{tool.name}</code>
                            <span className="tool-step__status">
                              {tool.status === 'success'
                                ? '成功'
                                : tool.status === 'cancelled'
                                  ? '已取消'
                                  : '失败'}
                            </span>
                            <span className="tool-step__duration">{tool.duration_ms}ms</span>
                            <span className="tool-step__args">
                              {typeof tool.arguments === 'string'
                                ? tool.arguments
                                : JSON.stringify(tool.arguments)}
                            </span>
                          </div>
                        ))}
                      </div>
                    )}

                    {/* 正文流式内容 */}
                    {msg.content ? (
                      <FormattedText text={msg.content} />
                    ) : isRunning ? (
                      <p className="message-placeholder">正在生成回复...</p>
                    ) : null}

                    {/* HITL 高风险建单授权卡片 */}
                    {msg.approval && (
                      <ApprovalCard
                        approval={msg.approval}
                        disabled={isStreaming && msg.approval.state !== 'pending'}
                        onDecide={onDecideApproval}
                      />
                    )}

                    {/* 知识库溯源与检索方式徽章 */}
                    {msg.citationDetails.length > 0 && (
                      <div className="citations-block">
                        <div className="citations-block__label">知识库引用与检索来源</div>
                        <div className="citations-list">
                          {msg.citationDetails.map(renderCitationBadge)}
                        </div>
                      </div>
                    )}

                    {/* 证据账本中的核验订单快捷跳转 */}
                    {orderKeys.length > 0 && (
                      <div className="ledger-orders">
                        <span className="ledger-orders__label">已核验订单证据：</span>
                        {orderKeys.map((oid) => (
                          <button
                            key={oid}
                            type="button"
                            className="order-link-chip"
                            onClick={() => onInspectOrder(oid)}
                          >
                            {oid}（{msg.evidenceLedger?.orders[oid]?.status || '已核验'}）
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                </article>
              );
            })}
            <div ref={endRef} className="chat-bottom-spacer" />
          </div>
        )}
      </div>

      <form className="chat-composer" onSubmit={handleSubmit}>
        <textarea
          ref={textareaRef}
          className="chat-composer__input"
          rows={1}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="输入售后咨询、订单号或建单诉求（Enter 发送，Shift+Enter 换行）..."
          disabled={isStreaming}
        />
        <div className="chat-composer__actions">
          {isStreaming ? (
            <button type="button" className="btn btn-danger-outline" onClick={onAbortStream}>
              断开流（测试补发）
            </button>
          ) : (
            <button type="submit" className="btn btn-primary" disabled={!input.trim()}>
              发送消息
            </button>
          )}
        </div>
      </form>
    </section>
  );
}
