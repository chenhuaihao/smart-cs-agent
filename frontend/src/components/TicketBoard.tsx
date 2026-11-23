/**
 * frontend/src/components/TicketBoard.tsx
 *
 * 售后工单看板：
 *   - 展示 GET /api/tickets 工单列表，支持按状态筛选与手动刷新；
 *   - 严格映射后端状态（新建工单默认为「待处理」），支持一键状态流转（待处理 → 处理中 → 已完成）；
 *   - 展示公网演示环境提示，重置演示数据提供二次确认并兼容可选 DEMO_ADMIN_TOKEN 口令校验。
 */

import { useCallback, useEffect, useState } from 'react';
import { ApiError, listTickets, resetDemoTickets, updateTicketStatus } from '../api/client';
import type { Ticket } from '../types/chat';

interface TicketBoardProps {
  /** 外部递增版本号（如对话完成建单后触发自动刷新） */
  refreshSignal: number;
  onInspectOrder: (orderId: string) => void;
  onDemoReset?: () => void;
}

const STATUS_OPTIONS = ['全部', '待处理', '处理中', '已完成'] as const;

// 严格单向状态机：待处理 → 处理中 / 已完成；处理中 → 已完成；已完成为终态不展示任何流转按钮
const NEXT_STATUS_TRANSITIONS: Record<string, string[]> = {
  待处理: ['处理中', '已完成'],
  处理中: ['已完成'],
  已完成: [],
  已结单: [],
  已关闭: [],
};

function getStatusClass(status: string): string {
  if (status === '待处理') return 'ticket-status ticket-status--pending';
  if (status === '处理中') return 'ticket-status ticket-status--processing';
  if (status === '已完成' || status === '已结单') return 'ticket-status ticket-status--completed';
  return 'ticket-status';
}

export function TicketBoard({ refreshSignal, onInspectOrder, onDemoReset }: TicketBoardProps) {
  const [tickets, setTickets] = useState<Ticket[]>([]);
  const [statusFilter, setStatusFilter] = useState<string>('全部');
  const [loading, setLoading] = useState<boolean>(false);
  const [updatingId, setUpdatingId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  // 重置演示数据二次确认与可选管理员口令状态
  const [confirmingReset, setConfirmingReset] = useState<boolean>(false);
  const [needAdminToken, setNeedAdminToken] = useState<boolean>(false);
  const [adminToken, setAdminToken] = useState<string>('');

  const loadTickets = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await listTickets(
        statusFilter === '全部' ? undefined : { status: statusFilter },
      );
      setTickets(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : '加载工单列表失败');
    } finally {
      setLoading(false);
    }
  }, [statusFilter]);

  useEffect(() => {
    void loadTickets();
  }, [loadTickets, refreshSignal]);

  const handleTransition = async (ticketId: string, nextStatus: string) => {
    setUpdatingId(ticketId);
    setError(null);
    setNotice(null);
    try {
      const updated = await updateTicketStatus(ticketId, nextStatus);
      setTickets((prev) =>
        prev
          .map((t) => (t.ticket_id === ticketId ? updated : t))
          .filter((t) => statusFilter === '全部' || t.status === statusFilter),
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : '更新工单状态失败');
    } finally {
      setUpdatingId(null);
    }
  };

  const handleConfirmReset = async () => {
    setLoading(true);
    setError(null);
    setNotice(null);
    try {
      const res = await resetDemoTickets(adminToken);
      setConfirmingReset(false);
      setNeedAdminToken(false);
      setAdminToken('');
      setNotice(res.message);
      onDemoReset?.();
      await loadTickets();
    } catch (err) {
      if (err instanceof ApiError && err.status === 403) {
        setNeedAdminToken(true);
      }
      setError(err instanceof Error ? err.message : '重置演示数据失败');
    } finally {
      setLoading(false);
    }
  };

  const counts = {
    total: tickets.length,
    pending: tickets.filter((t) => t.status === '待处理').length,
    processing: tickets.filter((t) => t.status === '处理中').length,
    completed: tickets.filter((t) => t.status === '已完成' || t.status === '已结单').length,
  };

  return (
    <section className="workspace-panel ticket-board">
      <header className="workspace-header">
        <div>
          <h2 className="workspace-title">售后工单看板</h2>
          <p className="workspace-subtitle">
            实时同步 SQLite 工单表 · 新建工单默认「待处理」，支持流转至「处理中 / 已完成」
          </p>
        </div>
        <div className="ticket-board__toolbar">
          <button
            type="button"
            className="btn btn-ghost btn-sm"
            onClick={() => void loadTickets()}
            disabled={loading}
          >
            {loading ? '刷新中...' : '刷新列表'}
          </button>
          <button
            type="button"
            className="btn btn-ghost btn-sm"
            onClick={() => {
              setConfirmingReset((v) => !v);
              setError(null);
              setNotice(null);
            }}
            disabled={loading}
          >
            {confirmingReset ? '取消重置' : '重置演示数据'}
          </button>
        </div>
      </header>

      <div className="demo-notice">
        公开演示环境：工单与订单数据可随时重置，状态流转仅作演示体验，请勿填写真实个人隐私信息。
      </div>

      {confirmingReset && (
        <div className="reset-confirm-bar">
          <div className="reset-confirm-bar__text">
            确认清空当前所有测试数据并重建 <strong>24 条标准订单 + 3 条演示工单</strong>？
          </div>
          <div className="reset-confirm-bar__controls">
            {needAdminToken && (
              <input
                type="password"
                className="text-input reset-confirm-bar__input"
                value={adminToken}
                onChange={(e) => setAdminToken(e.target.value)}
                placeholder="输入管理员口令 (DEMO_ADMIN_TOKEN)"
              />
            )}
            <button
              type="button"
              className="btn btn-danger-outline btn-sm"
              disabled={loading || (needAdminToken && !adminToken.trim())}
              onClick={() => void handleConfirmReset()}
            >
              {loading ? '重置中...' : '确认重置'}
            </button>
            <button
              type="button"
              className="btn btn-ghost btn-sm"
              disabled={loading}
              onClick={() => {
                setConfirmingReset(false);
                setNeedAdminToken(false);
                setAdminToken('');
              }}
            >
              取消
            </button>
          </div>
        </div>
      )}

      <div className="ticket-filters">
        <div className="filter-pills">
          {STATUS_OPTIONS.map((st) => (
            <button
              key={st}
              type="button"
              className={`filter-pill ${statusFilter === st ? 'filter-pill--active' : ''}`}
              onClick={() => setStatusFilter(st)}
            >
              {st}
            </button>
          ))}
        </div>
        <div className="ticket-summary-stats">
          <span>当前展示 {counts.total} 条</span>
          <span>待处理 {counts.pending}</span>
          <span>处理中 {counts.processing}</span>
          <span>已完成 {counts.completed}</span>
        </div>
      </div>

      {notice && <div className="alert-banner alert-banner--info">{notice}</div>}
      {error && <div className="alert-banner alert-banner--error">{error}</div>}

      <div className="ticket-list-scroll">
        {tickets.length === 0 ? (
          <div className="empty-inline">
            {loading ? '正在加载工单数据...' : '当前筛选条件下暂无工单记录'}
          </div>
        ) : (
          <div className="ticket-grid">
            {tickets.map((t) => {
              const displayStatus = (t.status || '待处理').trim();
              const actions = NEXT_STATUS_TRANSITIONS[displayStatus] ?? [];
              return (
                <article key={t.ticket_id} className="ticket-card">
                  <div className="ticket-card__top">
                    <div className="ticket-card__id-group">
                      <code className="ticket-card__id">{t.ticket_id}</code>
                      <span className="ticket-card__type">{t.issue_type}</span>
                    </div>
                    <span className={getStatusClass(displayStatus)}>{displayStatus}</span>
                  </div>

                  <div className="ticket-card__meta">
                    <button
                      type="button"
                      className="link-btn"
                      onClick={() => onInspectOrder(t.order_id)}
                    >
                      订单：{t.order_id}
                    </button>
                    <span>{t.created_at}</span>
                  </div>

                  {t.summary && <p className="ticket-card__summary">{t.summary}</p>}
                  {t.reason && <p className="ticket-card__reason">原因：{t.reason}</p>}

                  {actions.length > 0 && (
                    <div className="ticket-card__actions">
                      {actions.map((nextSt) => (
                        <button
                          key={nextSt}
                          type="button"
                          className="btn btn-ghost btn-xs"
                          disabled={updatingId === t.ticket_id}
                          onClick={() => void handleTransition(t.ticket_id, nextSt)}
                        >
                          {updatingId === t.ticket_id ? '更新中...' : `流转至「${nextSt}」`}
                        </button>
                      ))}
                    </div>
                  )}
                </article>
              );
            })}
          </div>
        )}
      </div>
    </section>
  );
}
