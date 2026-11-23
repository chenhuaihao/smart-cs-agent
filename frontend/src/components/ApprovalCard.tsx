/**
 * frontend/src/components/ApprovalCard.tsx
 *
 * 高风险写操作（如 create_after_sales_ticket）HITL 人工授权确认卡片：
 *   - 展示工具名称、待确认摘要与结构化参数（order_id / issue_type / reason / summary）；
 *   - 提供“确认建单”与“取消建单”按钮，点击后调用 POST /api/chat/turns/{turn_id}/approve。
 */

import type { ApprovalState } from '../types/chat';

interface ApprovalCardProps {
  approval: ApprovalState;
  disabled?: boolean;
  onDecide: (turnId: string, approved: boolean) => void;
}

const ARG_LABELS: Record<string, string> = {
  order_id: '关联订单号',
  issue_type: '工单类型',
  reason: '申请原因',
  summary: '诉求摘要',
};

export function ApprovalCard({ approval, disabled, onDecide }: ApprovalCardProps) {
  const { data, state, error } = approval;
  const entries = Object.entries(data.args ?? {});
  const isActionable = state === 'pending' && !disabled;

  return (
    <div className={`approval-card approval-card--${state}`} role="region" aria-label="高风险操作人工确认">
      <div className="approval-card__header">
        <div className="approval-card__title-wrap">
          <span className="approval-card__badge">HITL 高风险写入拦截</span>
          <code className="approval-card__tool">{data.tool}</code>
        </div>
        <span className={`approval-card__status approval-card__status--${state}`}>
          {state === 'pending' && '待人工确认'}
          {state === 'submitting' && '提交决策中...'}
          {state === 'approved' && '已授权落库'}
          {state === 'rejected' && '已取消建单'}
        </span>
      </div>

      <p className="approval-card__summary">{data.summary}</p>

      {entries.length > 0 && (
        <dl className="approval-card__args">
          {entries.map(([key, value]) => (
            <div key={key} className="approval-card__arg-row">
              <dt>{ARG_LABELS[key] ?? key}</dt>
              <dd>{typeof value === 'string' ? value : JSON.stringify(value)}</dd>
            </div>
          ))}
        </dl>
      )}

      {error && <div className="approval-card__error">{error}</div>}

      <div className="approval-card__actions">
        {state === 'pending' || state === 'submitting' ? (
          <>
            <button
              type="button"
              className="btn btn-primary"
              disabled={!isActionable}
              onClick={() => onDecide(data.turn_id, true)}
            >
              {state === 'submitting' ? '执行中...' : '确认建单'}
            </button>
            <button
              type="button"
              className="btn btn-danger-outline"
              disabled={!isActionable}
              onClick={() => onDecide(data.turn_id, false)}
            >
              取消建单
            </button>
          </>
        ) : (
          <span className="approval-card__resolved-note">
            {state === 'approved'
              ? '已向后端发送 approved=true，工单落库流程已续传完成。'
              : '已向后端发送 approved=false，本次未写入数据库。'}
          </span>
        )}
      </div>
    </div>
  );
}
