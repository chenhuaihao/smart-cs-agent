/**
 * frontend/src/components/InspectorDrawer.tsx
 *
 * 订单与知识库联调面板（右侧固定栏/抽屉）：
 *   1. 订单联调：按订单号精准查询（GET /api/orders/{order_id}）或快速点击全部种子订单查看状态与运单号；
 *   2. 知识库 Hybrid RAG 调试：调用 GET /api/knowledge/search，支持动态调节 top_k 与 min_score，
 *      直观展示每条命中的 retrieval_source（bm25 / vector / hybrid）与 score。
 */

import { useCallback, useEffect, useState } from 'react';
import type { FormEvent } from 'react';
import { getOrderById, listOrders, searchKnowledge } from '../api/client';
import type { KnowledgeSearchResponse, Order } from '../types/chat';
import { FormattedText } from './FormattedText';

interface InspectorDrawerProps {
  /** 从对话流或工单看板点击订单号时传入，自动触发订单精准查询 */
  selectedOrderId: string;
  onClose?: () => void;
}

type InspectorTab = 'knowledge' | 'orders';

export function InspectorDrawer({ selectedOrderId, onClose }: InspectorDrawerProps) {
  const [activeTab, setActiveTab] = useState<InspectorTab>('knowledge');

  // --- 订单联调状态 ---
  const [orderQuery, setOrderQuery] = useState<string>('');
  const [orderDetail, setOrderDetail] = useState<Order | null>(null);
  const [orders, setOrders] = useState<Order[]>([]);
  const [orderLoading, setOrderLoading] = useState<boolean>(false);
  const [orderError, setOrderError] = useState<string | null>(null);

  // --- Hybrid RAG 知识库联调状态（默认开启 Pre-RRF 门控，默认 RRF 最小阈值 0.016） ---
  const [kbQuery, setKbQuery] = useState<string>('耳机拆封后还能退货吗');
  const [topK, setTopK] = useState<number>(3);
  const [enablePreRrfGate, setEnablePreRrfGate] = useState<boolean>(true);
  const [rrfMinScore, setRrfMinScore] = useState<number>(0.016);
  const [kbResult, setKbResult] = useState<KnowledgeSearchResponse | null>(null);
  const [kbLoading, setKbLoading] = useState<boolean>(false);
  const [kbError, setKbError] = useState<string | null>(null);

  const loadAllOrders = useCallback(async () => {
    try {
      const data = await listOrders();
      setOrders(data);
    } catch {
      // 初始拉取静默，不阻断面板使用
    }
  }, []);

  useEffect(() => {
    void loadAllOrders();
  }, [loadAllOrders]);

  const queryOrder = useCallback(async (oid: string) => {
    const clean = oid.trim();
    if (!clean) return;
    setOrderLoading(true);
    setOrderError(null);
    try {
      const data = await getOrderById(clean);
      setOrderDetail(data);
    } catch (err) {
      setOrderDetail(null);
      setOrderError(err instanceof Error ? err.message : '订单查询失败');
    } finally {
      setOrderLoading(false);
    }
  }, []);

  useEffect(() => {
    if (selectedOrderId) {
      setActiveTab('orders');
      setOrderQuery(selectedOrderId);
      void queryOrder(selectedOrderId);
    }
  }, [selectedOrderId, queryOrder]);

  const handleOrderSubmit = (e: FormEvent) => {
    e.preventDefault();
    void queryOrder(orderQuery);
  };

  const handleKbSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!kbQuery.trim()) return;
    setKbLoading(true);
    setKbError(null);
    try {
      const data = await searchKnowledge(kbQuery, topK, {
        minScore: 3.0,
        enablePreRrfGate,
        rrfMinScore,
      });
      setKbResult(data);
    } catch (err) {
      setKbResult(null);
      setKbError(err instanceof Error ? err.message : '知识库检索失败');
    } finally {
      setKbLoading(false);
    }
  };

  return (
    <aside className="inspector-drawer" aria-label="订单与知识库联调面板">
      <div className="inspector-header">
        <div className="inspector-tabs">
          <button
            type="button"
            className={`inspector-tab ${activeTab === 'knowledge' ? 'inspector-tab--active' : ''}`}
            onClick={() => setActiveTab('knowledge')}
          >
            Hybrid RAG 调试
          </button>
          <button
            type="button"
            className={`inspector-tab ${activeTab === 'orders' ? 'inspector-tab--active' : ''}`}
            onClick={() => setActiveTab('orders')}
          >
            订单状态查询
          </button>
        </div>
        {onClose && (
          <button
            type="button"
            className="inspector-close-btn"
            onClick={onClose}
            title="收起联调面板"
            aria-label="收起联调面板"
          >
            ×
          </button>
        )}
      </div>

      {activeTab === 'knowledge' ? (
        <div className="inspector-body">
          <form className="inspector-form" onSubmit={(e) => void handleKbSubmit(e)}>
            <label className="field-label" htmlFor="kb-query-input">
              检索问题（支持书面条款与口语化查询）
            </label>
            <input
              id="kb-query-input"
              className="text-input"
              type="text"
              value={kbQuery}
              onChange={(e) => setKbQuery(e.target.value)}
              placeholder="例如：蓝牙耳机单侧无声排查步骤"
            />

            <label className="gate-toggle-row">
              <input
                type="checkbox"
                checked={enablePreRrfGate}
                onChange={(e) => setEnablePreRrfGate(e.target.checked)}
              />
              <div>
                <div className="gate-toggle-row__title">Pre-RRF 相关性门控（推荐开启）</div>
                <div className="gate-toggle-row__desc">
                  融合前过滤弱相关 BM25/向量候选，并在双路命中时裁剪低分单路尾部
                </div>
              </div>
            </label>

            <div className="param-row">
              <label className="param-field">
                <span>top_k: <strong>{topK}</strong></span>
                <input
                  type="range"
                  min={1}
                  max={10}
                  step={1}
                  value={topK}
                  onChange={(e) => setTopK(Number(e.target.value))}
                />
              </label>
              <label className="param-field">
                <span>RRF 最小阈值: <strong>{rrfMinScore.toFixed(3)}</strong></span>
                <input
                  type="range"
                  min={0}
                  max={0.04}
                  step={0.001}
                  value={rrfMinScore}
                  onChange={(e) => setRrfMinScore(Number(e.target.value))}
                />
              </label>
            </div>

            <div className="threshold-presets" role="group" aria-label="RRF 阈值快捷模式">
              <button
                type="button"
                className={`preset-pill ${Math.abs(rrfMinScore - 0.016) < 1e-6 ? 'preset-pill--active' : ''}`}
                onClick={() => setRrfMinScore(0.016)}
              >
                推荐均衡（0.016）
              </button>
              <button
                type="button"
                className={`preset-pill ${Math.abs(rrfMinScore - 0.025) < 1e-6 ? 'preset-pill--active' : ''}`}
                onClick={() => setRrfMinScore(0.025)}
              >
                仅双路高分（0.025）
              </button>
              <button
                type="button"
                className={`preset-pill ${rrfMinScore === 0 ? 'preset-pill--active' : ''}`}
                onClick={() => setRrfMinScore(0)}
              >
                不设下限
              </button>
            </div>

            <div className="param-hint">
              Hybrid 模式下按 RRF 分数过滤（双路约 0.031~0.033，单路约 0.016）；纯 BM25 降级模式自动沿用 BM25 ≥ 3.0 门控。
            </div>

            <button type="submit" className="btn btn-primary btn-full" disabled={kbLoading || !kbQuery.trim()}>
              {kbLoading ? '检索中...' : '执行混合检索'}
            </button>
          </form>

          {kbError && <div className="alert-banner alert-banner--error">{kbError}</div>}

          {kbResult && (
            <div className="kb-results">
              <div className="kb-results__summary">
                命中 <strong>{kbResult.total_hits}</strong> 条（top_k={kbResult.top_k}，门控={enablePreRrfGate ? '开启' : '关闭'}，RRF ≥ {rrfMinScore.toFixed(3)}）
              </div>
              {kbResult.results.length === 0 ? (
                <div className="empty-inline">当前门控/阈值下未命中知识库条款，可尝试调低 RRF 阈值。</div>
              ) : (
                kbResult.results.map((hit, idx) => (
                  <article key={`${hit.source}-${hit.section}-${idx}`} className="kb-hit-card">
                    <div className="kb-hit-card__header">
                      <span className={`badge badge--${hit.retrieval_source}`}>
                        {hit.retrieval_source}
                      </span>
                      <span className="kb-hit-card__score">
                        score: {hit.score >= 1 ? hit.score.toFixed(2) : hit.score.toFixed(4)}
                      </span>
                    </div>
                    <div className="kb-hit-card__source">
                      {hit.source}#{hit.section}
                    </div>
                    <div className="kb-hit-card__content">
                      <FormattedText text={hit.content} compact />
                    </div>
                  </article>
                ))
              )}
            </div>
          )}
        </div>
      ) : (
        <div className="inspector-body">
          <form className="inspector-form" onSubmit={handleOrderSubmit}>
            <label className="field-label" htmlFor="order-id-input">
              按订单号查询
            </label>
            <div className="inline-input-group">
              <input
                id="order-id-input"
                className="text-input"
                type="text"
                value={orderQuery}
                onChange={(e) => setOrderQuery(e.target.value)}
                placeholder="如 ORD20261101"
              />
              <button
                type="submit"
                className="btn btn-primary btn-sm btn-nowrap"
                disabled={orderLoading || !orderQuery.trim()}
              >
                {orderLoading ? '查询中' : '查询'}
              </button>
            </div>
          </form>

          {orderError && <div className="alert-banner alert-banner--error">{orderError}</div>}

          {orderDetail && (
            <div className="order-detail-card">
              <div className="order-detail-card__top">
                <code>{orderDetail.order_id}</code>
                <span className="badge badge--hybrid">{orderDetail.status}</span>
              </div>
              <dl className="order-detail-card__list">
                <div>
                  <dt>商品</dt>
                  <dd>{orderDetail.product_name}</dd>
                </div>
                <div>
                  <dt>客户 / 金额</dt>
                  <dd>
                    {orderDetail.user_name} · ¥{orderDetail.amount.toFixed(2)}
                  </dd>
                </div>
                <div>
                  <dt>物流运单号</dt>
                  <dd>{orderDetail.tracking_no || '暂未发货'}</dd>
                </div>
                <div>
                  <dt>下单时间</dt>
                  <dd>{orderDetail.created_at}</dd>
                </div>
              </dl>
            </div>
          )}

          <div className="order-quick-list">
            <div className="order-quick-list__header">
              <span>演示订单列表（{orders.length}）</span>
              <button type="button" className="link-btn" onClick={() => void loadAllOrders()}>
                刷新
              </button>
            </div>
            <div className="order-quick-items">
              {orders.map((o) => (
                <button
                  key={o.order_id}
                  type="button"
                  className={`order-quick-item ${orderDetail?.order_id === o.order_id ? 'order-quick-item--active' : ''}`}
                  onClick={() => {
                    setOrderQuery(o.order_id);
                    setOrderDetail(o);
                    setOrderError(null);
                  }}
                >
                  <div className="order-quick-item__top">
                    <code>{o.order_id}</code>
                    <span className="order-quick-item__status">{o.status}</span>
                  </div>
                  <div className="order-quick-item__product">{o.product_name}</div>
                </button>
              ))}
            </div>
          </div>
        </div>
      )}
    </aside>
  );
}
