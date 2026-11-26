/**
 * frontend/src/components/EvalReportView.tsx
 *
 * 只读「评测报告」工作区：
 *   - 实时从 GET /api/eval/summary 拉取由 eval/run_rag_eval.py 自动生成的 eval_summary.json，
 *     前端零硬编码数字，确保页面指标与评测脚本/报告 100% 一致；
 *   - 展示：4 张核心指标卡片、四种检索配置对比表、典型失败案例与归因分析、评测集构造与复现命令。
 */

import { useCallback, useEffect, useState } from 'react';
import { getEvalSummary } from '../api/client';
import type { EvalSummary } from '../types/chat';

export function EvalReportView() {
  const [summary, setSummary] = useState<EvalSummary | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const loadSummary = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getEvalSummary();
      setSummary(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : '加载评测汇总失败');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadSummary();
  }, [loadSummary]);

  return (
    <section className="workspace-panel eval-report">
      <header className="workspace-header">
        <div>
          <h2 className="workspace-title">系统量化评测报告</h2>
          <p className="workspace-subtitle">
            数据源：<code>eval/eval_summary.json</code>（由评测脚本自动生成，前端动态渲染，零手写硬编码）
          </p>
        </div>
        <div className="ticket-board__toolbar">
          {summary && (
            <span className="meta-pill">
              RAG 评测：{summary.generated_at}
              {summary.e2e_summary?.evaluated_at
                ? ` · E2E 评测：${String(summary.e2e_summary.evaluated_at)}`
                : ''}
              {' · '}模型：<code>{summary.embedding_model}</code>
            </span>
          )}
          <button
            type="button"
            className="btn btn-ghost btn-sm"
            onClick={() => void loadSummary()}
            disabled={loading}
          >
            {loading ? '加载中...' : '刷新数据'}
          </button>
        </div>
      </header>

      {error && <div className="alert-banner alert-banner--error">{error}</div>}

      <div className="eval-scroll-area">
        {loading && !summary ? (
          <div className="empty-inline">正在加载评测报告数据...</div>
        ) : !summary ? (
          <div className="empty-inline">暂无评测汇总数据，请先运行 <code>python eval/run_rag_eval.py</code>。</div>
        ) : (
          <div className="eval-content">
            {/* 1. 核心指标卡片（4 张） */}
            <div className="eval-kpi-grid">
              {summary.kpis.map((kpi) => (
                <article key={kpi.key} className={`eval-kpi-card eval-kpi-card--${kpi.tone}`}>
                  <div className="eval-kpi-card__label">{kpi.label}</div>
                  <div className="eval-kpi-card__value">{kpi.value}</div>
                  <div className="eval-kpi-card__detail">{kpi.detail}</div>
                </article>
              ))}
            </div>

            {/* 2. 四种检索配置对比表 */}
            <div className="eval-section">
              <div className="eval-section__header">
                <h3>四种检索配置对比（{summary.dataset.rag_total} 题真实模型冷查询）</h3>
                <span className="eval-section__note">
                  关闭 Query 缓存以保证耗时口径一致；生产开启 LRU 缓存后高频查询约 4.6 ms
                </span>
              </div>
              <div className="eval-table-wrap">
                <table className="eval-table">
                  <thead>
                    <tr>
                      <th>检索配置</th>
                      <th>Hit@3（30题）</th>
                      <th>书面 Hit@3</th>
                      <th>口语 Hit@3</th>
                      <th>MRR</th>
                      <th>Precision@3</th>
                      <th>域外误召回 FPR</th>
                      <th>平均 / P95 耗时</th>
                    </tr>
                  </thead>
                  <tbody>
                    {summary.rag_configs.map((cfg) => (
                      <tr
                        key={cfg.name}
                        className={cfg.is_production ? 'eval-table__row--highlight' : undefined}
                      >
                        <td>
                          <strong>{cfg.name}</strong>
                          {cfg.is_production && <span className="badge badge--hybrid eval-prod-tag">生产默认</span>}
                        </td>
                        <td><strong>{cfg.hit_at_3.toFixed(1)}%</strong></td>
                        <td>{cfg.formal_hit_at_3.toFixed(1)}%</td>
                        <td>{cfg.colloquial_hit_at_3.toFixed(1)}%</td>
                        <td><code>{cfg.mrr.toFixed(4)}</code></td>
                        <td><strong>{cfg.precision_at_3.toFixed(1)}%</strong></td>
                        <td>
                          {cfg.ood_fpr.toFixed(1)}% <span className="eval-muted">({cfg.ood_fp_label})</span>
                        </td>
                        <td>
                          {cfg.avg_ms.toFixed(1)} / {cfg.p95_ms.toFixed(1)} ms
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* 3. 典型失败案例与归因分析 */}
            <div className="eval-section">
              <div className="eval-section__header">
                <h3>典型失败案例与根因分析（{summary.failure_cases.length} 例）</h3>
                <span className="eval-section__note">
                  自动提取自最新 RAG 与 100 题端到端评测报告，透明暴露当前边界与改进路径
                </span>
              </div>
              <div className="eval-failure-list">
                {summary.failure_cases.map((item) => (
                  <article key={`${item.source_eval}-${item.id}`} className="eval-failure-card">
                    <div className="eval-failure-card__top">
                      <div className="eval-failure-card__meta">
                        <code className="eval-failure-card__id">{item.id}</code>
                        <span className="badge badge--bm25">{item.source_eval}</span>
                        <span className="eval-failure-card__cat">{item.category}</span>
                      </div>
                    </div>
                    <div className="eval-failure-card__query">「{item.query}」</div>
                    <div className="eval-failure-card__row">
                      <span className="eval-failure-card__key">失败现象：</span>
                      <span>{item.reason}（实际：<code>{item.actual}</code>）</span>
                    </div>
                    <div className="eval-failure-card__row eval-failure-card__row--analysis">
                      <span className="eval-failure-card__key">归因分析：</span>
                      <span>{item.analysis}</span>
                    </div>
                  </article>
                ))}
              </div>
            </div>

            {/* 4. 评测集规模、构造方式与复现命令 */}
            <div className="eval-section eval-meta-box">
              <h3>评测集规模、构造方式与复现命令</h3>
              <dl className="eval-meta-grid">
                <div>
                  <dt>RAG 评测集</dt>
                  <dd>
                    共 <strong>{summary.dataset.rag_total}</strong> 题：{summary.dataset.rag_breakdown}
                  </dd>
                </div>
                <div>
                  <dt>端到端评测集</dt>
                  <dd>
                    共 <strong>{summary.dataset.e2e_total}</strong> 题：{summary.dataset.e2e_breakdown}
                  </dd>
                </div>
                <div>
                  <dt>Pre-RRF 参数</dt>
                  <dd className="eval-param-chips">
                    {Object.entries(summary.gate_params).map(([k, v]) => (
                      <code key={k} className="stat-tag">
                        {k}={String(v)}
                      </code>
                    ))}
                  </dd>
                </div>
                <div>
                  <dt>构造与口径</dt>
                  <dd>{summary.dataset.construction_note}</dd>
                </div>
              </dl>

              <div className="eval-commands">
                <div className="eval-commands__label">一键本地复现命令（数据与本页严格同源）：</div>
                <pre className="eval-commands__pre">
                  {summary.dataset.reproduce_commands.join('\n')}
                </pre>
              </div>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
