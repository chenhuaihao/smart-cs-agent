/**
 * frontend/src/App.tsx
 *
 * Smart CS Agent v2.0 控制台主布局：
 *   - 顶部导航栏：品牌标识、后端健康探测、主工作区切换（智能客服对话 / 售后工单看板）与右侧联调面板开关；
 *   - 左侧主工作区：ChatWorkspace 与 TicketBoard 切换（建单完成后自动触发工单看板刷新信号）；
 *   - 右侧联调面板：InspectorDrawer（Hybrid RAG 检索调试 + 订单状态查询，支持与对话/工单订单号联动）。
 */

import { useCallback, useEffect, useState } from 'react';
import { checkHealth } from './api/client';
import { ChatWorkspace } from './components/ChatWorkspace';
import { EvalReportView } from './components/EvalReportView';
import { InspectorDrawer } from './components/InspectorDrawer';
import { TicketBoard } from './components/TicketBoard';
import { useChatStream } from './hooks/useChatStream';
import type { DoneEventData } from './types/chat';

type WorkspaceTab = 'chat' | 'tickets' | 'eval';

function getInitialWorkspace(): WorkspaceTab {
  const hash = window.location.hash.replace(/^#/, '').toLowerCase();
  if (hash === 'eval') return 'eval';
  if (hash === 'tickets') return 'tickets';
  return 'chat';
}

export default function App() {
  const [activeWorkspace, setActiveWorkspace] = useState<WorkspaceTab>(getInitialWorkspace);
  const [drawerOpen, setDrawerOpen] = useState<boolean>(false);
  const [selectedOrderId, setSelectedOrderId] = useState<string>('');
  const [ticketRefreshSignal, setTicketRefreshSignal] = useState<number>(0);
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);

  const switchWorkspace = useCallback((tab: WorkspaceTab) => {
    setActiveWorkspace(tab);
    const nextHash = tab === 'chat' ? '' : `#${tab}`;
    if (window.location.hash !== nextHash) {
      window.history.replaceState(null, '', nextHash || window.location.pathname);
    }
  }, []);

  useEffect(() => {
    const onHashChange = () => setActiveWorkspace(getInitialWorkspace());
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);

  const handleTurnComplete = useCallback((done: DoneEventData) => {
    if (done.tools_called.includes('create_after_sales_ticket')) {
      setTicketRefreshSignal((v) => v + 1);
    }
  }, []);

  const chat = useChatStream({ onTurnComplete: handleTurnComplete });

  useEffect(() => {
    let active = true;
    checkHealth()
      .then(() => {
        if (active) setBackendOnline(true);
      })
      .catch(() => {
        if (active) setBackendOnline(false);
      });
    return () => {
      active = false;
    };
  }, []);

  const handleInspectOrder = useCallback((orderId: string) => {
    setSelectedOrderId(orderId);
    setDrawerOpen(true);
  }, []);

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="topbar__brand">
          <div className="topbar__logo">CS</div>
          <div>
            <div className="topbar__name">
              Smart CS Agent <span className="topbar__ver">v2.0</span>
            </div>
            <div className="topbar__desc">企业智能售后与工单分流控制台</div>
          </div>
        </div>

        <nav className="topbar__nav" aria-label="主工作区切换">
          <button
            type="button"
            className={`nav-tab ${activeWorkspace === 'chat' ? 'nav-tab--active' : ''}`}
            onClick={() => switchWorkspace('chat')}
          >
            智能客服对话
          </button>
          <button
            type="button"
            className={`nav-tab ${activeWorkspace === 'tickets' ? 'nav-tab--active' : ''}`}
            onClick={() => {
              switchWorkspace('tickets');
              setTicketRefreshSignal((v) => v + 1);
            }}
          >
            售后工单看板
          </button>
          <button
            type="button"
            className={`nav-tab ${activeWorkspace === 'eval' ? 'nav-tab--active' : ''}`}
            onClick={() => switchWorkspace('eval')}
          >
            评测报告
          </button>
        </nav>

        <div className="topbar__actions">
          <span
            className={`health-pill ${
              backendOnline === true
                ? 'health-pill--ok'
                : backendOnline === false
                  ? 'health-pill--err'
                  : ''
            }`}
          >
            <span className="health-pill__dot" />
            {backendOnline === true
              ? 'API 在线'
              : backendOnline === false
                ? 'API 未连接'
                : '检测服务中'}
          </span>
          <a
            className="btn btn-ghost btn-sm github-link"
            href="https://github.com/chenhuaihao/smart-cs-agent"
            target="_blank"
            rel="noopener noreferrer"
            title="访问 GitHub 开源仓库"
          >
            <svg viewBox="0 0 16 16" width="14" height="14" fill="currentColor" aria-hidden="true">
              <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
            </svg>
            <span>GitHub</span>
          </a>
          <button
            type="button"
            className="btn btn-ghost btn-sm"
            onClick={() => setDrawerOpen((v) => !v)}
          >
            {drawerOpen ? '收起联调面板' : '展开联调面板'}
          </button>
        </div>
      </header>

      <main className={`console-grid ${drawerOpen ? 'console-grid--split' : 'console-grid--single'}`}>
        <div className="console-main">
          {activeWorkspace === 'chat' ? (
            <ChatWorkspace
              sessionId={chat.sessionId}
              messages={chat.messages}
              isStreaming={chat.isStreaming}
              isReplaying={chat.isReplaying}
              activeTurnId={chat.activeTurnId}
              lastSeq={chat.lastSeq}
              streamError={chat.streamError}
              onSend={(text) => void chat.sendMessage(text)}
              onDecideApproval={(tid, ok) => void chat.decideApproval(tid, ok)}
              onSyncTurn={(tid) => void chat.syncTurn(tid)}
              onAbortStream={chat.abortStream}
              onResetSession={chat.resetSession}
              onInspectOrder={handleInspectOrder}
            />
          ) : activeWorkspace === 'tickets' ? (
            <TicketBoard
              refreshSignal={ticketRefreshSignal}
              onInspectOrder={handleInspectOrder}
              onDemoReset={() => setTicketRefreshSignal((v) => v + 1)}
            />
          ) : (
            <EvalReportView />
          )}
        </div>

        {drawerOpen && (
          <InspectorDrawer
            selectedOrderId={selectedOrderId}
            refreshSignal={ticketRefreshSignal}
            onClose={() => setDrawerOpen(false)}
          />
        )}
      </main>
    </div>
  );
}
