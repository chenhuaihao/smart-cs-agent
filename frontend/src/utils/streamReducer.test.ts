/**
 * frontend/src/utils/streamReducer.test.ts
 *
 * v3.0 P3 前端核心流式协议与状态机单元测试（基于 Node 原生 node:test，零新增 npm 依赖）：
 *   1. SSE 分包/粘包/CRLF/心跳注释行解析；
 *   2. 按 seq 严格幂等去重与断线补发（replayedCount 计数准确不重复拼接 token）；
 *   3. 工具调用前中间旁白（英文/过渡语）在 tool_running 与 approval_required 阶段自动清草稿；
 *   4. policy_verifier 校验失败重写清草稿与 fallback 降级覆盖；
 *   5. citation_details 去重合并、HITL 审批过期、输入安全拦截与终态收敛。
 */

import assert from 'node:assert/strict';
import test from 'node:test';

import type { ChatEvent, ChatMessage } from '../types/chat.ts';
import {
  applyEventIdempotent,
  createTurnRuntimeMeta,
  feedSseChunk,
  formatStageLabel,
  parseSseBlock,
} from './streamReducer.ts';

function makeAssistantMsg(): ChatMessage {
  return {
    id: 'assistant-1',
    role: 'assistant',
    content: '',
    createdAt: '20:00:00',
    status: 'running',
    tools: [],
    citations: [],
    citationDetails: [],
  };
}

function makeEvent(seq: number, type: ChatEvent['type'], data: Record<string, unknown>, turnId = 't-1'): ChatEvent {
  return {
    turn_id: turnId,
    seq,
    type,
    data,
    created_at: '2026-11-25 20:00:00.000',
  };
}

test('parseSseBlock 解析标准 SSE 帧并忽略注释心跳行', () => {
  const raw = [
    ': keep-alive',
    'id: 3',
    'event: token',
    'data: {"turn_id":"t-1","seq":3,"type":"token","data":{"delta":"你好"},"created_at":"2026-11-25"}',
  ].join('\n');

  const evt = parseSseBlock(raw);
  assert.ok(evt);
  assert.equal(evt.seq, 3);
  assert.equal(evt.type, 'token');
  assert.equal(evt.data.delta, '你好');
});

test('parseSseBlock 丢弃无 data 行或非法 seq<=0 的损坏帧', () => {
  assert.equal(parseSseBlock(': ping only'), null);
  assert.equal(parseSseBlock('id: 0\nevent: token\ndata: {"turn_id":"t-1","seq":0,"type":"token"}'), null);
});

test('feedSseChunk 正确处理 TCP 跨包截断、粘包与 CRLF 换行', () => {
  const frame1 = 'id: 1\r\nevent: stage\r\ndata: {"turn_id":"t-1","seq":1,"type":"stage","data":{"stage":"received"}}\r\n\r\n';
  const frame2 = 'id: 2\r\nevent: token\r\ndata: {"turn_id":"t-1","seq":2,"type":"token","data":{"delta":"小智"}}\r\n\r\n';
  const combined = frame1 + frame2;

  // 刻意从第二帧中间劈开模拟 TCP 半包
  const splitPos = frame1.length + 22;
  const r1 = feedSseChunk('', combined.slice(0, splitPos));
  assert.equal(r1.events.length, 1);
  assert.equal(r1.events[0].seq, 1);
  assert.ok(r1.remainder.length > 0);

  const r2 = feedSseChunk(r1.remainder, combined.slice(splitPos));
  assert.equal(r2.events.length, 1);
  assert.equal(r2.events[0].seq, 2);
  assert.equal(r2.events[0].data.delta, '小智');
  assert.equal(r2.remainder, '');
});

test('applyEventIdempotent 按 seq 严格幂等去重，补发重复事件绝不重复拼接 token', () => {
  const meta = createTurnRuntimeMeta('assistant-1');
  let msg = makeAssistantMsg();

  const e1 = makeEvent(1, 'token', { delta: '您好，' });
  const e2 = makeEvent(2, 'token', { delta: '已核实。' });

  let res = applyEventIdempotent(meta, msg, e1, false);
  assert.equal(res.applied, true);
  msg = res.message;

  // 断线补发把 seq=1 再推一次：必须被幂等忽略
  res = applyEventIdempotent(meta, msg, e1, true);
  assert.equal(res.applied, false);
  assert.equal(res.message.content, '您好，');

  // 补发新事件 seq=2：正常应用且 replayedCount=1
  res = applyEventIdempotent(meta, msg, e2, true);
  assert.equal(res.applied, true);
  assert.equal(res.message.content, '您好，已核实。');
  assert.equal(res.message.replayedCount, 1);
  assert.equal(meta.lastSeq, 2);
});

test('工具调用前英文旁白在 tool_running 与 approval_required 阶段自动清零', () => {
  const meta = createTurnRuntimeMeta('assistant-1');
  let msg = makeAssistantMsg();

  msg = applyEventIdempotent(meta, msg, makeEvent(1, 'token', { delta: "I'll check your order." })).message;
  assert.equal(msg.content, "I'll check your order.");

  msg = applyEventIdempotent(meta, msg, makeEvent(2, 'stage', { stage: 'tool_running', tools: ['query_order'] })).message;
  assert.equal(msg.content, '');

  msg = applyEventIdempotent(meta, msg, makeEvent(3, 'token', { delta: 'Let me create ticket.' })).message;
  msg = applyEventIdempotent(
    meta,
    msg,
    makeEvent(4, 'approval_required', {
      turn_id: 't-1',
      tool: 'create_after_sales_ticket',
      call_id: 'c1',
      args: { order_id: 'ORD20261121' },
      summary: '即将创建退货工单',
    }),
  ).message;

  assert.equal(msg.content, '');
  assert.equal(msg.status, 'waiting_approval');
  assert.equal(meta.waitingApproval, true);
  assert.equal(msg.approval?.state, 'pending');
});

test('policy_verifier 拦截重写清空旧草稿，fallback 直接覆盖兜底文案', () => {
  const meta = createTurnRuntimeMeta('assistant-1');
  let msg = makeAssistantMsg();

  msg = applyEventIdempotent(meta, msg, makeEvent(1, 'token', { delta: '编造的草稿' })).message;
  msg = applyEventIdempotent(meta, msg, makeEvent(2, 'stage', { stage: 'verifying', passed: false })).message;
  assert.equal(msg.content, '');

  msg = applyEventIdempotent(meta, msg, makeEvent(3, 'token', { delta: '合规新回答' })).message;
  assert.equal(msg.content, '合规新回答');

  msg = applyEventIdempotent(meta, msg, makeEvent(4, 'token', { delta: '安全兜底回复', fallback: true })).message;
  assert.equal(msg.isFallback, true);
  assert.equal(msg.content, '安全兜底回复');
});

test('citation 事件按 source 去重合并且保留 Hybrid RAG 检索来源标签', () => {
  const meta = createTurnRuntimeMeta('assistant-1');
  let msg = makeAssistantMsg();

  msg = applyEventIdempotent(
    meta,
    msg,
    makeEvent(1, 'citation', {
      citations: ['return_policy.md#未拆封耳机退货'],
      citation_details: [
        { source: 'return_policy.md#未拆封耳机退货', retrieval_source: 'hybrid', score: 0.0328 },
      ],
    }),
  ).message;

  // 重复推送同一 source 应自动去重
  msg = applyEventIdempotent(
    meta,
    msg,
    makeEvent(2, 'citation', {
      citations: ['return_policy.md#未拆封耳机退货', 'warranty_policy.md#耳机质保'],
      citation_details: [
        { source: 'return_policy.md#未拆封耳机退货', retrieval_source: 'bm25', score: 5.0 },
        { source: 'warranty_policy.md#耳机质保', retrieval_source: 'vector', score: 0.0161 },
      ],
    }),
  ).message;

  assert.equal(msg.citations.length, 2);
  assert.equal(msg.citationDetails.length, 2);
  assert.equal(msg.citationDetails[0].retrieval_source, 'hybrid');
  assert.equal(msg.citationDetails[1].retrieval_source, 'vector');
});

test('done 与 error 终态收敛及 v3.0 新阶段文案映射正确', () => {
  assert.match(formatStageLabel({ stage: 'approval_expired', reason: '超时300s' }), /审批已超时取消/);
  assert.match(formatStageLabel({ stage: 'input_guard_blocked' }), /输入安全护栏/);

  const meta = createTurnRuntimeMeta('assistant-1');
  let msg = makeAssistantMsg();
  const res = applyEventIdempotent(
    meta,
    msg,
    makeEvent(1, 'done', {
      reply: '最终权威中文回复',
      tools_called: ['query_order'],
      citations: ['return_policy.md#规则'],
      ttft_ms: 120,
      latency_ms: 880,
    }),
  );
  assert.equal(meta.terminal, true);
  assert.equal(res.completedData?.latency_ms, 880);
  assert.equal(res.message.status, 'completed');
  assert.equal(res.message.content, '最终权威中文回复');

  const metaErr = createTurnRuntimeMeta('assistant-2');
  const errRes = applyEventIdempotent(
    metaErr,
    makeAssistantMsg(),
    makeEvent(1, 'error', { message: '服务重启导致中断', latency_ms: 50 }),
  );
  assert.equal(metaErr.terminal, true);
  assert.equal(errRes.message.status, 'failed');
  assert.equal(errRes.message.content, '服务重启导致中断');
});

test('evidence_updated 同步更新证据账本并自动合并去重 citation_details', () => {
  const meta = createTurnRuntimeMeta('assistant-1');
  let msg = makeAssistantMsg();

  msg = applyEventIdempotent(
    meta,
    msg,
    makeEvent(1, 'stage', {
      stage: 'evidence_updated',
      tool: 'query_order',
      evidence_ledger: {
        orders: {
          ORD20261101: { order_id: 'ORD20261101', status: '已签收', exists: true },
        },
        citations: ['return_policy.md#规则A'],
        citation_details: [
          { source: 'return_policy.md#规则A', retrieval_source: 'hybrid', score: 0.032 },
        ],
        tickets: {},
      },
    }),
  ).message;

  assert.equal(msg.evidenceLedger?.orders.ORD20261101.status, '已签收');
  assert.equal(msg.citationDetails.length, 1);
  assert.equal(msg.citationDetails[0].retrieval_source, 'hybrid');
});

test('tool_call 事件按顺序累积并保留 success / cancelled / error 状态', () => {
  const meta = createTurnRuntimeMeta('assistant-1');
  let msg = makeAssistantMsg();

  msg = applyEventIdempotent(
    meta,
    msg,
    makeEvent(1, 'tool_call', {
      call_id: 'c1',
      name: 'query_order',
      arguments: { order_id: 'ORD20261121' },
      status: 'success',
      duration_ms: 2,
    }),
  ).message;
  msg = applyEventIdempotent(
    meta,
    msg,
    makeEvent(2, 'tool_call', {
      call_id: 'c2',
      name: 'create_after_sales_ticket',
      arguments: { order_id: 'ORD20261121' },
      status: 'cancelled',
      duration_ms: 0,
    }),
  ).message;

  assert.equal(msg.tools.length, 2);
  assert.equal(msg.tools[0].status, 'success');
  assert.equal(msg.tools[1].status, 'cancelled');
});

test('feedSseChunk 单次粘包包含 3 帧时完整无漏切出', () => {
  const burst = [1, 2, 3]
    .map(
      (i) =>
        `id: ${i}\nevent: token\ndata: {"turn_id":"t-burst","seq":${i},"type":"token","data":{"delta":"字${i}"}}\n\n`,
    )
    .join('');

  const { events, remainder } = feedSseChunk('', burst);
  assert.equal(events.length, 3);
  assert.deepEqual(
    events.map((e) => e.seq),
    [1, 2, 3],
  );
  assert.equal(remainder, '');
});

test('applyEventIdempotent 对缺少 turn_id 的非法事件安全忽略', () => {
  const meta = createTurnRuntimeMeta('assistant-1');
  const msg = makeAssistantMsg();
  const invalid = makeEvent(1, 'token', { delta: 'bad' }, '');
  const res = applyEventIdempotent(meta, msg, invalid, true);
  assert.equal(res.applied, false);
  assert.equal(meta.lastSeq, 0);
});

