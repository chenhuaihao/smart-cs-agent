# Smart CS Agent v3.0 - 企业智能售后与工单分流 Agent

<p align="center">
  <b>面向真实电商售后与工单流转场景的生产级智能客服系统</b><br>
  SSE 事务事件流 · 双层安全护栏与证据账本 · HITL 生命周期守护 · Hybrid RAG + Pre-RRF 双路门控
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue.svg" alt="Python 3.12">
  <img src="https://img.shields.io/badge/FastAPI-0.110+-009688.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/React-19_+_TypeScript-61DAFB.svg" alt="React 19">
  <img src="https://img.shields.io/badge/Hybrid_RAG-Hit@3_100%25-brightgreen.svg" alt="Hybrid RAG Hit@3 100%">
  <img src="https://img.shields.io/badge/Tests-61_Passed_(49_BE_+_12_FE)-success.svg" alt="61 Tests Passed">
  <img src="https://img.shields.io/badge/Docker-Multi--Stage-2496ED.svg" alt="Docker Multi-Stage">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

<p align="center">
  🌐 <a href="http://8.166.124.35:8000/"><b>在线体验工作台</b></a> ·
  📊 <a href="http://8.166.124.35:8000/#eval"><b>在线评测报告页（实时读取 eval_summary.json）</b></a> ·
  🏛️ <a href="docs/ARCHITECTURE.md"><b>架构决策（ADR）与排障复盘</b></a> ·
  📖 <a href="http://8.166.124.35:8000/docs"><b>Swagger API 文档</b></a>
</p>

---

## 🖼️ 系统核心界面实拍

| 智能对话工作台 + Hybrid RAG 联调面板 | HITL 高风险建单人工确认 + 双路融合溯源 |
| :---: | :---: |
| ![智能对话工作台与Hybrid RAG联调](docs/images/01_hybrid_rag_chat.jpg) | ![HITL高风险建单人工确认](docs/images/02_hitl_approval.jpg) |
| **售后工单看板（严格单向状态机）** | **在线量化评测报告（4 组消融对比 + 失败归因）** |
| ![售后工单看板](docs/images/03_ticket_board.jpg) | ![系统量化评测报告](docs/images/04_eval_report.jpg) |

---

## 📊 核心评测结果速览（第一屏指标）

> 所有指标均由 [`eval/run_rag_eval.py`](eval/run_rag_eval.py) 自动输出至 [`eval/eval_summary.json`](eval/eval_summary.json) 并由网站「[评测报告](http://8.166.124.35:8000/#eval)」标签页实时渲染，与完整报告 [`eval/rag_eval_report.md`](eval/rag_eval_report.md) 严格同源。

### 1. 四种检索配置量化对比（真实 `BAAI/bge-m3` 冷查询实测）

**A. 40 题开发校准集**（`eval/rag_eval_cases.json`：15 书面 + 15 口语 + 10 域外，门控阈值基于此集分布定型）：

| 检索配置 | Hit@3（30题） | 书面 Hit@3 | 口语 Hit@3 | MRR | Precision@3 | 域外误召回率 FPR（10题） | 平均 / P95 耗时 (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. 纯 BM25** | 56.7% | 100.0% | 13.3% | 0.5278 | 40.6% | 10.0% (1/10) | 0.4 / 0.5 |
| **2. 纯向量 (Vector Only)** | 100.0% | 100.0% | 100.0% | 0.9833 | 46.7% | 100.0% (10/10) | 128.0 / 174.5 |
| **3. Hybrid 无门控** | 96.7% | 100.0% | 93.3% | 0.8833 | 42.2% | 100.0% (10/10) | 137.0 / 213.0 |
| **4. Hybrid + Pre-RRF 双路门控**（生产默认） | **100.0%** | **100.0%** | **100.0%** | **0.9833** | **72.8%** | **10.0% (1/10)** | **138.6 / 213.0** |

**B. 18 题独立留出验证集**（[`eval/rag_holdout_cases.json`](eval/rag_holdout_cases.json)：6 书面 + 6 口语 + 6 域外，**未参与任何阈值校准，零改参直接复测**，详见 [`eval/rag_holdout_report.md`](eval/rag_holdout_report.md)）：

| 检索配置 | Hit@3（12题） | 书面 Hit@3 | 口语 Hit@3 | MRR | Precision@3 | 域外误召回率 FPR（6题） | 平均 / P95 耗时 (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. 纯 BM25** | 58.3% | 100.0% | 16.7% | 0.5000 | 41.7% | 16.7% (1/6) | 0.9 / 2.8 |
| **2. 纯向量 (Vector Only)** | 100.0% | 100.0% | 100.0% | 0.8750 | 47.2% | 100.0% (6/6) | 197.8 / 713.7 |
| **3. Hybrid 无门控** | 100.0% | 100.0% | 100.0% | 0.8194 | 41.7% | 100.0% (6/6) | 135.9 / 269.9 |
| **4. Hybrid + Pre-RRF 双路门控**（生产默认） | **100.0%** | **100.0%** | **100.0%** | **0.9167** | **69.4%** | **16.7% (1/6)** | **135.8 / 193.2** |

### 2. 端到端 100 题并发业务评测核心指标（v3.0 实测）

| 总体通过率 (Pass Rate) | 工具路由准确率 (Tool Accuracy) | 防幻觉安全率 (Safety Rate) | 5 大模块达标情况 | 在线评测明细 |
| :---: | :---: | :---: | :---: | :---: |
| **96.0%** (96/100) | **97.0%** (97/100) | **98.0%** (98/100) | **5/5 全部 ≥ 90%（3 项 100%）** | 👉 [查看在线评测报告](http://8.166.124.35:8000/#eval) |

> **关键结论**：相比纯 BM25，Hybrid + Pre-RRF 将口语化 Hit@3 从 **13.3% 提升至 100.0%**；相比无门控 Hybrid，将 **Precision@3 从 42.2% 提升至 72.8%（+30.6pp）**、域外误召回率从 **100.0% 降至 10.0%（-90.0pp）**；生产环境开启 256 容量 Query Embedding LRU 缓存后，高频问法耗时可降至 **~4.6 ms**。

---

## 📌 项目定位与 v3.0 工程演进

`smart-cs-agent` 是一套面向电商平台与企业客服真实业务场景的**生产级轻量智能售后 Agent**。v3.0 在 v2.0（可恢复 SSE、Hybrid RAG、HITL 建单、React 控制台）基础上，进一步围绕**运行态收敛、双层安全护栏与全栈测试厚度**完成架构加固（详见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)）：

- **可恢复 SSE 事务事件流**：基于 SQLite WAL 事务日志（`chat_turns` / `chat_events`）先落库分配严格递增 `seq` 再推流；前端纯函数状态机 `streamReducer.ts` 维护 `turn_id + lastSeq + processedSeqs` 幂等去重，流中断后自动通过 `GET /api/chat/turns/{turn_id}/events?after_seq=...` 无缝补发。
- **双层安全护栏与间接注入隔离**：输入侧 0ms 拦截高置信 Prompt Injection / 系统提示词窃取 / 管理员身份伪造 / SQL 注入（6 类攻击样本全拦、6 条正常售后回归用例 0 误杀）；工具层对外部检索与历史工单返回施加 `<untrusted_tool_output>` 数据围栏防间接注入；输出侧 `LangGraph StateGraph`（`agent → tools → policy_verifier → fallback`）基于 `EvidenceLedger` 输出 6 类结构化违规码硬拦截编造订单、虚假建单与 `.md` 路径泄露。
- **高风险写操作 HITL 与生命周期闭环（v3.0）**：流式触发 `create_after_sales_ticket` 挂起落库并推送授权卡片；通过 `run_guard.py` 解耦活跃推理预算（`TURN_TIMEOUT_SEC`）与人工等待时限（`APPROVAL_TTL_SEC`），`/approve` 在发送 200 响应头前通过 SQLite `BEGIN IMMEDIATE` 原子 CAS 防并发双击重单，看门狗自动回收重启遗留僵尸 `running` 与过期 `expired` 审批。
- **Hybrid RAG + Pre-RRF 双路原生门控**：融合 `jieba + rank-bm25` 词法检索与 `BAAI/bge-m3 + Chroma` 本地持久化语义检索，经 Pre-RRF 双路原生相关性门控与 RRF（$k=60$）单路尾部裁剪，在 40 题校准集与 18 题独立留出集上均保持 **100% 口语 Hit@3**，留出集 Precision@3 相对无门控仍提升 **+27.7pp（41.7%→69.4%）**、域外 FPR 降至 **16.7%**；无 Key 或在线超时自动降级纯 BM25，并配备进程级启动预热 + 256 容量 Query Embedding LRU 缓存（缓存命中耗时 ~4.6 ms）。
- **全栈自动化测试与防刷加固（v3.0）**：沉淀 **49 条后端 pytest 用例**（覆盖并发写事件 `seq`、HITL 5 线程并发双击 CAS、TTL 过期、跨重启恢复、僵尸看门狗、双层护栏与间接注入围栏、代理头信任边界 `TRUST_PROXY_HEADERS` 与单 IP 日配额隔离、Hybrid RAG）+ **12 条前端原生 `node:test` 流式状态机用例**（零新增 npm 依赖）。
- **React + TypeScript + Vite 控制台**：集成智能客服流式工作台、严格单向状态机工单看板（`待处理 → 处理中 → 已完成`，终态禁逆转）以及订单与 Hybrid RAG 实时联调面板。

---

## 🏗️ v2.0 系统架构

```text
           +-------------------------------------------------------------------+
           |        Frontend Console (React 19 + TypeScript + Vite)            |
           |  ChatWorkspace (SSE+seq去重) | TicketBoard | InspectorDrawer      |
           +---------------------------------+---------------------------------+
                                             | HTTP / SSE (/api/*)
                                             v
+---------------------------------------------------------------------------------------------+
|                             FastAPI Gateway & Demo Guardrails                               |
|  - 对话滑动窗口限流 (RATE_LIMIT_PER_MINUTE) + 每日全局熔断 (DAILY_CHAT_LIMIT)                 |
|  - 业务接口分桶 IP 限流 (读 60/min, 写 20/min) + 重置双层保护 (DEMO_ADMIN_TOKEN / 5min冷却)   |
|  POST /api/chat/stream | GET /turns/{id}/events | POST /turns/{id}/approve | /orders|/tickets|
+--------------------------------------------+------------------------------------------------+
                                             |
                                             v
+---------------------------------------------------------------------------------------------+
|                       LangGraph StateGraph + Transactional Event Store                      |
|                                                                                             |
|   START -> agent --(tool_calls)--> tools --(高风险写操作 HITL 挂起)--> approval_required     |
|              |                       |                                    | (/approve)      |
|              |                       +--> 更新 EvidenceLedger + 派发 citation_details        |
|              v                                                                              |
|        policy_verifier --(pass)--> END (done 落库)                                          |
|              |--(fail & retry<=2)--> agent 重写                                             |
|              +--(fail>2 / 超时 / 步数耗尽)--> fallback -> END                               |
+--------------------------------------------+------------------------------------------------+
                                             |
                   +-------------------------+-------------------------+
                   |                                                   |
                   v                                                   v
   +------------------------------------------+      +----------------------------------------+
   |       Hybrid RAG Engine (低内存无Torch)   |      |        SQLite Native Storage (WAL)     |
   | - 词法路: jieba + BM25 (绝对+0.5相对门控) |      | - orders (24条标准覆盖全周期种子单)     |
   | - 语义路: bge-m3 + Chroma 持久化 (指纹复用)|      | - tickets (严格单向状态机)              |
   |   (余弦>=0.58 + Top1落差<=0.10 + LRU缓存) |      | - chat_logs (最近6轮会话记忆)           |
   | - 融合层: RRF (k=60) + 单路尾部裁剪 (0.6) |      | - chat_turns / chat_events / approvals |
   +------------------------------------------+      +----------------------------------------+
```

---

## 🛠️ 技术栈选型

| 模块 | 技术选型 | 说明 |
| :--- | :--- | :--- |
| **后端服务** | Python 3.12 · FastAPI · Pydantic v2 · Uvicorn | 全异步接口、严格 Schema 校验与 Swagger 文档 |
| **状态编排** | LangGraph (`StateGraph`) | 显式节点与护栏流转，不引入重型链式黑盒封装 |
| **大模型适配** | 官方 `openai` SDK（端口-适配器模式） | 支持 DeepSeek / 通义 / 豆包等 OpenAI 兼容协议热切换 |
| **混合检索** | `jieba` + `rank-bm25` + `chromadb` + 在线 Embedding | 零 `torch` / 本地大模型依赖，低内存容器友好，无 Key 自动降级 BM25 |
| **数据存储** | Python 标准库 `sqlite3`（WAL 模式 + 参数化 SQL） | 零 ORM 黑盒，事务保证事件 `seq` 严格递增与工单一致性 |
| **前端控制台** | React 19 · TypeScript · Vite 6 | 仅 7 个核心依赖，自定义 SSE 帧解析与 `seq` 幂等补发状态机 |
| **容器部署** | Multi-Stage Docker (`node:20-alpine` + `python:3.12-slim`) | 单容器托管前后端，挂载 `./data` 持久化 SQLite 与 Chroma 索引 |

---

## ⚡ 本地运行、容器部署与评测复现

### 1. 环境准备

```bash
git clone https://github.com/chenhuaihao/smart-cs-agent.git
cd smart-cs-agent

# 创建 Python 3.12 虚拟环境并安装依赖
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt

# 复制环境变量模板并填入真实 API Key（EMBEDDING_API_KEY 可选，留空自动降级纯 BM25）
cp .env.example .env
```

### 2. 前后端本地启动

**方式 A：开发热更新联调（推荐）**
```bash
# 终端 1：启动 FastAPI 后端（端口 8000）
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

# 终端 2：安装并启动 React + Vite 前端（端口 5173，/api 自动代理至 8000 且不缓冲 SSE）
cd frontend
npm install
npm run dev
```
打开浏览器访问 **[http://127.0.0.1:5173/](http://127.0.0.1:5173/)**（Swagger 文档：[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)）。

**方式 B：单端口生产预览**
```bash
cd frontend && npm install && npm run build && cd ..
uvicorn app.main:app --host 127.0.0.1 --port 8000
```
直接访问 **http://127.0.0.1:8000/** 即可使用编译后的 v2.0 控制台。

### 3. Docker 多阶段一键部署

```bash
# 确保项目根目录已配置 .env
docker compose up -d --build

# 查看容器状态与健康检查
docker compose ps
docker compose logs -f smart-cs-agent
```
容器默认通过 `./data:/app/data` 挂载持久化 SQLite（`smart_cs.db`）与 Chroma 向量库（`chroma_db/`）。

### 4. 单元测试与评测复现

```bash
# 1. 运行全量后端单元测试（49 条：流式补发 / 双层护栏 / 间接注入围栏 / HITL 生命周期 / 并发压测 / Hybrid RAG，全离线）
pytest tests/ -v

# 2. 运行前端原生单元测试（12 条 SSE 分包与状态机用例）、类型检查与生产构建
cd frontend && npm test && npx tsc --noEmit && npm run build

# 3. 复现 40 题校准集 & 18 题独立留出集 RAG 量化评测（需 .env 已配置 EMBEDDING_API_KEY）
python eval/run_rag_eval.py
python eval/run_rag_eval.py --holdout

# 4. 复现 100 题端到端对话并发评测（需服务已启动，全量跑完自动同步 eval_summary.json）
RATE_LIMIT_PER_MINUTE=1000 DAILY_CHAT_LIMIT=10000 python eval/run_eval.py --workers 4
```

---

## 📂 目录结构

```text
smart-cs-agent/
├── app/
│   ├── main.py                   # FastAPI 入口、启动预热与 frontend/dist 自动托管
│   ├── agent/
│   │   ├── core.py               # 流式调度、活跃预算与 HITL 挂起/恢复
│   │   ├── run_guard.py          # v3.0 活跃预算解耦、APPROVAL_TTL 过期与僵尸轮次看门狗
│   │   ├── graph.py              # LangGraph 状态机（input_guard / agent / tools / policy_verifier / fallback）
│   │   ├── guardrails.py         # 双层护栏：输入安全检查 + EvidenceLedger 结构化违规码校验器
│   │   ├── knowledge.py          # Hybrid RAG（BM25 + 向量 + Pre-RRF 门控 + RRF 裁剪 + LRU 缓存）
│   │   ├── vector_store.py       # Chroma 本地持久化封装与内容哈希增量同步
│   │   ├── tools.py              # 4 个业务工具定义与安全分发器
│   │   ├── memory.py             # 会话最近 6 轮上下文读写
│   │   └── prompts.py            # 系统指令与业务守则
│   ├── llm/
│   │   ├── base.py               # BaseLLMProvider 抽象端口
│   │   ├── deepseek.py           # OpenAI 兼容流式 Provider 适配器
│   │   ├── factory.py            # LLM Provider 单例工厂
│   │   └── embedding.py          # BaseEmbeddingProvider + OpenAI 兼容在线 Embedding 适配器
│   ├── core/limiter.py           # 滑动窗口 IP 限流、每日熔断与重置冷却/口令保护
│   ├── db/                       # SQLite DDL、事务事件存储（events.py）与 24 条标准种子数据（seed.py）
│   ├── routers/                  # /api/chat、/api/orders、/api/tickets、/api/knowledge
│   └── schemas/                  # Pydantic v2 契约模型
├── frontend/                     # React 19 + TypeScript + Vite 6 控制台工程
│   ├── vite.config.ts            # /api 代理与 SSE 透传配置
│   └── src/
│       ├── types/chat.ts         # 与后端对齐的完整 TypeScript 协议类型
│       ├── api/client.ts         # REST + ReadableStream SSE 解析客户端
│       ├── utils/streamReducer.ts# 纯函数 SSE 切帧器与流式事件状态机（附 12 条原生单元测试）
│       ├── hooks/useChatStream.ts# 流式状态机 Hook、after_seq 自动补发与 seq 幂等去重
│       └── components/           # ChatWorkspace / ApprovalCard / TicketBoard / InspectorDrawer / EvalReportView
├── docs/
│   ├── ARCHITECTURE.md           # v3.0 架构权衡决策（ADR）、HITL 状态机与 5 项真实排障复盘
│   └── images/                   # 生产界面实拍图
├── data/knowledge/               # 6 篇售后规则与故障排查 Markdown 知识库
├── eval/                         # 100 题端到端评测 + 40 题校准集 + 18 题独立留出集 RAG 消融套件
├── tests/                        # 全离线自动化验收测试套件（49 条后端用例，含并发/安全/HITL 回归）
├── Dockerfile                    # Node 20 + Python 3.12 多阶段轻量构建
├── docker-compose.yml            # 单容器编排与数据卷持久化
└── .env.example                  # 全量环境变量模板
```

---

## 🌐 公网演示环境与开源协议

- **公网演示地址**：[http://8.166.124.35:8000](http://8.166.124.35:8000)（阿里云 ECS 华南 3·广州，Docker Compose 单容器部署，启用 IP 限流与每日配额保护）
- **开源协议**：本项目基于 [MIT License](LICENSE) 开源。
