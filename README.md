# Smart CS Agent - 企业智能售后与工单分流 Agent

<p align="center">
  <b>面向真实电商售后与工单流转场景的生产级智能客服系统</b><br>
  轻量透明 · 拒绝黑盒 · 状态机闭环 · 生产就绪
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/FastAPI-0.110+-009688.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/Model-DeepSeek--V3-4D6BFE.svg" alt="DeepSeek">
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED.svg" alt="Docker">
  <img src="https://img.shields.io/badge/Eval_Pass_Rate-90.0%25-brightgreen.svg" alt="Pass Rate">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

---

## 📌 项目定位

`smart-cs-agent` 是一套面向电商平台与企业客户服务真实业务场景的**生产级轻量智能售后 Agent**。系统深度贯穿 **「意图路由识别 $\to$ 订单/物流核验 $\to$ 售后政策条款检索 $\to$ 时间与状态边界推理 $\to$ 售后工单创建与人机协同流转」** 的完整业务闭环。

### 💡 核心设计哲学
在当前大模型应用开发泛滥依赖重型框架（如 LangChain、LlamaIndex 等）的背景下，本项目坚持**生产透明与工程可控原则**：
- **拒绝框架黑盒**：不引入 LangChain、LlamaIndex 或复杂 ORM（SQLAlchemy），业务逻辑直白透明，方便技术面试逐行追问与生产故障排查定位。
- **真实业务读写**：摒弃单纯问答玩具（Toy Demo），基于原生 SQLite 参数化事务实现订单底库核验与工单状态持久化流转。
- **白盒安全审计**：每次思考、Function Calling 工具分发、参数校验、溯源引用（Citations）与毫秒级延迟均被完整结构化沉淀。

---

## 🚀 核心工程亮点

1. **Agentic RAG 与 BM25 词元覆盖度降噪**
   - 知识库按 Markdown 二级/三级标题进行细粒度语义物理切块；
   - 采用 `jieba.cut_for_search` 搜索引擎模式分词，结合 `rank_bm25 (BM25Okapi)` 算法建立内存倒排索引；
   - **创新引入词元覆盖度校验（Token Overlap Ratio）与动态分数阈值**：对客问答时杜绝跨品类误召回与低置信度虚假条款注入，保障回答精准可靠。

2. **真实业务读写闭环与时间边界推理**
   - 针对售后核心场景（7天无理由退货、15天换货、1-2年质保期、物流催单、发票重开等）建立严密的日期时效与状态机判定；
   - 针对**拆封损毁、个性化刻字定制商品、大促秒杀价保排除**等特殊边界条件执行刚性约束，杜绝客服胡乱承诺。

3. **人机协同工单状态流转（Human-in-the-loop）**
   - 原生 SQLite 事务保障工单生命周期：`待处理` $\to$ `处理中` $\to$ `已完结 / 已拒绝`；
   - 工具端防重设计：支持对客反问补充信息、前置订单真实性校验，工单建立后生成唯一追踪号 `TKT{YYYYMMDD}-{HEX}`。

4. **白盒全链路追踪与调试工作台**
   - 自带原生轻量级可视化单页应用（SPA），直连 `/api/chat`、`/api/orders`、`/api/tickets` 与 `/api/knowledge/search`；
   - 对话过程实时渲染推理工具链（`tools_called`）、引用条款来源（`citations`）、端到端耗时（`latency_ms`）。

5. **轻量级内存滑动窗口限流防刷**
   - 纯 Python 标准库实现 60 秒内存滑动窗口限流，默认限制单 IP 每分钟最多 15 次请求（`RATE_LIMIT_PER_MINUTE` 可配置）；
   - 自动解析 `X-Forwarded-For` 反向代理真实 IP，超限返回 HTTP `429 Too Many Requests` 及友好重试指引，防止公网部署 API Key 额度被刷爆。

---

## 🏗️ 系统整体架构

```text
               +-------------------------------------------------------+
               |           Client Layer (Browser / Webhook)            |
               |       可视化售后工作台 / HTTP API 客户端 / 自动化评测集      |
               +---------------------------+---------------------------+
                                           | HTTP /api/*
                                           v
+-----------------------------------------------------------------------------------------+
|                               FastAPI Application Gateway                               |
|  +-----------------------------------------------------------------------------------+  |
|  | [Core Limiter] 内存滑动窗口防刷限流器 (60s 窗口 / X-Forwarded-For 透传识别 / HTTP 429) |  |
|  +-----------------------------------------------------------------------------------+  |
|          |                                   |                                |         |
|          v                                   v                                v         |
|   /api/orders (查询)                  /api/chat (核心对话)             /api/tickets (工单) |
+----------------------------------------------+------------------------------------------+
                                               |
                                               v
+-----------------------------------------------------------------------------------------+
|                                Agent Core Orchestrator                                  |
|  +-----------------------------------------------------------------------------------+  |
|  | System Prompt 守则注入 (对客话术规范 / 越狱与注入防御 / 防幻觉约束 / 业务裁决准则)      |  |
|  | 会话历史滑动窗口加载 (chat_logs 最近 6 轮白盒追踪)                                 |  |
|  | OpenAI 兼容协议驱动 (DeepSeek-V3 / Qwen / GPT-4o 零成本热切换)                        |  |
|  +-------------------------------------------+---------------------------------------+  |
|                                              | Function Calling (tool_calls 调度循环)    |
|                                              v                                          |
|                          +---------------------------------------+                      |
|                          |           Tools Registry              |                      |
|                          +-------------------+-------------------+                      |
|                                              |                                          |
|                     +------------------------+-----------------------+                  |
|                     |                                                |                  |
|                     v                                                v                  |
|      [search_knowledge_base]                          [query_order / tickets]           |
|   +------------------------------------+           +---------------------------------+  |
|   | Local Agentic RAG Engine           |           | SQLite Native Storage           |  |
|   | - Markdown 标题物理分块            |           | - 参数化原生 SQL 防注入         |  |
|   | - Jieba 搜索引擎分词               |           | - orders 订单数据 (500+ 底库)   |  |
|   | - Rank-BM25 (BM25Okapi) 排序       |           | - tickets 售后工单流转记录      |  |
|   | - 词元覆盖度校验 (Overlap Ratio)   |           | - chat_logs 全链路白盒审计      |  |
|   +------------------------------------+           +---------------------------------+  |
+-----------------------------------------------------------------------------------------+
```

---

## 🛠️ 技术栈选型

| 层次 / 模块 | 技术选型 | 版本 / 说明 | 选型考量与原则 |
| :--- | :--- | :--- | :--- |
| **开发语言** | Python | 3.11+ | 现代化类型注解，原生高效性能与异步支持 |
| **Web 框架** | FastAPI | $\ge 0.110.0$ | 极速异步接口响应，原生 Pydantic v2 参数校验与 Swagger 文档 |
| **应用服务器**| Uvicorn | $\ge 0.28.0$ | 高性能 ASGI 生产服务器 |
| **数据持久化**| SQLite 3 | 标准库 | 原生 SQL 参数化查询防注入，零独立进程运维成本，数据操作透明易读 |
| **大模型 SDK**| OpenAI Python | $\ge 1.14.0$ | 标准化协议，无缝接入 DeepSeek、通义千问、MiniMax 等国内主流模型 |
| **全文检索** | rank-bm25 + jieba | $\ge 0.2.2$ / $\ge 0.42.1$ | 纯本地快速分词与倒排索引，无向量模型依赖与显存开销，轻量可解释 |
| **容器化方案**| Docker + Compose | 生产级 Multi-Stage / Compose | 镜像基于 `python:3.11-slim`，无缓存瘦身，数据持久卷挂载 |
| **接口限流** | 自研滑动窗口限流 | 纯 Python 标准库 | 基于 `collections.defaultdict` + `threading.Lock`，无外部 Redis 依赖 |

---

## 📊 自动化业务评测与性能基准

项目集成自主研发的端到端并发评测套件（`eval/run_eval.py`），基于 **500 条模拟真实电商订单底库** 与 **100 条多场景复合测试集** 进行了 4 线程高并发严格压测。

### 1. 核心指标概览

> 数据来源：最新端到端自动化并发评测报告（详见 `eval/eval_report.md`）

| 评测维度 | 实测数据 | 工业级达标基线 | 判定结论 |
| :--- | :---: | :---: | :---: |
| **综合用例通过率 (Overall Pass Rate)** | **90.0%** (90/100) | $\ge 90.0\%$ | ✅ **PASS 达标** |
| **工具调用路由准确率 (Tool Accuracy)** | **94.0%** (94/100) | $\ge 90.0\%$ | ✅ **优秀** |
| **防幻觉与对客安全率 (Safety Rate)** | **97.0%** (97/100) | $\ge 95.0\%$ | ✅ **优秀** |
| **平均端到端耗时 (Avg Latency)** | **1763 ms** | $\le 4000\text{ ms}$ | ✅ **表现优异** |
| **P95 响应耗时 (P95 Latency)** | **2873 ms** | $\le 6500\text{ ms}$ | ✅ **稳定可控** |

### 2. 五大业务场景模块通过率明细

```text
1. 意图路由与闲聊   [====================] 100.0% (20/20)  Avg: 1230ms
2. 政策检索与知识库 [===================.]  95.0% (19/20)  Avg: 2168ms
3. 时间状态边界推理 [==================..]  90.0% (18/20)  Avg: 2166ms
4. 工单工作流与闭环 [================....]  80.0% (16/20)  Avg: 2042ms
5. 安全边界与防越狱 [=================...]  85.0% (17/20)  Avg: 1209ms
```

| 场景分类模块 | 用例数 | 通过数 | 通过率 | 平均耗时 | 考察重点与判定要点 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **1. 意图路由与闲聊** | 20 | 20 | **100.0%** | 1230 ms | 准确识别日常问候、辱骂情绪安抚，订单/工单查询直调工具，无废话 |
| **2. 政策检索与知识库** | 20 | 19 | **95.0%** | 2168 ms | 退换货运费承担、换新期限、发票开具时效准确检索，引用条例合规 |
| **3. 时间与状态边界推理** | 20 | 18 | **90.0%** | 2166 ms | 结合签收日期动态推算「7天退货 / 15天换货 / 质保期 / 刻字不退」边界 |
| **4. 工单工作流与决策闭环** | 20 | 16 | **80.0%** | 2042 ms | 缺参主动追问，证据充分直接建单；非退换货范围礼貌拒单并讲明原因 |
| **5. 安全边界与防幻觉** | 20 | 17 | **85.0%** | 1209 ms | 抵御最高管理员越狱、Prompt 泄漏刺探、非法私下改地址、SQL 注入攻击 |

---

## ⚡ 快速开始与部署

### 方式一：本地极速启动（开发者模式）

#### 1. 克隆代码与配置虚拟环境
```bash
git clone https://github.com/your-username/smart-cs-agent.git
cd smart-cs-agent

# 创建并激活 Python 3.11+ 虚拟环境
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate

# 安装项目依赖
pip install -r requirements.txt
```

#### 2. 配置环境变量
复制配置模板并填入您的真实模型 API Key：
```bash
# Windows (PowerShell)
Copy-Item .env.example .env
# Linux / macOS
cp .env.example .env
```
在 `.env` 中填入你的配置：
```ini
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat
SQLITE_DB_PATH=./data/smart_cs.db
RATE_LIMIT_PER_MINUTE=15
```

#### 3. 启动服务与访问
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
服务就绪后即可访问：
- 🖥️ **智能售后工作台**：[http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- 📖 **Swagger 交互式文档**：[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- 🩺 **健康检查探测接口**：[http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

### 方式二：Docker & Compose 一键容器化部署（生产模式）

无需在宿主机安装 Python 环境，开箱即用：

```bash
# 1. 确保已配置好 .env
# 2. 一键构建并以后台守护模式启动服务
docker compose up -d --build

# 查看运行状态与健康检查
docker compose ps

# 实时查看服务运行日志
docker compose logs -f smart-cs-agent
```
> **持久化保障**：`docker-compose.yml` 已经默认将宿主机的 `./data` 目录挂载至容器内部 `/app/data`，即便容器重启或镜像升级，所有的历史工单、订单与会话记录均完好保留。

---

### 方式三：运行 100 题自动化端到端并发评测

针对上线前的功能完备性、工具准确率与防刷稳定性进行一键回归压测：

```bash
# 本地压测前建议调高单 IP 限流阈值（如设为 1000），避免压测请求被限流拦截
# Windows (PowerShell)
$env:RATE_LIMIT_PER_MINUTE="1000"; python eval/run_eval.py --workers 4

# Linux / Bash
RATE_LIMIT_PER_MINUTE=1000 python eval/run_eval.py --workers 4
```
评测完成后将在控制台输出统计图表，并在 `eval/eval_report.md` 生成包含全部用例执行明细与 Bad Case 归因的完整分析报告。

---

## 📸 界面演示与体验

> 💡 *项目内置开箱即用的前端交互工作台（SPA），支持直观体验订单状态穿透、政策 RAG 检索核验与工单状态实时流转。*

| 售后工单决策流转演示 | 知识库检索与全链路白盒追踪 |
| :---: | :---: |
| *(工作台演示截图预留位置：`docs/images/demo_chat.png`)* | *(全链路调用与追踪截图预留位置：`docs/images/demo_trace.png`)* |

🌐 **公网在线体验 Demo**：`https://smart-cs.your-domain.com` *(公网部署链接预留位)*

---

## 📂 项目工程目录规范

```text
smart-cs-agent/
├── app/                          # 核心业务应用代码
│   ├── main.py                   # FastAPI 应用入口与生命周期管理
│   ├── agent/                    # Agent 核心推理层
│   │   ├── core.py               # Function Calling 循环与会话编排
│   │   ├── prompts.py            # 对客话术、安全防御与系统指令提示词
│   │   ├── tools.py              # 工具注册定义与执行分发器
│   │   └── knowledge.py          # 基于 BM25 与分词的轻量本地 RAG 检索器
│   ├── core/                     # 生产级核心组件
│   │   └── limiter.py            # 纯标准库滑动窗口防刷限流器 (Rate Limiter)
│   ├── db/                       # 数据底座层 (纯原生 SQL)
│   │   ├── database.py           # SQLite 连接上下文池与建表 DDL
│   │   └── seed.py               # 500+ 真实/模拟订单数据补录
│   ├── routers/                  # API 路由分发层
│   │   ├── chat.py               # POST /api/chat (挂载滑动窗口限流)
│   │   ├── orders.py             # 订单详情与物流查询
│   │   ├── tickets.py            # 工单创建、状态流转与列表查询
│   │   └── knowledge.py          # 知识库检索独立调试接口
│   ├── schemas/                  # Pydantic v2 入参与出参类型定义
│   └── static/                   # 售后工作台前端静态单页应用
├── data/                         # 持久化数据与业务资产
│   ├── knowledge/                # 售后政策 Markdown 原文 (退换货/保修/价保/发票)
│   └── smart_cs.db               # SQLite 数据库持久化文件 (挂载卷)
├── eval/                         # 自动化评测套件
│   ├── run_eval.py               # 多线程并发评测驱动脚本
│   ├── cases/                    # 100 道分类业务评测用例集 (5 大模块)
│   └── eval_report.md            # 自动生成的端到端压测详尽分析报告
├── Dockerfile                    # 生产容器化构建定义 (python:3.11-slim)
├── docker-compose.yml            # 容器编排与数据卷持久化配置
├── .dockerignore                 # 容器构建上下文隔离过滤配置
├── .gitignore                    # Git 版本控制忽略清单
├── .env.example                  # 环境变量参考模版 (严禁提交敏感 Key)
├── requirements.txt              # 生产核心依赖清单
└── README.md                     # 项目技术架构与使用说明文档
```

---

## 📄 开源许可证

本项目基于 [MIT License](LICENSE) 开源。欢迎交流、学习与技术探讨！
