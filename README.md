# LLM Agent Evaluation & Optimization Platform

一个面向企业知识库 Agent 的双层项目：

- 上层是在线知识库助手，面向真实提问、真实会话和带引用回答。
- 下层是离线评测与优化平台，复用同一套工具和 runner 做 benchmark、失败分析和策略对比。

这不是一个普通的聊天 demo，也不是单纯的 RAG 页面。它更接近一个真实 Agent 系统的最小闭环：

1. 用户在在线助手里提问。
2. Agent 调用文档检索、文档读取、SQL、结构化 case API、计算器等工具。
3. 系统记录完整执行轨迹、引用、时延、token 和成本估算。
4. 同一套能力再被离线 benchmark 复用，用于比较不同策略与模型配置。

界面默认使用中文。

## 项目目标

这个项目主要解决两个问题：

1. 如何做一个最小可落地的企业知识库 Agent。
2. 如何用统一实验框架评估和优化 Agent 的效果、稳定性和成本。

相比“做一个 AI 助手”，这个项目强调的是：

- tool calling
- planner / executor / verifier 策略差异
- trajectory logging
- offline evaluation
- failure classification
- recovery / fallback
- latency / token / cost tradeoff

## 当前能力

### 在线层：知识库助手

入口：

- `/assistant`
- `/assistant/{session_id}`

能力：

- 支持真实提问与会话保存
- 支持切换不同 Agent 配置
- 优先使用 live model 配置
- live model 不可用时自动回退到本地确定性检索回答
- 每条回答都保留：
  - 引用的 `chunk_id`
  - 执行轨迹
  - 总时延
  - token 估算
  - 成本估算

### 离线层：评测与优化平台

入口：

- `/`
- `/experiments`
- `/experiments/{experiment_id}`
- `/leaderboard`
- `/runs/{run_id}`
- `/failures`

能力：

- 固定 benchmark 数据集
- 任务组 / 配置组预设切片
- 统一批量实验执行
- 聚合指标统计
- 失败类型分析
- 单条 run 轨迹回放

## Benchmark 设计

### 文档语料

当前内置 3 份本地文档快照：

- `FastAPI`
- `SQLite`
- `DashScope / Qwen`

所有语料都落在：

- [agent_eval/assets/corpus](J:/ide-workspace/新建文件夹/agent_eval/assets/corpus)

### 任务集

共 50 个种子任务：

- 25 个单跳问答任务
- 15 个多步工具调用任务
- 10 个故障恢复任务

任务定义和种子生成位于：

- [agent_eval/seed.py](J:/ide-workspace/新建文件夹/agent_eval/seed.py)

### 工具层

当前支持 6 个工具：

- `doc_search`
- `doc_read`
- `sql_query`
- `case_api`
- `calculator`
- `web_lookup`

实现位于：

- [agent_eval/tools.py](J:/ide-workspace/新建文件夹/agent_eval/tools.py)

### 策略层

支持 3 种核心策略：

- `baseline_tool_calling`
- `planner_executor`
- `planner_executor_verifier`

启发式和实时模型 runner 都在：

- [agent_eval/runners.py](J:/ide-workspace/新建文件夹/agent_eval/runners.py)

## 在线助手怎么工作

在线助手的主逻辑位于：

- [agent_eval/assistant.py](J:/ide-workspace/新建文件夹/agent_eval/assistant.py)

工作流如下：

1. 用户在 `/assistant` 输入问题。
2. 系统根据当前配置选择 `heuristic / Ollama / DashScope`。
3. 如果是 live config，优先走实时 tool-calling runner。
4. 如果 live model 不可用，或输出没有有效引用，则回退到本地确定性检索回答。
5. 会话和消息写入 SQLite。
6. 页面展示答案、引用和完整轨迹。

这意味着项目即使没有实时模型，也能作为一个“可运行的知识库助手”展示；而一旦接入 `qwen3.5:9b` 或 DashScope，就能自然升级成真实 Agent。

## 离线平台怎么工作

离线实验编排位于：

- [agent_eval/experiments.py](J:/ide-workspace/新建文件夹/agent_eval/experiments.py)

流程如下：

1. 解析任务预设与配置预设。
2. 生成一个 experiment。
3. 对选中的 `config x task` 全量运行。
4. 每个 run 结束后写入：
   - run
   - steps
   - tool calls
   - evaluation
5. 聚合 experiment 指标并落库。
6. 页面展示排行榜、失败分布和实验详情。

数据库实现位于：

- [agent_eval/storage.py](J:/ide-workspace/新建文件夹/agent_eval/storage.py)

## 指标体系

每次 run 记录：

- tool 调用顺序
- tool 参数与输出
- step 级 thought / observation
- citations
- total latency
- total tokens
- total cost estimate
- rule-based evaluation

当前核心聚合指标：

- `success_rate`
- `avg_latency_ms`
- `avg_tokens`
- `avg_cost`
- `tool_error_rate`
- `invalid_action_rate`
- `recovery_rate`
- `avg_steps`

失败类型：

- `wrong_retrieval`
- `bad_tool_choice`
- `tool_error_not_recovered`
- `format_violation`
- `hallucinated_answer`

评测器位于：

- [agent_eval/evaluator.py](J:/ide-workspace/新建文件夹/agent_eval/evaluator.py)

## 技术栈

- Python
- FastAPI
- Pydantic
- SQLite + FTS5
- Jinja2 Templates
- Plotly
- Ollama / DashScope OpenAI-compatible API

项目配置位于：

- [pyproject.toml](J:/ide-workspace/新建文件夹/pyproject.toml)
- [agent_eval/config.py](J:/ide-workspace/新建文件夹/agent_eval/config.py)

## 项目结构

```text
agent_eval/
  assets/corpus/        本地文档快照
  assistant.py          在线知识库助手服务
  cli.py                Seed 与 benchmark 命令
  config.py             环境变量配置
  evaluator.py          规则评测器
  experiments.py        实验编排与聚合
  llm.py                DashScope / Ollama OpenAI-compatible 客户端
  models.py             任务、实验、助手、轨迹等数据模型
  presets.py            benchmark 任务组 / 配置组预设
  runners.py            启发式与 live runner
  seed.py               文档、任务、配置种子
  storage.py            SQLite schema 与持久化
  tools.py              工具注册与故障注入
  utils.py              JSON / token / cost / FTS 工具函数
  web.py                FastAPI 页面与接口
  templates/            中文服务端模板
  static/               样式文件
tests/
  test_seed.py
  test_runner.py
  test_web.py
  test_assistant.py
```

## 本地启动

### 1. 安装依赖

```bash
python -m pip install -e .[dev]
```

### 2. 灌种子数据

```bash
python -m agent_eval.cli seed-demo
```

### 3. 跑一轮 benchmark

```bash
python -m agent_eval.cli run-benchmark
```

### 4. 启动 Web

```bash
python -m uvicorn agent_eval.web:app --reload
```

Windows 下推荐从项目目录启动：

```powershell
cd "J:\ide-workspace\新建文件夹"
python -m uvicorn agent_eval.web:app --reload --reload-dir "J:\ide-workspace\新建文件夹\agent_eval"
```

打开：

- `http://127.0.0.1:8000/assistant`
- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/experiments`
- `http://127.0.0.1:8000/leaderboard`
- `http://127.0.0.1:8000/failures`

## CLI 用法

初始化数据库：

```bash
python -m agent_eval.cli init-db
```

重新灌种子：

```bash
python -m agent_eval.cli seed-demo
```

运行完整 benchmark：

```bash
python -m agent_eval.cli run-benchmark
```

按预设切片运行：

```bash
python -m agent_eval.cli run-benchmark --task-preset single_hop --config-preset heuristic --limit 5
python -m agent_eval.cli run-benchmark --task-preset recovery --config-preset ollama_live --limit 3
```

追加显式任务 / 配置：

```bash
python -m agent_eval.cli run-benchmark --task-id TASK-SH-001 --config-id baseline_heuristic
```

支持的任务预设：

- `all`
- `single_hop`
- `multi_step`
- `recovery`

支持的配置预设：

- `all`
- `heuristic`
- `dashscope_live`
- `ollama_live`

## Web 路由

### 在线助手

- `GET /assistant`
- `GET /assistant/{session_id}`
- `POST /assistant/ask`

### 评测平台

- `GET /`
- `POST /experiments/run`
- `GET /experiments`
- `GET /experiments/{experiment_id}`
- `GET /leaderboard`
- `GET /runs/{run_id}`
- `GET /tasks`
- `GET /failures`

HTML 页面支持 `?format=json` 返回 JSON。

## 配置实时模型

### Ollama + qwen3.5:9b

如果你本地已经安装了 Ollama 并拉好了 `qwen3.5:9b`，可以这样启用：

```powershell
$env:AGENT_EVAL_INCLUDE_LIVE_OLLAMA_CONFIGS="true"
$env:AGENT_EVAL_OLLAMA_BASE_URL="http://127.0.0.1:11434/v1"
$env:AGENT_EVAL_OLLAMA_API_KEY="ollama"
$env:AGENT_EVAL_PLANNER_MODEL="qwen3.5:9b"
$env:AGENT_EVAL_EXECUTOR_MODEL="qwen3.5:9b"
$env:AGENT_EVAL_VERIFIER_MODEL="qwen3.5:9b"

python -m agent_eval.cli seed-demo
```

启用后会增加 3 个实时配置：

- `Ollama 实时基线`
- `Ollama 实时规划-执行`
- `Ollama 实时规划-执行-校验`

在线助手会优先选实时配置；如果实时模型调用失败，会自动回退到本地确定性检索回答。

### DashScope

```powershell
$env:AGENT_EVAL_DASHSCOPE_API_KEY="your_key"
$env:AGENT_EVAL_INCLUDE_LIVE_QWEN_CONFIGS="true"
$env:AGENT_EVAL_PLANNER_MODEL="qwen-plus"
$env:AGENT_EVAL_EXECUTOR_MODEL="qwen-plus"
$env:AGENT_EVAL_VERIFIER_MODEL="qwen-max"

python -m agent_eval.cli seed-demo
```

## 当前项目适合怎么讲

最好的讲法不是“我做了一个聊天机器人”，而是：

> 我做了一个企业知识库 Agent，并设计了配套的离线评测与优化平台。在线层负责真实问答、工具调用和带引用回答；离线层负责统一 benchmark、记录轨迹、分析失败并比较不同策略在成功率、时延、成本和恢复能力上的差异。

这个讲法更贴近：

- AI 应用研发
- LLM / Agent 工程
- Agent 优化工程
- AI 平台 / 智能体基础设施

## 已验证内容

当前本地已验证：

- `python -m pytest` 通过
- benchmark 预设可运行
- 中文页面可正常渲染
- `/assistant/ask` 可创建会话并返回带引用回答
- `CASE-001` 场景可结合 case API、SQL 和文档检索生成回答

## 后续建议

如果继续把项目往“更像真实产品”推进，优先级建议是：

1. 把在线会话中的失败样本一键转成 benchmark 候选任务。
2. 接真实文档目录导入，而不是只用种子快照。
3. 增加人工反馈和简单质量标注。
4. 增加实验对比导出页，用于写报告或简历素材。
5. 接权限与用户体系，把助手做成真正的内部工具。

## 开发与测试

运行测试：

```bash
python -m pytest
```

推荐在提交前执行：

```bash
python -m pytest
python -m agent_eval.cli seed-demo
python -m agent_eval.cli run-benchmark --task-preset single_hop --config-preset heuristic --limit 3
```
