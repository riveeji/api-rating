from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_eval.models import (
    AgentConfig,
    AnswerFormat,
    FaultInjection,
    FaultMode,
    StrategyType,
    TaskCategory,
    TaskSpec,
    ValidationMode,
    ValidationSpec,
)
from agent_eval.utils import slugify


def parse_corpus_markdown(corpus_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    documents: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    for path in sorted(corpus_dir.glob("*.md")):
        lines = path.read_text(encoding="utf-8").splitlines()
        title = lines[0].removeprefix("# ").strip()
        metadata: dict[str, str] = {}
        body_lines: list[str] = []
        section_heading: str | None = None
        section_lines: list[str] = []
        order = 0

        def flush_section() -> None:
            nonlocal order, section_heading, section_lines
            if not section_heading:
                return
            order += 1
            chunks.append(
                {
                    "chunk_id": f"{path.stem.replace('_', '-')}-{slugify(section_heading)}",
                    "doc_id": path.stem.replace("_", "-"),
                    "heading": section_heading,
                    "body": "\n".join(section_lines).strip(),
                    "chunk_order": order,
                }
            )
            section_heading = None
            section_lines = []

        for line in lines[1:]:
            if line.startswith("## "):
                flush_section()
                section_heading = line.removeprefix("## ").strip()
                continue
            if section_heading is None and ": " in line:
                key, value = line.split(": ", 1)
                metadata[key.strip().lower().replace("-", "_")] = value.strip()
            elif section_heading is None:
                body_lines.append(line)
            else:
                section_lines.append(line)
        flush_section()
        documents.append(
            {
                "doc_id": path.stem.replace("_", "-"),
                "title": title,
                "source_url": metadata.get("source", ""),
                "source_label": metadata.get("source", ""),
                "reference_date": metadata.get("reference_date", ""),
                "body": "\n".join(body_lines).strip(),
            }
        )
    return documents, chunks


FACTS: list[dict[str, Any]] = [
    {
        "chunk_id": "fastapi-response-model",
        "answer": "Use response_model to validate, serialize, and filter the response payload.",
        "query": "FastAPI response model filter returned fields",
        "accepted_answers": ["response_model", "response model"],
        "required_term_groups": [
            ["response_model", "response model"],
            ["validate", "validation"],
            ["filter", "filters", "filtered"],
        ],
        "prompts": [
            "In FastAPI, which feature validates and filters the returned payload before the response is sent?",
            "You need to hide internal fields from a FastAPI response while keeping the route return value richer internally. What should you use?",
        ],
    },
    {
        "chunk_id": "fastapi-dependencies",
        "answer": "Use Depends() to reuse request-scoped logic such as auth context, database handles, or shared settings.",
        "query": "FastAPI Depends reused auth context",
        "accepted_answers": ["depends", "depends()"],
        "required_term_groups": [
            ["depends", "depends()"],
            ["request-scoped logic", "shared logic", "auth context", "database handles", "shared settings"],
        ],
        "prompts": [
            "Which FastAPI mechanism lets multiple routes reuse auth context or other request-scoped logic?",
            "What should you declare in FastAPI when several endpoints need the same auth or database setup logic?",
        ],
    },
    {
        "chunk_id": "fastapi-background-tasks",
        "answer": "Use BackgroundTasks for short follow-up work that should run after the response returns.",
        "query": "FastAPI background tasks after response",
        "accepted_answers": ["backgroundtasks", "background tasks"],
        "required_term_groups": [
            ["backgroundtasks", "background tasks"],
            ["after the response", "after response", "after the response returns", "follow-up work"],
        ],
        "prompts": [
            "Which FastAPI feature is appropriate for audit logging or email notifications that should happen after the response is returned?",
            "How should a FastAPI route schedule short non-critical follow-up work without blocking the request?",
        ],
    },
    {
        "chunk_id": "fastapi-api-router",
        "answer": "Use APIRouter to split a FastAPI service into smaller route modules with shared prefixes or dependencies.",
        "query": "FastAPI APIRouter organize larger service",
        "accepted_answers": ["apirouter", "api router"],
        "required_term_groups": [
            ["apirouter", "api router"],
            ["split", "modular", "modules", "route modules"],
        ],
        "prompts": [
            "What FastAPI feature helps split a larger service into modular route groups with shared prefixes or tags?",
            "How do you keep a growing FastAPI codebase from putting every endpoint in one file?",
        ],
    },
    {
        "chunk_id": "fastapi-templates-and-static",
        "answer": "Use Jinja2Templates for server-rendered HTML and StaticFiles for mounted assets.",
        "query": "FastAPI Jinja2Templates StaticFiles dashboard",
        "accepted_answers": ["jinja2templates and staticfiles", "jinja2 templates and static files"],
        "required_term_groups": [
            ["jinja2templates", "jinja2 templates"],
            ["staticfiles", "static files"],
        ],
        "prompts": [
            "How can FastAPI serve a lightweight dashboard without introducing a separate frontend framework?",
            "Which FastAPI components are used for server-rendered HTML pages and mounted static assets?",
        ],
    },
    {
        "chunk_id": "fastapi-test-client",
        "answer": "Use TestClient to issue requests against the FastAPI ASGI app in tests.",
        "query": "FastAPI TestClient route tests",
        "accepted_answers": ["testclient", "test client"],
        "required_term_groups": [
            ["testclient", "test client"],
        ],
        "prompts": [
            "Which FastAPI testing helper lets you call the ASGI app without starting a live server?",
        ],
    },
    {
        "chunk_id": "sqlite-parameter-binding",
        "answer": "Use parameter binding with placeholders instead of string concatenation to avoid SQL injection.",
        "query": "SQLite parameter binding placeholders avoid sql injection",
        "accepted_answers": ["parameter binding", "placeholders"],
        "required_term_groups": [
            ["parameter binding", "placeholders", "placeholder"],
            ["sql injection"],
        ],
        "prompts": [
            "What is the recommended SQLite pattern for safely inserting external values into a query?",
            "How should an app pass user values into SQLite to avoid SQL injection?",
        ],
    },
    {
        "chunk_id": "sqlite-wal-mode",
        "answer": "Enable WAL mode when you need readers to continue while a writer appends to the log.",
        "query": "SQLite WAL mode readers writer concurrency",
        "accepted_answers": ["wal mode", "write-ahead logging", "wal"],
        "required_term_groups": [
            ["wal", "write-ahead logging", "wal mode"],
            ["reader", "readers"],
            ["writer", "writes"],
        ],
        "prompts": [
            "What SQLite mode improves reader and writer concurrency by appending writes to a WAL file?",
            "Which SQLite setting is useful for frequent reads with occasional writes because readers can continue during writes?",
        ],
    },
    {
        "chunk_id": "sqlite-fts5",
        "answer": "Use SQLite FTS5 with MATCH queries for compact local full-text search over document chunks.",
        "query": "SQLite FTS5 MATCH bm25 document search",
        "accepted_answers": ["fts5"],
        "required_term_groups": [
            ["fts5"],
            ["match"],
            ["full-text search", "text search", "document chunks", "chunk retrieval"],
        ],
        "prompts": [
            "Which SQLite extension is appropriate for compact local full-text search over document chunks?",
            "What SQLite feature supports MATCH-based text search and ranking helpers such as bm25?",
        ],
    },
    {
        "chunk_id": "sqlite-busy-timeout",
        "answer": "Use busy_timeout to wait briefly on a lock instead of failing immediately.",
        "query": "SQLite busy_timeout lock contention",
        "accepted_answers": ["busy_timeout", "busy timeout"],
        "required_term_groups": [
            ["busy_timeout", "busy timeout"],
            ["wait"],
            ["lock", "locked"],
        ],
        "prompts": [
            "What SQLite setting can wait briefly when the database is locked instead of failing at once?",
            "Which SQLite option helps mitigate short lock contention in lightweight apps?",
        ],
    },
    {
        "chunk_id": "qwen-dashscope-function-calling",
        "answer": "DashScope function calling uses the tools parameter with function names, descriptions, and JSON Schema arguments.",
        "query": "DashScope function calling tools parameter json schema",
        "accepted_answers": ["tools parameter", "tools"],
        "required_term_groups": [
            ["tools", "tools parameter"],
            ["json schema", "schema"],
            ["function", "function calling"],
        ],
        "prompts": [
            "How does DashScope describe callable tools for Qwen function calling?",
            "What should a DashScope tool definition contain so the model can choose and call the right function?",
        ],
    },
    {
        "chunk_id": "qwen-dashscope-tool-choice",
        "answer": "Use tool_choice to let the model choose automatically, disable tools, or force a specific function.",
        "query": "DashScope tool_choice auto none force function",
        "accepted_answers": ["tool_choice", "tool choice"],
        "required_term_groups": [
            ["tool_choice", "tool choice"],
            ["auto", "none", "specific function", "force"],
        ],
        "prompts": [
            "Which DashScope parameter controls whether Qwen selects tools automatically, disables tools, or is forced to use one function?",
            "How can a DashScope client force the agent onto one known tool path instead of letting the model decide freely?",
        ],
    },
    {
        "chunk_id": "qwen-dashscope-validate-arguments",
        "answer": "Validate tool arguments before execution because model-generated arguments are untrusted input.",
        "query": "DashScope validate tool arguments before execution",
        "accepted_answers": ["validate tool arguments", "tool arguments"],
        "required_term_groups": [
            ["validate", "validated"],
            ["tool arguments", "arguments"],
            ["untrusted input", "before execution", "execute"],
        ],
        "prompts": [
            "What should an agent runner do with model-generated tool arguments before executing the tool?",
            "Why should DashScope tool arguments be treated as untrusted input?",
        ],
    },
]


SERVICES: list[dict[str, str]] = [
    {
        "service": "docs-gateway",
        "owner_team": "platform-api",
        "runbook_url": "https://internal.example/runbooks/docs-gateway",
        "fallback_policy": "Reuse a shared dependency for auth context and filter the public payload with response_model.",
    },
    {
        "service": "notification-worker",
        "owner_team": "workflow-core",
        "runbook_url": "https://internal.example/runbooks/notification-worker",
        "fallback_policy": "Move short follow-up actions into BackgroundTasks and keep the request path small.",
    },
    {
        "service": "kb-search",
        "owner_team": "search-infra",
        "runbook_url": "https://internal.example/runbooks/kb-search",
        "fallback_policy": "Use SQLite FTS5 for chunk retrieval and avoid string-built SQL.",
    },
    {
        "service": "traffic-dashboard",
        "owner_team": "data-services",
        "runbook_url": "https://internal.example/runbooks/traffic-dashboard",
        "fallback_policy": "Use WAL mode for read-heavy concurrency and keep lock contention short.",
    },
    {
        "service": "agent-orchestrator",
        "owner_team": "agent-platform",
        "runbook_url": "https://internal.example/runbooks/agent-orchestrator",
        "fallback_policy": "Keep tool_choice constrained when needed and validate tool arguments before execution.",
    },
]


CASES: list[dict[str, Any]] = [
    {
        "case_id": "CASE-001",
        "service": "docs-gateway",
        "title": "Hide internal fields and share auth context",
        "region": "cn-hangzhou",
        "priority": "high",
        "target_rps": 1200,
        "worker_count": 3,
        "primary_chunk_id": "fastapi-response-model",
        "supporting_chunk_id": "fastapi-dependencies",
        "recommended_feature": "response_model",
        "notes": "The public API must hide internal fields and reuse auth context across routes.",
    },
    {
        "case_id": "CASE-002",
        "service": "notification-worker",
        "title": "Send follow-up email without blocking the request",
        "region": "cn-shanghai",
        "priority": "medium",
        "target_rps": 900,
        "worker_count": 3,
        "primary_chunk_id": "fastapi-background-tasks",
        "supporting_chunk_id": "fastapi-api-router",
        "recommended_feature": "background_tasks",
        "notes": "The route should return quickly while a short follow-up notification happens afterwards.",
    },
    {
        "case_id": "CASE-003",
        "service": "kb-search",
        "title": "Build a local retriever for support docs",
        "region": "ap-southeast-1",
        "priority": "high",
        "target_rps": 600,
        "worker_count": 2,
        "primary_chunk_id": "sqlite-fts5",
        "supporting_chunk_id": "sqlite-parameter-binding",
        "recommended_feature": "fts5",
        "notes": "The team needs compact local search and wants safe parameterized SQL around metadata lookups.",
    },
    {
        "case_id": "CASE-004",
        "service": "traffic-dashboard",
        "title": "Reduce writer impact on read-heavy traffic views",
        "region": "us-west-1",
        "priority": "high",
        "target_rps": 1400,
        "worker_count": 4,
        "primary_chunk_id": "sqlite-wal-mode",
        "supporting_chunk_id": "sqlite-busy-timeout",
        "recommended_feature": "wal_mode",
        "notes": "Read traffic is heavy and brief lock contention should not fail the dashboard immediately.",
    },
    {
        "case_id": "CASE-005",
        "service": "agent-orchestrator",
        "title": "Keep tool calling reliable under agent retries",
        "region": "eu-central-1",
        "priority": "critical",
        "target_rps": 300,
        "worker_count": 2,
        "primary_chunk_id": "qwen-dashscope-function-calling",
        "supporting_chunk_id": "qwen-dashscope-validate-arguments",
        "recommended_feature": "function_calling",
        "notes": "The agent must describe tools well, constrain tool choice when needed, and validate arguments safely.",
    },
]


def _service_value(service_name: str, field: str) -> Any:
    return next(service[field] for service in SERVICES if service["service"] == service_name)


def _supporting_pattern(chunk_id: str) -> str:
    return chunk_id.split("-", 1)[1].replace("-", "_")


def build_single_hop_tasks() -> list[TaskSpec]:
    tasks: list[TaskSpec] = []
    counter = 1
    for fact in FACTS:
        for variant_index, prompt in enumerate(fact["prompts"], start=1):
            tasks.append(
                TaskSpec(
                    task_id=f"TASK-SH-{counter:03d}",
                    title=f"Single-hop knowledge check {counter}",
                    category=TaskCategory.SINGLE_HOP,
                    prompt=prompt,
                    answer_format=AnswerFormat.TEXT,
                    expected_answer=fact["answer"],
                    required_tools=["doc_search", "doc_read"],
                    reference_chunk_ids=[fact["chunk_id"]],
                    search_hints=[fact["query"], fact["chunk_id"].replace("-", " ")],
                    validation=ValidationSpec(
                        mode=ValidationMode.EXACT_TEXT,
                        citation_chunk_ids=[fact["chunk_id"]],
                        accepted_answers=[fact["answer"], *fact.get("accepted_answers", [])],
                        required_term_groups=fact.get("required_term_groups", []),
                    ),
                    metadata={
                        "canonical_answer": fact["answer"],
                        "primary_query": fact["query"],
                        "fact_variant": variant_index,
                    },
                )
            )
            counter += 1
    return tasks


def build_multi_step_tasks() -> list[TaskSpec]:
    tasks: list[TaskSpec] = []
    counter = 1
    for case in CASES:
        owner_sql = (
            "SELECT owner_team, runbook_url, fallback_policy "
            f"FROM service_catalog WHERE service = '{case['service']}'"
        )
        per_worker_rps = round(case["target_rps"] / case["worker_count"], 2)
        canonical_summary = {
            "case_id": case["case_id"],
            "owner_team": _service_value(case["service"], "owner_team"),
            "recommended_feature": case["recommended_feature"],
            "per_worker_rps": per_worker_rps,
        }
        tasks.append(
            TaskSpec(
                task_id=f"TASK-MS-{counter:03d}",
                title=f"{case['case_id']} ownership summary",
                category=TaskCategory.MULTI_STEP,
                prompt=(
                    f"For {case['case_id']}, use the case API, SQL, document tools, and calculator. "
                    "Return JSON with case_id, owner_team, recommended_feature, and per_worker_rps."
                ),
                answer_format=AnswerFormat.JSON,
                expected_answer=canonical_summary,
                required_tools=["case_api", "sql_query", "doc_search", "doc_read", "calculator"],
                reference_chunk_ids=[case["primary_chunk_id"]],
                search_hints=[
                    case["recommended_feature"].replace("_", " "),
                    case["primary_chunk_id"].replace("-", " "),
                ],
                validation=ValidationSpec(
                    mode=ValidationMode.FIELD_MATCH,
                    required_fields=list(canonical_summary.keys()),
                    citation_chunk_ids=[case["primary_chunk_id"]],
                ),
                metadata={
                    "case_id": case["case_id"],
                    "sql_query": owner_sql,
                    "calculation": f"{case['target_rps']} / {case['worker_count']}",
                    "canonical_answer": canonical_summary,
                },
            )
        )
        counter += 1

        canonical_runbook = {
            "case_id": case["case_id"],
            "runbook_url": _service_value(case["service"], "runbook_url"),
            "fallback_policy": _service_value(case["service"], "fallback_policy"),
            "recommended_feature": case["recommended_feature"],
        }
        tasks.append(
            TaskSpec(
                task_id=f"TASK-MS-{counter:03d}",
                title=f"{case['case_id']} runbook mapping",
                category=TaskCategory.MULTI_STEP,
                prompt=(
                    f"For {case['case_id']}, look up the service owner in SQL and ground the recommendation in the docs. "
                    "Return JSON with case_id, runbook_url, fallback_policy, and recommended_feature."
                ),
                answer_format=AnswerFormat.JSON,
                expected_answer=canonical_runbook,
                required_tools=["case_api", "sql_query", "doc_search", "doc_read"],
                reference_chunk_ids=[case["primary_chunk_id"], case["supporting_chunk_id"]],
                search_hints=[
                    case["recommended_feature"].replace("_", " "),
                    case["supporting_chunk_id"].replace("-", " "),
                ],
                validation=ValidationSpec(
                    mode=ValidationMode.FIELD_MATCH,
                    required_fields=list(canonical_runbook.keys()),
                    citation_chunk_ids=[case["primary_chunk_id"]],
                ),
                metadata={
                    "case_id": case["case_id"],
                    "sql_query": owner_sql,
                    "canonical_answer": canonical_runbook,
                },
            )
        )
        counter += 1

        canonical_ops = {
            "case_id": case["case_id"],
            "region": case["region"],
            "priority": case["priority"],
            "recommended_feature": case["recommended_feature"],
            "supporting_pattern": _supporting_pattern(case["supporting_chunk_id"]),
        }
        tasks.append(
            TaskSpec(
                task_id=f"TASK-MS-{counter:03d}",
                title=f"{case['case_id']} operations brief",
                category=TaskCategory.MULTI_STEP,
                prompt=(
                    f"For {case['case_id']}, use the case API and document tools to return JSON with "
                    "case_id, region, priority, recommended_feature, and supporting_pattern."
                ),
                answer_format=AnswerFormat.JSON,
                expected_answer=canonical_ops,
                required_tools=["case_api", "doc_search", "doc_read"],
                reference_chunk_ids=[case["primary_chunk_id"], case["supporting_chunk_id"]],
                search_hints=[
                    case["recommended_feature"].replace("_", " "),
                    case["supporting_chunk_id"].replace("-", " "),
                ],
                validation=ValidationSpec(
                    mode=ValidationMode.FIELD_MATCH,
                    required_fields=list(canonical_ops.keys()),
                    citation_chunk_ids=[case["primary_chunk_id"], case["supporting_chunk_id"]],
                ),
                metadata={
                    "case_id": case["case_id"],
                    "secondary_query": case["supporting_chunk_id"].replace("-", " "),
                    "canonical_answer": canonical_ops,
                },
            )
        )
        counter += 1
    return tasks


def build_recovery_tasks() -> list[TaskSpec]:
    tasks: list[TaskSpec] = []
    counter = 1
    for case in CASES:
        search_answer = {
            "case_id": case["case_id"],
            "recommended_feature": case["recommended_feature"],
            "owner_team": _service_value(case["service"], "owner_team"),
        }
        tasks.append(
            TaskSpec(
                task_id=f"TASK-RC-{counter:03d}",
                title=f"{case['case_id']} recovery after empty search",
                category=TaskCategory.RECOVERY,
                prompt=(
                    f"For {case['case_id']}, recover from a failed first retrieval and return JSON with "
                    "case_id, recommended_feature, and owner_team."
                ),
                answer_format=AnswerFormat.JSON,
                expected_answer=search_answer,
                required_tools=["case_api", "sql_query", "doc_search", "doc_read"],
                reference_chunk_ids=[case["primary_chunk_id"]],
                search_hints=["nonexistent retrieval phrase", case["recommended_feature"].replace("_", " ")],
                validation=ValidationSpec(
                    mode=ValidationMode.FIELD_MATCH,
                    required_fields=list(search_answer.keys()),
                    citation_chunk_ids=[case["primary_chunk_id"]],
                ),
                faults=[FaultInjection(tool_name="doc_search", mode=FaultMode.EMPTY_ONCE)],
                metadata={
                    "case_id": case["case_id"],
                    "sql_query": f"SELECT owner_team FROM service_catalog WHERE service = '{case['service']}'",
                    "canonical_answer": search_answer,
                },
            )
        )
        counter += 1

        api_answer = {
            "case_id": case["case_id"],
            "priority": case["priority"],
            "recommended_feature": case["recommended_feature"],
            "per_worker_rps": round(case["target_rps"] / case["worker_count"], 2),
        }
        tasks.append(
            TaskSpec(
                task_id=f"TASK-RC-{counter:03d}",
                title=f"{case['case_id']} recovery after transient API error",
                category=TaskCategory.RECOVERY,
                prompt=(
                    f"For {case['case_id']}, the first case API call may fail. Recover and return JSON with "
                    "case_id, priority, recommended_feature, and per_worker_rps."
                ),
                answer_format=AnswerFormat.JSON,
                expected_answer=api_answer,
                required_tools=["case_api", "doc_search", "doc_read", "calculator"],
                reference_chunk_ids=[case["primary_chunk_id"]],
                search_hints=[case["recommended_feature"].replace("_", " ")],
                validation=ValidationSpec(
                    mode=ValidationMode.FIELD_MATCH,
                    required_fields=list(api_answer.keys()),
                    citation_chunk_ids=[case["primary_chunk_id"]],
                ),
                faults=[FaultInjection(tool_name="case_api", mode=FaultMode.ERROR_ONCE)],
                metadata={
                    "case_id": case["case_id"],
                    "calculation": f"{case['target_rps']} / {case['worker_count']}",
                    "canonical_answer": api_answer,
                },
            )
        )
        counter += 1
    return tasks


def build_agent_configs(
    *,
    include_live_configs: bool = False,
    include_ollama_configs: bool = False,
    planner_model: str | None = None,
    executor_model: str | None = None,
    verifier_model: str | None = None,
) -> list[AgentConfig]:
    configs = [
        AgentConfig(
            config_id="baseline_heuristic",
            display_name="启发式基线",
            strategy=StrategyType.BASELINE,
            model_provider="heuristic",
            max_steps=8,
            allow_replan=False,
            enable_verifier=False,
            enable_web_lookup=False,
            settings={"notes": "单轮反应式策略，不做恢复。"},
        ),
        AgentConfig(
            config_id="planner_heuristic",
            display_name="启发式规划-执行",
            strategy=StrategyType.PLANNER,
            model_provider="heuristic",
            max_steps=8,
            allow_replan=True,
            enable_verifier=False,
            enable_web_lookup=False,
            settings={"notes": "显式规划，并允许一次重规划。"},
        ),
        AgentConfig(
            config_id="verifier_heuristic",
            display_name="启发式规划-执行-校验",
            strategy=StrategyType.VERIFIER,
            model_provider="heuristic",
            max_steps=10,
            allow_replan=True,
            enable_verifier=True,
            enable_web_lookup=False,
            settings={"notes": "增加校验与一次修复循环。"},
        ),
    ]
    if include_live_configs:
        configs.extend(
            [
                AgentConfig(
                    config_id="baseline_qwen_live",
                    display_name="Qwen 实时基线",
                    strategy=StrategyType.BASELINE,
                    model_provider="dashscope",
                    executor_model=executor_model,
                    max_steps=8,
                    enable_web_lookup=False,
                    settings={"notes": "DashScope/Qwen 真实工具调用基线。"},
                ),
                AgentConfig(
                    config_id="planner_qwen_live",
                    display_name="Qwen 实时规划-执行",
                    strategy=StrategyType.PLANNER,
                    model_provider="dashscope",
                    planner_model=planner_model,
                    executor_model=executor_model,
                    max_steps=8,
                    allow_replan=True,
                    enable_web_lookup=False,
                    settings={"notes": "DashScope/Qwen 先规划后执行。"},
                ),
                AgentConfig(
                    config_id="verifier_qwen_live",
                    display_name="Qwen 实时规划-执行-校验",
                    strategy=StrategyType.VERIFIER,
                    model_provider="dashscope",
                    planner_model=planner_model,
                    executor_model=executor_model,
                    verifier_model=verifier_model,
                    max_steps=10,
                    allow_replan=True,
                    enable_verifier=True,
                    enable_web_lookup=False,
                    settings={"notes": "DashScope/Qwen 增加校验与修复。"},
                ),
            ]
        )
    if include_ollama_configs:
        configs.extend(
            [
                AgentConfig(
                    config_id="baseline_ollama_live",
                    display_name="Ollama 实时基线",
                    strategy=StrategyType.BASELINE,
                    model_provider="ollama",
                    executor_model=executor_model,
                    max_steps=8,
                    enable_web_lookup=False,
                    settings={"notes": "Ollama 本地模型实时工具调用基线。"},
                ),
                AgentConfig(
                    config_id="planner_ollama_live",
                    display_name="Ollama 实时规划-执行",
                    strategy=StrategyType.PLANNER,
                    model_provider="ollama",
                    planner_model=planner_model,
                    executor_model=executor_model,
                    max_steps=8,
                    allow_replan=True,
                    enable_web_lookup=False,
                    settings={"notes": "Ollama 本地模型先规划后执行。"},
                ),
                AgentConfig(
                    config_id="verifier_ollama_live",
                    display_name="Ollama 实时规划-执行-校验",
                    strategy=StrategyType.VERIFIER,
                    model_provider="ollama",
                    planner_model=planner_model,
                    executor_model=executor_model,
                    verifier_model=verifier_model,
                    max_steps=10,
                    allow_replan=True,
                    enable_verifier=True,
                    enable_web_lookup=False,
                    settings={"notes": "Ollama 本地模型增加校验与修复。"},
                ),
            ]
        )
    return configs


def build_seed_payload(
    corpus_dir: Path,
    *,
    include_live_configs: bool = False,
    include_ollama_configs: bool = False,
    planner_model: str | None = None,
    executor_model: str | None = None,
    verifier_model: str | None = None,
) -> dict[str, Any]:
    documents, chunks = parse_corpus_markdown(corpus_dir)
    single_hop = build_single_hop_tasks()
    multi_step = build_multi_step_tasks()
    recovery = build_recovery_tasks()
    tasks = single_hop + multi_step + recovery
    assert len(single_hop) == 25, f"Expected 25 single-hop tasks, got {len(single_hop)}"
    assert len(multi_step) == 15, f"Expected 15 multi-step tasks, got {len(multi_step)}"
    assert len(recovery) == 10, f"Expected 10 recovery tasks, got {len(recovery)}"
    return {
        "documents": documents,
        "chunks": chunks,
        "services": SERVICES,
        "cases": CASES,
        "tasks": tasks,
        "agent_configs": build_agent_configs(
            include_live_configs=include_live_configs,
            include_ollama_configs=include_ollama_configs,
            planner_model=planner_model,
            executor_model=executor_model,
            verifier_model=verifier_model,
        ),
    }
