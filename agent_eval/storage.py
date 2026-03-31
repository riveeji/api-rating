from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

from agent_eval.models import (
    AgentConfig,
    AssistantMessageRecord,
    AssistantSessionSummary,
    ExperimentSummary,
    FailureRecord,
    LeaderboardEntry,
    RunRecord,
    TaskSpec,
    utc_now,
)
from agent_eval.utils import fts_query, json_dumps, json_loads


class Storage:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 3000")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init_db(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    source_label TEXT,
                    reference_date TEXT,
                    body TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS document_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
                    heading TEXT NOT NULL,
                    body TEXT NOT NULL,
                    chunk_order INTEGER NOT NULL
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
                    chunk_id UNINDEXED,
                    doc_id UNINDEXED,
                    heading,
                    body
                );

                CREATE TABLE IF NOT EXISTS service_catalog (
                    service TEXT PRIMARY KEY,
                    owner_team TEXT NOT NULL,
                    runbook_url TEXT NOT NULL,
                    fallback_policy TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS mock_cases (
                    case_id TEXT PRIMARY KEY,
                    service TEXT NOT NULL REFERENCES service_catalog(service),
                    title TEXT NOT NULL,
                    region TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    target_rps INTEGER NOT NULL,
                    worker_count INTEGER NOT NULL,
                    primary_chunk_id TEXT NOT NULL,
                    supporting_chunk_id TEXT NOT NULL,
                    recommended_feature TEXT NOT NULL,
                    notes TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    category TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS agent_configs (
                    config_id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    config_preset TEXT,
                    task_preset TEXT,
                    config_ids_json TEXT NOT NULL,
                    task_ids_json TEXT NOT NULL,
                    total_runs INTEGER NOT NULL DEFAULT 0,
                    metrics_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id) ON DELETE CASCADE,
                    task_id TEXT NOT NULL REFERENCES tasks(task_id),
                    config_id TEXT NOT NULL REFERENCES agent_configs(config_id),
                    strategy TEXT NOT NULL,
                    status TEXT NOT NULL,
                    final_answer_json TEXT,
                    citations_json TEXT NOT NULL,
                    total_latency_ms REAL NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    total_cost_estimate REAL NOT NULL,
                    metrics_json TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS run_steps (
                    step_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                    step_index INTEGER NOT NULL,
                    phase TEXT NOT NULL,
                    thought TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    action_payload_json TEXT NOT NULL,
                    observation_json TEXT,
                    status TEXT NOT NULL,
                    latency_ms REAL NOT NULL,
                    tool_call_json TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS tool_calls (
                    tool_call_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                    step_index INTEGER NOT NULL,
                    tool_name TEXT NOT NULL,
                    arguments_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    output_json TEXT,
                    error TEXT,
                    latency_ms REAL NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS evaluations (
                    run_id TEXT PRIMARY KEY REFERENCES runs(run_id) ON DELETE CASCADE,
                    success INTEGER NOT NULL,
                    score REAL NOT NULL,
                    exact_match INTEGER NOT NULL,
                    field_match INTEGER NOT NULL,
                    citation_match INTEGER NOT NULL,
                    sql_result_match INTEGER NOT NULL,
                    tool_usage_accuracy REAL NOT NULL,
                    invalid_action_rate REAL NOT NULL,
                    tool_error_rate REAL NOT NULL,
                    recovery_rate REAL NOT NULL,
                    failure_type TEXT NOT NULL,
                    details_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS assistant_sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    config_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS assistant_messages (
                    message_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES assistant_sessions(session_id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    config_id TEXT,
                    citations_json TEXT NOT NULL DEFAULT '[]',
                    run_payload_json TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_tasks_category ON tasks(category);
                CREATE INDEX IF NOT EXISTS idx_runs_experiment_id ON runs(experiment_id);
                CREATE INDEX IF NOT EXISTS idx_runs_config_id ON runs(config_id);
                CREATE INDEX IF NOT EXISTS idx_runs_task_id ON runs(task_id);
                CREATE INDEX IF NOT EXISTS idx_tool_calls_run_id ON tool_calls(run_id);
                CREATE INDEX IF NOT EXISTS idx_assistant_messages_session_id ON assistant_messages(session_id);
                """
            )
            for statement in (
                "ALTER TABLE experiments ADD COLUMN config_preset TEXT",
                "ALTER TABLE experiments ADD COLUMN task_preset TEXT",
            ):
                try:
                    conn.execute(statement)
                except sqlite3.OperationalError:
                    pass

    def has_seed_data(self) -> bool:
        with self.connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS count FROM tasks").fetchone()
        return bool(row and row["count"])

    def seed_reference_data(
        self,
        documents: list[dict[str, Any]],
        chunks: list[dict[str, Any]],
        services: list[dict[str, Any]],
        cases: list[dict[str, Any]],
        tasks: list[TaskSpec],
        agent_configs: list[AgentConfig],
    ) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM evaluations")
            conn.execute("DELETE FROM tool_calls")
            conn.execute("DELETE FROM run_steps")
            conn.execute("DELETE FROM runs")
            conn.execute("DELETE FROM experiments")
            conn.execute("DELETE FROM tasks")
            conn.execute("DELETE FROM agent_configs")
            conn.execute("DELETE FROM mock_cases")
            conn.execute("DELETE FROM service_catalog")
            conn.execute("DELETE FROM chunk_fts")
            conn.execute("DELETE FROM document_chunks")
            conn.execute("DELETE FROM documents")

            conn.executemany(
                """
                INSERT INTO documents(doc_id, title, source_url, source_label, reference_date, body)
                VALUES(:doc_id, :title, :source_url, :source_label, :reference_date, :body)
                """,
                documents,
            )
            conn.executemany(
                """
                INSERT INTO document_chunks(chunk_id, doc_id, heading, body, chunk_order)
                VALUES(:chunk_id, :doc_id, :heading, :body, :chunk_order)
                """,
                chunks,
            )
            conn.executemany(
                """
                INSERT INTO chunk_fts(chunk_id, doc_id, heading, body)
                VALUES(:chunk_id, :doc_id, :heading, :body)
                """,
                chunks,
            )
            conn.executemany(
                """
                INSERT INTO service_catalog(service, owner_team, runbook_url, fallback_policy)
                VALUES(:service, :owner_team, :runbook_url, :fallback_policy)
                """,
                services,
            )
            conn.executemany(
                """
                INSERT INTO mock_cases(
                    case_id, service, title, region, priority, target_rps, worker_count,
                    primary_chunk_id, supporting_chunk_id, recommended_feature, notes
                )
                VALUES(
                    :case_id, :service, :title, :region, :priority, :target_rps, :worker_count,
                    :primary_chunk_id, :supporting_chunk_id, :recommended_feature, :notes
                )
                """,
                cases,
            )
            conn.executemany(
                "INSERT INTO tasks(task_id, title, category, payload_json) VALUES(:task_id, :title, :category, :payload_json)",
                [
                    {
                        "task_id": task.task_id,
                        "title": task.title,
                        "category": task.category.value,
                        "payload_json": json_dumps(task.model_dump(mode="json")),
                    }
                    for task in tasks
                ],
            )
            conn.executemany(
                "INSERT INTO agent_configs(config_id, display_name, strategy, payload_json) VALUES(:config_id, :display_name, :strategy, :payload_json)",
                [
                    {
                        "config_id": config.config_id,
                        "display_name": config.display_name,
                        "strategy": config.strategy.value,
                        "payload_json": json_dumps(config.model_dump(mode="json")),
                    }
                    for config in agent_configs
                ],
            )

    def list_tasks(self) -> list[TaskSpec]:
        with self.connect() as conn:
            rows = conn.execute("SELECT payload_json FROM tasks ORDER BY task_id").fetchall()
        return [TaskSpec.model_validate(json_loads(row["payload_json"])) for row in rows]

    def get_task(self, task_id: str) -> TaskSpec:
        with self.connect() as conn:
            row = conn.execute("SELECT payload_json FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown task_id: {task_id}")
        return TaskSpec.model_validate(json_loads(row["payload_json"]))

    def list_agent_configs(self) -> list[AgentConfig]:
        with self.connect() as conn:
            rows = conn.execute("SELECT payload_json FROM agent_configs ORDER BY display_name").fetchall()
        return [AgentConfig.model_validate(json_loads(row["payload_json"])) for row in rows]

    def get_agent_config(self, config_id: str) -> AgentConfig:
        with self.connect() as conn:
            row = conn.execute("SELECT payload_json FROM agent_configs WHERE config_id = ?", (config_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown config_id: {config_id}")
        return AgentConfig.model_validate(json_loads(row["payload_json"]))

    def list_documents(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM documents ORDER BY doc_id").fetchall()
        return [dict(row) for row in rows]

    def create_assistant_session(self, title: str, config_id: str | None = None) -> str:
        session_id = f"session_{uuid4().hex}"
        created_at = utc_now().isoformat()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO assistant_sessions(session_id, title, config_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, title, config_id, created_at, created_at),
            )
        return session_id

    def update_assistant_session(
        self,
        session_id: str,
        *,
        title: str | None = None,
        config_id: str | None = None,
    ) -> None:
        updates: list[str] = ["updated_at = ?"]
        params: list[Any] = [utc_now().isoformat()]
        if title is not None:
            updates.append("title = ?")
            params.append(title)
        if config_id is not None:
            updates.append("config_id = ?")
            params.append(config_id)
        params.append(session_id)
        with self.connect() as conn:
            conn.execute(
                f"UPDATE assistant_sessions SET {', '.join(updates)} WHERE session_id = ?",
                tuple(params),
            )

    def add_assistant_message(
        self,
        session_id: str,
        *,
        role: str,
        content: str,
        config_id: str | None = None,
        citations: list[str] | None = None,
        run_payload: dict[str, Any] | None = None,
    ) -> str:
        message = AssistantMessageRecord(
            session_id=session_id,
            role=role,
            content=content,
            config_id=config_id,
            citations=citations or [],
            run_payload=run_payload,
        )
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO assistant_messages(
                    message_id, session_id, role, content, config_id,
                    citations_json, run_payload_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message.message_id,
                    message.session_id,
                    message.role,
                    message.content,
                    message.config_id,
                    json_dumps(message.citations),
                    json_dumps(message.run_payload) if message.run_payload is not None else None,
                    message.created_at.isoformat(),
                ),
            )
            conn.execute(
                "UPDATE assistant_sessions SET updated_at = ? WHERE session_id = ?",
                (message.created_at.isoformat(), session_id),
            )
        return message.message_id

    def list_assistant_sessions(self, limit: int = 20) -> list[AssistantSessionSummary]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    s.session_id,
                    s.title,
                    s.config_id,
                    s.created_at,
                    s.updated_at,
                    COUNT(m.message_id) AS message_count,
                    (
                        SELECT content
                        FROM assistant_messages m2
                        WHERE m2.session_id = s.session_id
                        ORDER BY m2.created_at DESC, m2.message_id DESC
                        LIMIT 1
                    ) AS last_message_preview
                FROM assistant_sessions s
                LEFT JOIN assistant_messages m ON m.session_id = s.session_id
                GROUP BY s.session_id, s.title, s.config_id, s.created_at, s.updated_at
                ORDER BY s.updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [AssistantSessionSummary.model_validate(dict(row)) for row in rows]

    def get_assistant_session(self, session_id: str) -> dict[str, Any]:
        with self.connect() as conn:
            session_row = conn.execute(
                """
                SELECT
                    s.session_id,
                    s.title,
                    s.config_id,
                    s.created_at,
                    s.updated_at,
                    COUNT(m.message_id) AS message_count
                FROM assistant_sessions s
                LEFT JOIN assistant_messages m ON m.session_id = s.session_id
                WHERE s.session_id = ?
                GROUP BY s.session_id, s.title, s.config_id, s.created_at, s.updated_at
                """,
                (session_id,),
            ).fetchone()
            if session_row is None:
                raise KeyError(f"Unknown session_id: {session_id}")
            message_rows = conn.execute(
                """
                SELECT *
                FROM assistant_messages
                WHERE session_id = ?
                ORDER BY created_at ASC, message_id ASC
                """,
                (session_id,),
            ).fetchall()
        session = dict(session_row)
        messages = []
        for row in message_rows:
            item = dict(row)
            item["citations"] = json_loads(item.pop("citations_json"), [])
            item["run_payload"] = json_loads(item.pop("run_payload_json"), None)
            messages.append(item)
        return {"session": session, "messages": messages}

    def search_chunks(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        match_query = fts_query(query)
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    c.chunk_id,
                    c.doc_id,
                    c.heading,
                    c.body,
                    snippet(chunk_fts, 3, '[', ']', '...', 18) AS snippet,
                    bm25(chunk_fts) AS score
                FROM chunk_fts
                JOIN document_chunks c ON c.chunk_id = chunk_fts.chunk_id
                WHERE chunk_fts MATCH ?
                ORDER BY bm25(chunk_fts), c.chunk_order
                LIMIT ?
                """,
                (match_query, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_chunk(self, chunk_id: str) -> dict[str, Any]:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT c.*, d.title AS doc_title, d.source_url
                FROM document_chunks c
                JOIN documents d ON d.doc_id = c.doc_id
                WHERE c.chunk_id = ?
                """,
                (chunk_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown chunk_id: {chunk_id}")
        return dict(row)

    def get_case(self, case_id: str) -> dict[str, Any]:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM mock_cases WHERE case_id = ?", (case_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown case_id: {case_id}")
        return dict(row)

    def run_readonly_query(self, sql: str) -> list[dict[str, Any]]:
        normalized = sql.strip().lower()
        forbidden = ("insert ", "update ", "delete ", "drop ", "alter ", "create ", "replace ", "attach ", "detach ", "vacuum", "reindex ")
        if not normalized.startswith(("select", "with", "pragma")) or any(token in normalized for token in forbidden):
            raise ValueError("sql_query only allows read-only SELECT/WITH/PRAGMA statements")
        with self.connect() as conn:
            rows = conn.execute(sql).fetchall()
        return [dict(row) for row in rows]

    def create_experiment(
        self,
        config_ids: list[str],
        task_ids: list[str],
        *,
        config_preset: str | None = None,
        task_preset: str | None = None,
    ) -> str:
        experiment_id = f"exp_{uuid4().hex}"
        summary = ExperimentSummary(
            experiment_id=experiment_id,
            config_preset=config_preset,
            task_preset=task_preset,
            config_ids=config_ids,
            task_ids=task_ids,
        )
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO experiments(
                    experiment_id, created_at, config_preset, task_preset,
                    config_ids_json, task_ids_json, total_runs, metrics_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    summary.created_at.isoformat(),
                    summary.config_preset,
                    summary.task_preset,
                    json_dumps(config_ids),
                    json_dumps(task_ids),
                    0,
                    json_dumps({}),
                ),
            )
        return experiment_id

    def save_run(self, run: RunRecord) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO runs(
                    run_id, experiment_id, task_id, config_id, strategy, status, final_answer_json,
                    citations_json, total_latency_ms, total_tokens, total_cost_estimate, metrics_json,
                    started_at, finished_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.run_id,
                    run.experiment_id,
                    run.task_id,
                    run.config_id,
                    run.strategy.value,
                    run.status.value,
                    json_dumps(run.final_answer),
                    json_dumps(run.citations),
                    run.total_latency_ms,
                    run.total_tokens,
                    run.total_cost_estimate,
                    json_dumps(run.metrics),
                    run.started_at.isoformat(),
                    run.finished_at.isoformat(),
                ),
            )
            for step in run.steps:
                conn.execute(
                    """
                    INSERT INTO run_steps(
                        step_id, run_id, step_index, phase, thought, action_type, action_payload_json,
                        observation_json, status, latency_ms, tool_call_json, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        step.step_id,
                        run.run_id,
                        step.step_index,
                        step.phase,
                        step.thought,
                        step.action_type,
                        json_dumps(step.action_payload),
                        json_dumps(step.observation),
                        step.status,
                        step.latency_ms,
                        json_dumps(step.tool_call.model_dump(mode="json")) if step.tool_call else None,
                        step.created_at.isoformat(),
                    ),
                )
                if step.tool_call:
                    tool = step.tool_call
                    conn.execute(
                        """
                        INSERT INTO tool_calls(
                            tool_call_id, run_id, step_index, tool_name, arguments_json, status, output_json,
                            error, latency_ms, created_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            tool.tool_call_id,
                            run.run_id,
                            step.step_index,
                            tool.tool_name,
                            json_dumps(tool.arguments),
                            tool.status.value,
                            json_dumps(tool.output),
                            tool.error,
                            tool.latency_ms,
                            tool.created_at.isoformat(),
                        ),
                    )
            evaluation = run.evaluation
            conn.execute(
                """
                INSERT INTO evaluations(
                    run_id, success, score, exact_match, field_match, citation_match, sql_result_match,
                    tool_usage_accuracy, invalid_action_rate, tool_error_rate, recovery_rate,
                    failure_type, details_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.run_id,
                    int(evaluation.success),
                    evaluation.score,
                    int(evaluation.exact_match),
                    int(evaluation.field_match),
                    int(evaluation.citation_match),
                    int(evaluation.sql_result_match),
                    evaluation.tool_usage_accuracy,
                    evaluation.invalid_action_rate,
                    evaluation.tool_error_rate,
                    evaluation.recovery_rate,
                    evaluation.failure_type.value,
                    json_dumps(evaluation.details),
                ),
            )
            conn.execute(
                """
                UPDATE experiments
                SET total_runs = (SELECT COUNT(*) FROM runs WHERE experiment_id = ?)
                WHERE experiment_id = ?
                """,
                (run.experiment_id, run.experiment_id),
            )

    def update_experiment_metrics(self, experiment_id: str, metrics: dict[str, Any]) -> None:
        with self.connect() as conn:
            conn.execute("UPDATE experiments SET metrics_json = ? WHERE experiment_id = ?", (json_dumps(metrics), experiment_id))

    def list_leaderboard(self, experiment_id: str | None = None) -> list[LeaderboardEntry]:
        where_clause = ""
        params: tuple[Any, ...] = ()
        if experiment_id:
            where_clause = "WHERE r.experiment_id = ?"
            params = (experiment_id,)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT
                    a.config_id,
                    a.display_name,
                    a.strategy,
                    AVG(e.success) AS success_rate,
                    AVG(r.total_latency_ms) AS avg_latency_ms,
                    AVG(r.total_tokens) AS avg_tokens,
                    AVG(r.total_cost_estimate) AS avg_cost,
                    AVG(e.tool_error_rate) AS tool_error_rate,
                    AVG(e.invalid_action_rate) AS invalid_action_rate,
                    AVG(e.recovery_rate) AS recovery_rate,
                    AVG(json_extract(r.metrics_json, '$.step_count')) AS avg_steps,
                    COUNT(*) AS total_runs
                FROM runs r
                JOIN evaluations e ON e.run_id = r.run_id
                JOIN agent_configs a ON a.config_id = r.config_id
                {where_clause}
                GROUP BY a.config_id, a.display_name, a.strategy
                ORDER BY success_rate DESC, recovery_rate DESC, avg_latency_ms ASC
                """,
                params,
            ).fetchall()
        return [LeaderboardEntry.model_validate(dict(row)) for row in rows]

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self.connect() as conn:
            run_row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            if run_row is None:
                raise KeyError(f"Unknown run_id: {run_id}")
            step_rows = conn.execute("SELECT * FROM run_steps WHERE run_id = ? ORDER BY step_index", (run_id,)).fetchall()
            eval_row = conn.execute("SELECT * FROM evaluations WHERE run_id = ?", (run_id,)).fetchone()
            task_row = conn.execute("SELECT payload_json FROM tasks WHERE task_id = ?", (run_row["task_id"],)).fetchone()
            config_row = conn.execute("SELECT payload_json FROM agent_configs WHERE config_id = ?", (run_row["config_id"],)).fetchone()
        run = dict(run_row)
        run["final_answer"] = json_loads(run.pop("final_answer_json"), None)
        run["citations"] = json_loads(run.pop("citations_json"), [])
        run["metrics"] = json_loads(run.pop("metrics_json"), {})
        steps = []
        for row in step_rows:
            item = dict(row)
            item["action_payload"] = json_loads(item.pop("action_payload_json"), {})
            item["observation"] = json_loads(item.pop("observation_json"), None)
            item["tool_call"] = json_loads(item.pop("tool_call_json"), None)
            steps.append(item)
        evaluation = dict(eval_row) if eval_row else {}
        if evaluation:
            evaluation["details"] = json_loads(evaluation.pop("details_json"), {})
        return {
            "run": run,
            "steps": steps,
            "evaluation": evaluation,
            "task": json_loads(task_row["payload_json"]) if task_row else None,
            "config": json_loads(config_row["payload_json"]) if config_row else None,
        }

    def list_failures(self, experiment_id: str | None = None) -> list[FailureRecord]:
        where_clause = "WHERE e.success = 0"
        params: tuple[Any, ...] = ()
        if experiment_id:
            where_clause += " AND r.experiment_id = ?"
            params = (experiment_id,)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT
                    r.run_id,
                    r.task_id,
                    r.config_id,
                    r.strategy,
                    e.failure_type,
                    t.title,
                    json_extract(t.payload_json, '$.prompt') AS prompt,
                    e.details_json
                FROM evaluations e
                JOIN runs r ON r.run_id = e.run_id
                JOIN tasks t ON t.task_id = r.task_id
                {where_clause}
                ORDER BY r.finished_at DESC
                """,
                params,
            ).fetchall()
        return [FailureRecord.model_validate({**dict(row), "details": json_loads(row["details_json"], {})}) for row in rows]

    def latest_experiment(self) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM experiments ORDER BY created_at DESC LIMIT 1").fetchone()
        return self._deserialize_experiment_row(row) if row else None

    def list_experiments(self, limit: int = 20) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM experiments ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._deserialize_experiment_row(row) for row in rows]

    def get_experiment(self, experiment_id: str) -> dict[str, Any]:
        with self.connect() as conn:
            experiment_row = conn.execute(
                "SELECT * FROM experiments WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchone()
            if experiment_row is None:
                raise KeyError(f"Unknown experiment_id: {experiment_id}")
            run_rows = conn.execute(
                """
                SELECT
                    r.run_id,
                    r.task_id,
                    t.title AS task_title,
                    t.category AS task_category,
                    r.config_id,
                    a.display_name,
                    a.strategy,
                    r.status,
                    r.total_latency_ms,
                    r.total_tokens,
                    r.total_cost_estimate,
                    r.metrics_json,
                    r.finished_at,
                    COALESCE(e.success, 0) AS success,
                    COALESCE(e.score, 0.0) AS score,
                    COALESCE(e.failure_type, 'none') AS failure_type
                FROM runs r
                JOIN tasks t ON t.task_id = r.task_id
                JOIN agent_configs a ON a.config_id = r.config_id
                LEFT JOIN evaluations e ON e.run_id = r.run_id
                WHERE r.experiment_id = ?
                ORDER BY r.finished_at DESC, r.run_id DESC
                """,
                (experiment_id,),
            ).fetchall()
            category_rows = conn.execute(
                """
                SELECT
                    t.category,
                    COUNT(*) AS total_runs,
                    AVG(e.success) AS success_rate,
                    AVG(r.total_latency_ms) AS avg_latency_ms
                FROM runs r
                JOIN tasks t ON t.task_id = r.task_id
                JOIN evaluations e ON e.run_id = r.run_id
                WHERE r.experiment_id = ?
                GROUP BY t.category
                ORDER BY t.category
                """,
                (experiment_id,),
            ).fetchall()
            failure_rows = conn.execute(
                """
                SELECT e.failure_type, COUNT(*) AS count
                FROM evaluations e
                JOIN runs r ON r.run_id = e.run_id
                WHERE r.experiment_id = ? AND e.success = 0
                GROUP BY e.failure_type
                ORDER BY count DESC, e.failure_type ASC
                """,
                (experiment_id,),
            ).fetchall()
        runs = []
        success_count = 0
        for row in run_rows:
            item = dict(row)
            item["metrics"] = json_loads(item.pop("metrics_json"), {})
            item["success"] = bool(item["success"])
            success_count += int(item["success"])
            runs.append(item)
        return {
            "experiment": self._deserialize_experiment_row(experiment_row),
            "leaderboard": [entry.model_dump(mode="json") for entry in self.list_leaderboard(experiment_id)],
            "failures": [failure.model_dump(mode="json") for failure in self.list_failures(experiment_id)],
            "category_breakdown": [dict(row) for row in category_rows],
            "failure_breakdown": [dict(row) for row in failure_rows],
            "runs": runs,
            "run_count": len(runs),
            "success_count": success_count,
        }

    def dashboard_summary(self) -> dict[str, Any]:
        with self.connect() as conn:
            counts = conn.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM documents) AS documents,
                    (SELECT COUNT(*) FROM document_chunks) AS chunks,
                    (SELECT COUNT(*) FROM tasks) AS tasks,
                    (SELECT COUNT(*) FROM agent_configs) AS configs,
                    (SELECT COUNT(*) FROM experiments) AS experiments,
                    (SELECT COUNT(*) FROM runs) AS runs,
                    (SELECT COUNT(*) FROM assistant_sessions) AS assistant_sessions
                """
            ).fetchone()
            recent_runs = conn.execute(
                """
                SELECT run_id, task_id, config_id, strategy, status, finished_at
                FROM runs
                ORDER BY finished_at DESC
                LIMIT 10
                """
            ).fetchall()
        return {**dict(counts), "recent_runs": [dict(row) for row in recent_runs]}

    def _deserialize_experiment_row(self, row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
        item = dict(row)
        item.setdefault("config_preset", None)
        item.setdefault("task_preset", None)
        item["config_ids"] = json_loads(item.pop("config_ids_json"), [])
        item["task_ids"] = json_loads(item.pop("task_ids_json"), [])
        item["metrics"] = json_loads(item.pop("metrics_json"), {})
        return item
