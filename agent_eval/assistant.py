from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from agent_eval.experiments import PlatformServices
from agent_eval.models import (
    AgentConfig,
    AnswerFormat,
    AssistantAskRequest,
    EvalResult,
    FailureType,
    RunStatus,
    StepRecord,
    StrategyType,
    TaskCategory,
    TaskSpec,
    ToolCallStatus,
    ValidationMode,
    ValidationSpec,
    utc_now,
)
from agent_eval.runners import build_run_record
from agent_eval.tools import FaultController


@dataclass
class AssistantDefaults:
    default_config_id: str | None
    configs: list[dict[str, Any]]


class AssistantService:
    def __init__(self, services: PlatformServices) -> None:
        self.services = services

    @property
    def storage(self):
        return self.services.storage

    def home(self) -> dict[str, Any]:
        sessions = self.storage.list_assistant_sessions(limit=20)
        defaults = self._defaults()
        latest_session = sessions[0].session_id if sessions else None
        payload = self.session_detail(latest_session) if latest_session else {"session": None, "messages": []}
        payload["sessions"] = [item.model_dump(mode="json") for item in sessions]
        payload["configs"] = defaults.configs
        payload["default_config_id"] = defaults.default_config_id
        return payload

    def session_detail(self, session_id: str | None) -> dict[str, Any]:
        defaults = self._defaults()
        if session_id is None:
            return {
                "session": None,
                "messages": [],
                "sessions": [item.model_dump(mode="json") for item in self.storage.list_assistant_sessions(limit=20)],
                "configs": defaults.configs,
                "default_config_id": defaults.default_config_id,
            }
        payload = self.storage.get_assistant_session(session_id)
        payload["sessions"] = [item.model_dump(mode="json") for item in self.storage.list_assistant_sessions(limit=20)]
        payload["configs"] = defaults.configs
        payload["default_config_id"] = defaults.default_config_id
        return payload

    def ask(self, request: AssistantAskRequest) -> dict[str, Any]:
        prompt = request.prompt.strip()
        if not prompt:
            raise ValueError("Prompt cannot be empty")
        config = self._resolve_config(request.config_id)
        session_id = request.session_id
        title = self._session_title(prompt)
        if session_id is None:
            session_id = self.storage.create_assistant_session(title=title, config_id=config.config_id)
        else:
            self.storage.update_assistant_session(session_id, config_id=config.config_id)

        self.storage.add_assistant_message(
            session_id,
            role="user",
            content=prompt,
            config_id=config.config_id,
        )

        run_payload = self._run_live_query(prompt=prompt, config=config, session_id=session_id)
        answer_text = self._answer_text(run_payload.get("final_answer"))
        self.storage.add_assistant_message(
            session_id,
            role="assistant",
            content=answer_text,
            config_id=config.config_id,
            citations=run_payload.get("citations", []),
            run_payload=run_payload,
        )
        return self.session_detail(session_id)

    def _defaults(self) -> AssistantDefaults:
        configs = self.storage.list_agent_configs()
        sorted_configs = sorted(configs, key=self._assistant_config_score, reverse=True)
        default_config_id = sorted_configs[0].config_id if sorted_configs else None
        return AssistantDefaults(
            default_config_id=default_config_id,
            configs=[
                {
                    "config_id": config.config_id,
                    "display_name": config.display_name,
                    "model_provider": config.model_provider,
                    "strategy": config.strategy.value,
                    "recommended": config.config_id == default_config_id,
                }
                for config in sorted_configs
            ],
        )

    def _resolve_config(self, config_id: str | None) -> AgentConfig:
        if config_id:
            return self.storage.get_agent_config(config_id)
        defaults = self._defaults()
        if defaults.default_config_id is None:
            raise ValueError("No agent configs available")
        return self.storage.get_agent_config(defaults.default_config_id)

    def _assistant_config_score(self, config: AgentConfig) -> tuple[int, int, int]:
        provider_score = {"ollama": 3, "dashscope": 2, "heuristic": 1}.get(config.model_provider, 0)
        strategy_score = {
            StrategyType.VERIFIER: 3,
            StrategyType.PLANNER: 2,
            StrategyType.BASELINE: 1,
        }.get(config.strategy, 0)
        web_score = 1 if config.enable_web_lookup else 0
        return (provider_score, strategy_score, web_score)

    def _session_title(self, prompt: str) -> str:
        trimmed = re.sub(r"\s+", " ", prompt).strip()
        return trimmed[:32] if len(trimmed) > 32 else trimmed

    def _run_live_query(self, *, prompt: str, config: AgentConfig, session_id: str) -> dict[str, Any]:
        if config.model_provider == "heuristic":
            run = self._run_grounded_fallback(prompt=prompt, config=config, session_id=session_id)
            return run.model_dump(mode="json")
        task = self._build_live_task(prompt)
        run = self.services.runner.run(task=task, config=config, experiment_id=f"assistant_{session_id}")
        if run.final_answer in (None, "", {}) or not run.citations:
            run = self._run_grounded_fallback(prompt=prompt, config=config, session_id=session_id)
        return run.model_dump(mode="json")

    def _build_live_task(self, prompt: str) -> TaskSpec:
        search_hints = self._search_hints(prompt)
        metadata: dict[str, Any] = {"live_assistant": True}
        required_tools = ["doc_search", "doc_read"]

        case_match = re.search(r"\bCASE-\d+\b", prompt.upper())
        if case_match:
            case_id = case_match.group(0)
            required_tools.extend(["case_api", "sql_query"])
            metadata["case_id"] = case_id
            metadata["sql_query"] = (
                "SELECT owner_team, runbook_url, fallback_policy "
                "FROM service_catalog "
                f"WHERE service = (SELECT service FROM mock_cases WHERE case_id = '{case_id}')"
            )
            search_hints.insert(0, case_id)

        return TaskSpec(
            task_id=f"LIVE-{utc_now().strftime('%Y%m%d%H%M%S%f')}",
            title="在线知识库问答",
            category=TaskCategory.SINGLE_HOP,
            prompt=prompt,
            answer_format=AnswerFormat.TEXT,
            expected_answer="",
            required_tools=list(dict.fromkeys(required_tools)),
            reference_chunk_ids=[],
            search_hints=search_hints,
            validation=ValidationSpec(mode=ValidationMode.EXACT_TEXT),
            metadata=metadata,
        )

    def _search_hints(self, prompt: str) -> list[str]:
        compact = re.sub(r"\s+", " ", prompt.strip())
        tokens = re.findall(r"[A-Za-z0-9_]+", compact)
        keyword_hint = " ".join(tokens[:8]).strip()
        hints = [compact]
        if keyword_hint and keyword_hint.lower() != compact.lower():
            hints.append(keyword_hint)
        return hints

    def _run_grounded_fallback(self, *, prompt: str, config: AgentConfig, session_id: str):
        started_at = utc_now()
        steps: list[StepRecord] = []
        citations: list[str] = []
        fault_controller = FaultController([])
        step_index = 1
        case_match = re.search(r"\bCASE-\d+\b", prompt.upper())
        case_payload: dict[str, Any] | None = None
        ownership_rows: list[dict[str, Any]] = []

        def add_step(
            phase: str,
            thought: str,
            action_type: str,
            action_payload: dict[str, Any] | None = None,
            observation: Any = None,
            status: str = "ok",
            latency_ms: float = 0.0,
            tool_call=None,
        ) -> None:
            nonlocal step_index
            steps.append(
                StepRecord(
                    step_index=step_index,
                    phase=phase,
                    thought=thought,
                    action_type=action_type,
                    action_payload=action_payload or {},
                    observation=observation,
                    status=status,
                    latency_ms=latency_ms,
                    tool_call=tool_call,
                )
            )
            step_index += 1

        if case_match:
            case_id = case_match.group(0)
            case_record = self.services.tools.execute("case_api", {"case_id": case_id}, fault_controller)
            add_step(
                phase="execution",
                thought="检测到 case_id，先读取结构化案例信息。",
                action_type="tool",
                action_payload={"tool_name": "case_api", "arguments": {"case_id": case_id}},
                observation=case_record.output if case_record.error is None else {"error": case_record.error},
                status="ok" if case_record.status == ToolCallStatus.SUCCESS else case_record.status.value,
                latency_ms=case_record.latency_ms,
                tool_call=case_record,
            )
            if case_record.status == ToolCallStatus.SUCCESS:
                case_payload = case_record.output
                sql = (
                    "SELECT owner_team, runbook_url, fallback_policy "
                    "FROM service_catalog "
                    f"WHERE service = '{case_payload['service']}'"
                )
                sql_record = self.services.tools.execute("sql_query", {"sql": sql}, fault_controller)
                add_step(
                    phase="execution",
                    thought="补充责任团队和 runbook 信息。",
                    action_type="tool",
                    action_payload={"tool_name": "sql_query", "arguments": {"sql": sql}},
                    observation=sql_record.output if sql_record.error is None else {"error": sql_record.error},
                    status="ok" if sql_record.status == ToolCallStatus.SUCCESS else sql_record.status.value,
                    latency_ms=sql_record.latency_ms,
                    tool_call=sql_record,
                )
                if sql_record.status == ToolCallStatus.SUCCESS:
                    ownership_rows = sql_record.output

        search_query = prompt
        if case_payload:
            search_query = f"{case_payload['recommended_feature']} {case_payload['notes']}"
        search_record = self.services.tools.execute("doc_search", {"query": search_query, "limit": 5}, fault_controller)
        add_step(
            phase="execution",
            thought="在本地知识库中检索最相关的文档切块。",
            action_type="tool",
            action_payload={"tool_name": "doc_search", "arguments": {"query": search_query, "limit": 5}},
            observation=search_record.output if search_record.error is None else {"error": search_record.error},
            status="ok" if search_record.status == ToolCallStatus.SUCCESS else search_record.status.value,
            latency_ms=search_record.latency_ms,
            tool_call=search_record,
        )

        final_answer = "我没有在当前知识库中找到足够直接的证据，请换一个更具体的问题或提供 case_id。"
        if search_record.status == ToolCallStatus.SUCCESS and search_record.output:
            top_hit = search_record.output[0]
            read_record = self.services.tools.execute(
                "doc_read",
                {"chunk_id": top_hit["chunk_id"]},
                fault_controller,
            )
            add_step(
                phase="execution",
                thought="读取命中的文档切块，生成带引用的回答。",
                action_type="tool",
                action_payload={"tool_name": "doc_read", "arguments": {"chunk_id": top_hit["chunk_id"]}},
                observation=read_record.output if read_record.error is None else {"error": read_record.error},
                status="ok" if read_record.status == ToolCallStatus.SUCCESS else read_record.status.value,
                latency_ms=read_record.latency_ms,
                tool_call=read_record,
            )
            if read_record.status == ToolCallStatus.SUCCESS:
                citations.append(top_hit["chunk_id"])
                final_answer = self._compose_grounded_text(
                    prompt=prompt,
                    chunk=read_record.output,
                    case_payload=case_payload,
                    ownership_rows=ownership_rows,
                )

        add_step(
            phase="final",
            thought="返回在线知识库助手的最终答案和引用。",
            action_type="final",
            action_payload={"session_id": session_id, "config_id": config.config_id},
            observation={"answer": final_answer, "citations": citations},
            latency_ms=10.0,
        )

        run = build_run_record(
            task=self._build_live_task(prompt),
            config=config,
            experiment_id=f"assistant_{session_id}",
            started_at=started_at,
            steps=steps,
            citations=citations,
            final_answer=final_answer,
            extra_metrics={"provider": "grounded_fallback", "tool_count": len([step for step in steps if step.tool_call])},
        )
        run.status = RunStatus.COMPLETED if citations else RunStatus.FAILED
        run.evaluation = EvalResult(
            success=bool(citations),
            score=1.0 if citations else 0.0,
            citation_match=bool(citations),
            failure_type=FailureType.NONE if citations else FailureType.WRONG_RETRIEVAL,
        )
        return run

    def _compose_grounded_text(
        self,
        *,
        prompt: str,
        chunk: dict[str, Any],
        case_payload: dict[str, Any] | None,
        ownership_rows: list[dict[str, Any]],
    ) -> str:
        body = re.sub(r"\s+", " ", chunk.get("body", "")).strip()
        summary = body[:260].rstrip()
        pieces = [f"根据《{chunk.get('doc_title', '文档')}》中“{chunk.get('heading', '相关章节')}”的内容：{summary}"]
        if case_payload:
            pieces.insert(
                0,
                f"{case_payload['case_id']} 属于 {case_payload['service']}，优先级为 {case_payload['priority']}，区域为 {case_payload['region']}，建议重点关注 {case_payload['recommended_feature']}。",
            )
        if ownership_rows:
            owner = ownership_rows[0]
            pieces.append(
                f"当前责任团队是 {owner.get('owner_team')}，runbook 为 {owner.get('runbook_url')}。"
            )
        if "?" not in prompt and "？" not in prompt:
            pieces.append("如果你需要，我可以继续基于同一知识库追问细节。")
        return " ".join(pieces)

    def _answer_text(self, final_answer: Any) -> str:
        if isinstance(final_answer, dict):
            parts = [f"{key}: {value}" for key, value in final_answer.items()]
            return "\n".join(parts)
        return str(final_answer or "").strip()
