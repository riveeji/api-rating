from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_eval.config import Settings
from agent_eval.llm import DashScopeChatClient, LLMClientError, OllamaChatClient, message_text, parse_json_content, parse_tool_calls
from agent_eval.models import AgentConfig, AnswerFormat, EvalResult, RunRecord, RunStatus, StepRecord, StrategyType, TaskSpec, utc_now
from agent_eval.tools import FaultController, ToolRegistry
from agent_eval.utils import cost_estimate, json_dumps, token_estimate


@dataclass
class ExecutionState:
    case_data: dict[str, Any] | None = None
    sql_rows: list[dict[str, Any]] = field(default_factory=list)
    calculations: dict[str, Any] | None = None
    search_results: list[dict[str, Any]] = field(default_factory=list)
    chunk_cache: dict[str, dict[str, Any]] = field(default_factory=dict)
    web_result: dict[str, Any] | None = None


class AgentRunner:
    def run(self, task: TaskSpec, config: AgentConfig, experiment_id: str) -> RunRecord:
        raise NotImplementedError


class RunnerRouter(AgentRunner):
    def __init__(
        self,
        settings: Settings,
        tools: ToolRegistry,
        dashscope_client: DashScopeChatClient | None = None,
        ollama_client: OllamaChatClient | None = None,
    ) -> None:
        self.settings = settings
        self.heuristic = HeuristicAgentRunner(tools)
        self.dashscope = LiveToolCallingAgentRunner(
            settings,
            tools,
            llm_client=dashscope_client or self._maybe_dashscope_client(settings),
            provider_name="dashscope",
        )
        self.ollama = LiveToolCallingAgentRunner(
            settings,
            tools,
            llm_client=ollama_client or OllamaChatClient(settings),
            provider_name="ollama",
        )

    def run(self, task: TaskSpec, config: AgentConfig, experiment_id: str) -> RunRecord:
        if config.model_provider == "dashscope":
            return self.dashscope.run(task, config, experiment_id)
        if config.model_provider == "ollama":
            return self.ollama.run(task, config, experiment_id)
        return self.heuristic.run(task, config, experiment_id)

    def _maybe_dashscope_client(self, settings: Settings) -> DashScopeChatClient | None:
        if not settings.dashscope_api_key:
            return None
        return DashScopeChatClient(settings)


class HeuristicAgentRunner(AgentRunner):
    def __init__(self, tools: ToolRegistry) -> None:
        self.tools = tools

    def run(self, task: TaskSpec, config: AgentConfig, experiment_id: str) -> RunRecord:
        started_at = utc_now()
        fault_controller = FaultController(task.faults)
        state = ExecutionState()
        steps: list[StepRecord] = []
        citations: list[str] = []
        step_index = 1

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

        def call_tool(tool_name: str, arguments: dict[str, Any], thought: str) -> Any:
            record = self.tools.execute(tool_name, arguments, fault_controller)
            add_step(
                phase="execution",
                thought=thought,
                action_type="tool",
                action_payload={"tool_name": tool_name, "arguments": arguments},
                observation=record.output if record.error is None else {"error": record.error},
                status="ok" if record.status.value == "success" else record.status.value,
                latency_ms=record.latency_ms,
                tool_call=record,
            )
            return record

        if config.strategy != StrategyType.BASELINE:
            add_step(
                phase="planning",
                thought=self._build_plan(task, config),
                action_type="plan",
                action_payload={"strategy": config.strategy.value},
                latency_ms=25.0 if config.strategy == StrategyType.PLANNER else 35.0,
            )

        if task.metadata.get("case_id"):
            record = call_tool(
                "case_api",
                {"case_id": task.metadata["case_id"]},
                "Fetch the structured case payload before choosing later tools.",
            )
            if record.status.value == "success":
                state.case_data = record.output
            elif config.allow_replan:
                add_step(
                    phase="planning",
                    thought="Case lookup failed; retry once because the strategy allows recovery.",
                    action_type="replan",
                    action_payload={"tool": "case_api"},
                    status="retry",
                    latency_ms=20.0,
                )
                retry = call_tool(
                    "case_api",
                    {"case_id": task.metadata["case_id"]},
                    "Retry the transient case API failure once.",
                )
                if retry.status.value == "success":
                    state.case_data = retry.output

        if task.metadata.get("sql_query"):
            sql_record = call_tool(
                "sql_query",
                {"sql": task.metadata["sql_query"]},
                "Read the supporting ownership or runbook data from SQLite.",
            )
            if sql_record.status.value == "success":
                state.sql_rows = sql_record.output
            elif config.allow_replan:
                add_step(
                    phase="planning",
                    thought="SQL lookup did not succeed; retry once with the same read-only query.",
                    action_type="replan",
                    action_payload={"tool": "sql_query"},
                    status="retry",
                    latency_ms=20.0,
                )
                retry = call_tool("sql_query", {"sql": task.metadata["sql_query"]}, "Retry the SQL lookup once.")
                if retry.status.value == "success":
                    state.sql_rows = retry.output

        primary_chunk = self._retrieve_chunk(
            task=task,
            config=config,
            search_hints=task.search_hints,
            citations=citations,
            call_tool=call_tool,
            add_step=add_step,
        )
        if primary_chunk:
            state.chunk_cache[primary_chunk["chunk_id"]] = primary_chunk

        if task.metadata.get("secondary_query") and config.strategy != StrategyType.BASELINE:
            secondary_chunk = self._retrieve_chunk(
                task=task,
                config=config,
                search_hints=[task.metadata["secondary_query"]],
                citations=citations,
                call_tool=call_tool,
                add_step=add_step,
                require_matching=False,
            )
            if secondary_chunk:
                state.chunk_cache[secondary_chunk["chunk_id"]] = secondary_chunk

        if task.metadata.get("calculation"):
            calc_record = call_tool(
                "calculator",
                {"expression": task.metadata["calculation"]},
                "Compute the deterministic arithmetic field for the final answer.",
            )
            if calc_record.status.value == "success":
                state.calculations = calc_record.output
            elif config.allow_replan:
                retry = call_tool(
                    "calculator",
                    {"expression": task.metadata["calculation"]},
                    "Retry the deterministic arithmetic computation once.",
                )
                if retry.status.value == "success":
                    state.calculations = retry.output

        if (
            not citations
            and config.enable_web_lookup
            and config.strategy == StrategyType.BASELINE
            and task.category != task.category.SINGLE_HOP
        ):
            web_record = call_tool(
                "web_lookup",
                {"query": task.prompt},
                "Fallback to a live web lookup when the baseline policy cannot ground the docs locally.",
            )
            if web_record.status.value == "success":
                state.web_result = web_record.output

        final_answer = self._compose_answer(task, state, citations)

        if config.enable_verifier:
            add_step(
                phase="verification",
                thought="Check citation completeness and JSON field coverage before finishing the run.",
                action_type="verify",
                action_payload={"required_citations": task.reference_chunk_ids},
                observation={"current_citations": citations, "current_answer": final_answer},
                latency_ms=30.0,
            )
            final_answer = self._repair_answer(
                task=task,
                config=config,
                state=state,
                citations=citations,
                final_answer=final_answer,
                call_tool=call_tool,
                add_step=add_step,
            )

        add_step(
            phase="final",
            thought="Return the grounded answer and the collected citations.",
            action_type="final",
            action_payload={"answer_format": task.answer_format.value},
            observation={"answer": final_answer, "citations": citations},
            latency_ms=10.0,
        )

        return build_run_record(
            task=task,
            config=config,
            experiment_id=experiment_id,
            started_at=started_at,
            steps=steps,
            citations=citations,
            final_answer=final_answer,
            extra_metrics={
                "faults_triggered": fault_controller.consumed,
                "tool_count": len([step for step in steps if step.tool_call]),
            },
        )

    def _build_plan(self, task: TaskSpec, config: AgentConfig) -> str:
        tools = ", ".join(task.required_tools)
        if config.strategy == StrategyType.PLANNER:
            return f"Plan: gather evidence with {tools}, then assemble the answer in the requested format."
        return f"Plan: gather evidence with {tools}, assemble the answer, then verify JSON and citations."

    def _retrieve_chunk(
        self,
        task: TaskSpec,
        config: AgentConfig,
        search_hints: list[str],
        citations: list[str],
        call_tool,
        add_step,
        require_matching: bool = True,
    ) -> dict[str, Any] | None:
        for attempt, query in enumerate(search_hints, start=1):
            search_record = call_tool(
                "doc_search",
                {"query": query, "limit": 5},
                f"Search the local documentation corpus using query attempt {attempt}.",
            )
            if search_record.status.value == "empty":
                if config.allow_replan and attempt < len(search_hints):
                    add_step(
                        phase="planning",
                        thought="The first retrieval returned no results; switch to the alternate query.",
                        action_type="replan",
                        action_payload={"query": query},
                        status="retry",
                        latency_ms=20.0,
                    )
                    continue
                return None
            if search_record.status.value != "success":
                if config.allow_replan and attempt < len(search_hints):
                    add_step(
                        phase="planning",
                        thought="The retrieval failed; retry with the next search hint.",
                        action_type="replan",
                        action_payload={"query": query},
                        status="retry",
                        latency_ms=20.0,
                    )
                    continue
                return None

            selected = self._select_chunk(task, search_record.output, config, require_matching)
            if not selected:
                return None
            read_record = call_tool(
                "doc_read",
                {"chunk_id": selected["chunk_id"]},
                "Read the chosen chunk to ground the final answer.",
            )
            if read_record.status.value == "success":
                citations.append(selected["chunk_id"])
                return read_record.output
            return None
        return None

    def _select_chunk(
        self,
        task: TaskSpec,
        search_results: list[dict[str, Any]],
        config: AgentConfig,
        require_matching: bool,
    ) -> dict[str, Any] | None:
        if not search_results:
            return None
        if config.strategy == StrategyType.BASELINE:
            return search_results[0]
        if not require_matching:
            return search_results[0]
        for result in search_results:
            if result["chunk_id"] in task.reference_chunk_ids:
                return result
        return search_results[0]

    def _compose_answer(self, task: TaskSpec, state: ExecutionState, citations: list[str]) -> Any:
        canonical = task.metadata.get("canonical_answer", task.expected_answer)
        if task.answer_format == AnswerFormat.TEXT:
            if set(task.reference_chunk_ids).intersection(citations):
                return canonical
            if state.web_result:
                return "Web lookup returned unverified context; no grounded answer available."
            return "No grounded answer found."

        answer: dict[str, Any] = {}
        for key, value in canonical.items():
            if key == "case_id" and state.case_data:
                answer[key] = value
            elif key in {"owner_team", "runbook_url", "fallback_policy"} and state.sql_rows:
                answer[key] = value
            elif key in {"priority", "region"} and state.case_data:
                answer[key] = value
            elif key == "per_worker_rps" and state.calculations:
                answer[key] = value
            elif key == "supporting_pattern":
                if len(task.reference_chunk_ids) > 1 and task.reference_chunk_ids[1] in citations:
                    answer[key] = value
            elif key == "recommended_feature":
                if task.reference_chunk_ids and task.reference_chunk_ids[0] in citations:
                    answer[key] = value
            else:
                answer[key] = value
        return answer

    def _repair_answer(
        self,
        task: TaskSpec,
        config: AgentConfig,
        state: ExecutionState,
        citations: list[str],
        final_answer: Any,
        call_tool,
        add_step,
    ) -> Any:
        if task.answer_format != AnswerFormat.JSON:
            return final_answer

        needed_citations = [chunk_id for chunk_id in task.reference_chunk_ids if chunk_id not in citations]
        if needed_citations:
            hint = task.metadata.get("secondary_query") or needed_citations[0].replace("-", " ")
            chunk = self._retrieve_chunk(
                task=task,
                config=config,
                search_hints=[hint],
                citations=citations,
                call_tool=call_tool,
                add_step=add_step,
                require_matching=True,
            )
            if chunk:
                state.chunk_cache[chunk["chunk_id"]] = chunk

        if task.metadata.get("sql_query") and not state.sql_rows:
            retry = call_tool("sql_query", {"sql": task.metadata["sql_query"]}, "Repair missing SQL-backed fields.")
            if retry.status.value == "success":
                state.sql_rows = retry.output

        if task.metadata.get("calculation") and not state.calculations:
            retry = call_tool(
                "calculator",
                {"expression": task.metadata["calculation"]},
                "Repair the missing computed field.",
            )
            if retry.status.value == "success":
                state.calculations = retry.output

        if task.metadata.get("case_id") and not state.case_data:
            retry = call_tool("case_api", {"case_id": task.metadata["case_id"]}, "Repair the missing case payload.")
            if retry.status.value == "success":
                state.case_data = retry.output

        repaired = dict(final_answer) if isinstance(final_answer, dict) else {}
        repaired.update(self._compose_answer(task, state, citations))
        add_step(
            phase="verification",
            thought="Repair completed; merge recovered fields back into the answer payload.",
            action_type="repair",
            action_payload={"repaired_fields": sorted(repaired.keys())},
            observation=repaired,
            latency_ms=25.0,
        )
        return repaired


class LiveToolCallingAgentRunner(AgentRunner):
    def __init__(
        self,
        settings: Settings,
        tools: ToolRegistry,
        llm_client: DashScopeChatClient | OllamaChatClient | None,
        provider_name: str,
    ) -> None:
        self.settings = settings
        self.tools = tools
        self.llm_client = llm_client
        self.provider_name = provider_name

    def run(self, task: TaskSpec, config: AgentConfig, experiment_id: str) -> RunRecord:
        started_at = utc_now()
        steps: list[StepRecord] = []
        citations: list[str] = []
        step_index = 1
        usage_tokens = 0
        fault_controller = FaultController(task.faults)
        final_answer: Any = None
        evidence_reminder_used = False
        if self.llm_client is None:
            steps.append(
                StepRecord(
                    step_index=step_index,
                    phase="execution",
                    thought=f"{self.provider_name} 客户端未配置，无法执行实时模型运行。",
                    action_type="llm_error",
                    observation={"error": f"{self.provider_name} client is not configured"},
                    status="error",
                    latency_ms=1.0,
                )
            )
            return build_run_record(
                task=task,
                config=config,
                experiment_id=experiment_id,
                started_at=started_at,
                steps=steps,
                citations=citations,
                final_answer=None,
                extra_metrics={"provider": self.provider_name},
            )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt(task, config)},
            {"role": "user", "content": self._user_prompt(task)},
        ]
        tool_schemas = self.tools.tool_schemas(enable_web_lookup=config.enable_web_lookup)

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

        if config.strategy != StrategyType.BASELINE:
            plan = self._plan(task, config)
            add_step(
                phase="planning",
                thought="模型生成了执行计划。",
                action_type="plan",
                action_payload={"strategy": config.strategy.value},
                observation=plan,
                latency_ms=30.0,
            )
            messages.append({"role": "assistant", "content": f"计划：{plan.get('steps', [])}"})

        try:
            for _ in range(config.max_steps):
                response = self.llm_client.chat(
                    messages=messages,
                    model=config.executor_model or self.settings.executor_model,
                    tools=tool_schemas,
                    tool_choice="auto",
                    temperature=0.0,
                    parallel_tool_calls=False,
                )
                usage_tokens += self._usage_tokens(response.usage)
                assistant_message = response.message
                text = message_text(assistant_message)
                tool_calls = parse_tool_calls(assistant_message)

                if tool_calls:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": assistant_message.get("content") or "",
                            "tool_calls": assistant_message.get("tool_calls", []),
                        }
                    )
                    for tool_call in tool_calls:
                        record = self.tools.execute(tool_call.name, tool_call.arguments, fault_controller)
                        if tool_call.name == "doc_read" and record.status.value == "success":
                            chunk_id = str(tool_call.arguments.get("chunk_id", ""))
                            if chunk_id:
                                citations.append(chunk_id)
                        add_step(
                            phase="execution",
                            thought=text or f"模型请求调用工具 {tool_call.name}。",
                            action_type="tool",
                            action_payload={"tool_name": tool_call.name, "arguments": tool_call.arguments},
                            observation=record.output if record.error is None else {"error": record.error},
                            status="ok" if record.status.value == "success" else record.status.value,
                            latency_ms=record.latency_ms,
                            tool_call=record,
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.tool_call_id,
                                "name": tool_call.name,
                                "content": self._tool_content(record),
                            }
                        )
                    continue

                final_answer = self._parse_final_answer(task, text)
                if (
                    not citations
                    and "doc_read" in task.required_tools
                    and not evidence_reminder_used
                ):
                    evidence_reminder_used = True
                    messages.append({"role": "assistant", "content": text})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "你还没有调用 doc_read 获取最终证据，也没有形成 chunk 级引用。"
                                "请继续使用工具，不要直接结束。"
                            ),
                        }
                    )
                    add_step(
                        phase="planning",
                        thought="模型试图在没有读取最终证据的情况下结束，本轮要求其继续调用 doc_read。",
                        action_type="continue",
                        action_payload={"reason": "missing_doc_read"},
                        observation={"draft_answer": final_answer},
                        status="retry",
                        latency_ms=10.0,
                    )
                    final_answer = None
                    continue
                add_step(
                    phase="execution",
                    thought="模型返回最终答案。",
                    action_type="assistant",
                    action_payload={"raw_text": text},
                    observation=final_answer,
                    latency_ms=25.0,
                )
                break
        except (LLMClientError, Exception) as exc:
            add_step(
                phase="execution",
                thought=f"{self.provider_name} 调用失败，记录错误并结束本次运行。",
                action_type="llm_error",
                action_payload={},
                observation={"error": str(exc)},
                status="error",
                latency_ms=10.0,
            )

        if config.enable_verifier and final_answer is not None:
            verification = self._verify(task, config, final_answer, citations)
            add_step(
                phase="verification",
                thought="校验模型检查答案字段、引用和格式。",
                action_type="verify",
                action_payload={"required_fields": task.validation.required_fields},
                observation=verification,
                latency_ms=30.0,
            )
            if not verification.get("is_valid", True):
                repair_text = self._repair(task, config, final_answer, citations, verification)
                try:
                    final_answer = self._parse_final_answer(task, repair_text)
                except Exception:
                    pass
                add_step(
                    phase="verification",
                    thought="校验未通过，已发起一次修复。",
                    action_type="repair",
                    action_payload={"issues": verification.get("issues", [])},
                    observation=final_answer,
                    latency_ms=30.0,
                )

        add_step(
            phase="final",
            thought="返回最终答案与引用。",
            action_type="final",
            action_payload={"answer_format": task.answer_format.value},
            observation={"answer": final_answer, "citations": list(dict.fromkeys(citations))},
            latency_ms=10.0,
        )

        return build_run_record(
            task=task,
            config=config,
            experiment_id=experiment_id,
            started_at=started_at,
            steps=steps,
            citations=citations,
            final_answer=final_answer,
            total_tokens_override=usage_tokens or None,
            extra_metrics={
                "faults_triggered": fault_controller.consumed,
                "tool_count": len([step for step in steps if step.tool_call]),
                "provider": self.provider_name,
            },
        )

    def _system_prompt(self, task: TaskSpec, config: AgentConfig) -> str:
        parts = [
            "你是企业知识库 Agent，必须优先使用工具而不是猜测。",
            "所有工具参数都必须是合法 JSON，执行前要自我检查。",
            "答案必须严格基于工具观察结果，不能编造。",
            f"任务类型：{task.category.value}。",
        ]
        if task.answer_format == AnswerFormat.JSON:
            parts.append("最终答案必须是 JSON 对象，且不要输出 Markdown 代码块。")
        else:
            parts.append("最终答案输出简洁文本，尽量复用文档中的原始术语，不要加多余前缀。")
        if "doc_read" in task.required_tools:
            parts.append("在给出最终答案前，至少调用一次 doc_read 读取最终证据。")
        if config.strategy == StrategyType.PLANNER:
            parts.append("先形成计划，再执行；遇到失败时允许调整路径。")
        if config.strategy == StrategyType.VERIFIER:
            parts.append("输出前要自查字段完整性、引用充分性和格式正确性。")
        return "\n".join(parts)

    def _user_prompt(self, task: TaskSpec) -> str:
        hint_text = "；".join(task.search_hints) if task.search_hints else "无"
        citation_text = "、".join(task.reference_chunk_ids) if task.reference_chunk_ids else "无"
        return (
            f"任务ID：{task.task_id}\n"
            f"任务标题：{task.title}\n"
            f"任务要求：{task.prompt}\n"
            f"建议检索提示：{hint_text}\n"
            f"期望引用 chunk_id：{citation_text}\n"
            "如果需要返回 JSON，请直接返回 JSON 对象。"
        )

    def _plan(self, task: TaskSpec, config: AgentConfig) -> dict[str, Any]:
        try:
            return self.llm_client.complete_json(
                system_prompt="你是规划器。请只输出 JSON，给出完成任务的步骤列表。",
                user_prompt=(
                    f"任务：{task.prompt}\n"
                    f"可用工具：{', '.join(task.required_tools)}\n"
                    "输出格式：{\"steps\": [\"...\"]}"
                ),
                model=config.planner_model or self.settings.planner_model,
                json_schema={
                    "type": "object",
                    "properties": {
                        "steps": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["steps"],
                    "additionalProperties": False,
                },
            )
        except Exception:
            return {"steps": ["搜索相关文档", "按需调用结构化工具", "整理答案并返回"]}

    def _verify(self, task: TaskSpec, config: AgentConfig, final_answer: Any, citations: list[str]) -> dict[str, Any]:
        required_fields = task.validation.required_fields if task.answer_format == AnswerFormat.JSON else []
        try:
            return self.llm_client.complete_json(
                system_prompt="你是答案校验器。只输出 JSON。",
                user_prompt=(
                    f"任务：{task.prompt}\n"
                    f"答案：{final_answer}\n"
                    f"引用：{citations}\n"
                    f"必需字段：{required_fields}\n"
                    f"必需引用：{task.reference_chunk_ids}\n"
                    '输出格式：{"is_valid": true/false, "issues": [], "missing_fields": [], "missing_citations": []}'
                ),
                model=config.verifier_model or self.settings.verifier_model,
                json_schema={
                    "type": "object",
                    "properties": {
                        "is_valid": {"type": "boolean"},
                        "issues": {"type": "array", "items": {"type": "string"}},
                        "missing_fields": {"type": "array", "items": {"type": "string"}},
                        "missing_citations": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["is_valid", "issues", "missing_fields", "missing_citations"],
                    "additionalProperties": False,
                },
            )
        except Exception:
            return {"is_valid": True, "issues": [], "missing_fields": [], "missing_citations": []}

    def _repair(
        self,
        task: TaskSpec,
        config: AgentConfig,
        final_answer: Any,
        citations: list[str],
        verification: dict[str, Any],
    ) -> str:
        response = self.llm_client.chat(
            model=config.executor_model or self.settings.executor_model,
            messages=[
                {"role": "system", "content": "你是修复器。根据校验问题修复答案，必要时补齐 JSON 字段，保持内容简洁。"},
                {
                    "role": "user",
                    "content": (
                        f"原任务：{task.prompt}\n"
                        f"原答案：{final_answer}\n"
                        f"已有引用：{citations}\n"
                        f"校验问题：{verification}\n"
                        "请直接返回修复后的最终答案。"
                    ),
                },
            ],
            temperature=0.0,
        )
        return message_text(response.message)

    def _parse_final_answer(self, task: TaskSpec, text: str) -> Any:
        if task.answer_format == AnswerFormat.JSON:
            return parse_json_content(text)
        return text.strip()

    def _usage_tokens(self, usage: dict[str, Any]) -> int:
        return int(
            usage.get("total_tokens")
            or usage.get("totalTokens")
            or usage.get("output_tokens", 0) + usage.get("input_tokens", 0)
            or 0
        )

    def _tool_content(self, record) -> str:
        if record.error:
            return json_dumps({"status": "error", "error": record.error})
        return json_dumps(record.output)


def build_run_record(
    *,
    task: TaskSpec,
    config: AgentConfig,
    experiment_id: str,
    started_at,
    steps: list[StepRecord],
    citations: list[str],
    final_answer: Any,
    total_tokens_override: int | None = None,
    extra_metrics: dict[str, Any] | None = None,
) -> RunRecord:
    total_tokens = total_tokens_override or token_estimate(
        [
            task.prompt,
            *[step.thought for step in steps],
            str(final_answer),
        ]
    )
    total_latency_ms = round(sum(step.latency_ms for step in steps), 2)
    metrics = {
        "step_count": len(steps),
        **(extra_metrics or {}),
    }
    return RunRecord(
        experiment_id=experiment_id,
        task_id=task.task_id,
        config_id=config.config_id,
        strategy=config.strategy,
        status=RunStatus.COMPLETED if final_answer not in (None, {}, "") else RunStatus.FAILED,
        final_answer=final_answer,
        citations=list(dict.fromkeys(citations)),
        steps=steps,
        evaluation=EvalResult(success=False, score=0.0),
        total_latency_ms=total_latency_ms,
        total_tokens=total_tokens,
        total_cost_estimate=cost_estimate(total_tokens),
        metrics=metrics,
        started_at=started_at,
        finished_at=utc_now(),
    )
