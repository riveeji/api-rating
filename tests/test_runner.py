from __future__ import annotations

from agent_eval.evaluator import evaluate_run
from agent_eval.llm import LLMResponse
from agent_eval.models import AgentConfig, EvalResult, RunRecord, RunStatus, StrategyType
from agent_eval.runners import RunnerRouter
from agent_eval.tools import FaultController


def test_sql_query_rejects_writes(experiment_service) -> None:
    tool_record = experiment_service.services.tools.execute(
        "sql_query",
        {"sql": "UPDATE service_catalog SET owner_team = 'x'"},
        fault_controller=FaultController([]),
    )
    assert tool_record.status.value == "error"


def test_recovery_strategy_beats_baseline(experiment_service) -> None:
    task = experiment_service.storage.get_task("TASK-RC-001")
    baseline = experiment_service.storage.get_agent_config("baseline_heuristic")
    planner = experiment_service.storage.get_agent_config("planner_heuristic")

    baseline_run = experiment_service.services.runner.run(task, baseline, "exp_test")
    baseline_run.evaluation = evaluate_run(task, baseline_run)
    planner_run = experiment_service.services.runner.run(task, planner, "exp_test")
    planner_run.evaluation = evaluate_run(task, planner_run)

    assert not baseline_run.evaluation.success
    assert planner_run.evaluation.success


class FakeDashScopeClient:
    def __init__(self) -> None:
        self._chat_calls = 0

    def complete_json(self, **kwargs):
        return {"steps": ["搜索文档", "读取片段", "输出答案"]}

    def chat(self, **kwargs):
        self._chat_calls += 1
        if self._chat_calls == 1:
            return LLMResponse(
                message={
                    "role": "assistant",
                    "content": "先搜索相关文档。",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "doc_search", "arguments": "{\"query\":\"response model\"}"},
                        }
                    ],
                },
                usage={"total_tokens": 40},
                raw={},
            )
        if self._chat_calls == 2:
            return LLMResponse(
                message={
                    "role": "assistant",
                    "content": "读取命中的文档片段。",
                    "tool_calls": [
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {"name": "doc_read", "arguments": "{\"chunk_id\":\"fastapi-response-model\"}"},
                        }
                    ],
                },
                usage={"total_tokens": 35},
                raw={},
            )
        return LLMResponse(
            message={
                "role": "assistant",
                "content": "Use response_model to validate, serialize, and filter the response payload.",
            },
            usage={"total_tokens": 25},
            raw={},
        )


def test_dashscope_runner_can_complete_single_hop_task(settings, experiment_service) -> None:
    router = RunnerRouter(
        settings,
        experiment_service.services.tools,
        ollama_client=FakeDashScopeClient(),
    )
    task = experiment_service.storage.get_task("TASK-SH-001")
    config = AgentConfig(
        config_id="baseline_ollama_live_test",
        display_name="Ollama 实时基线测试",
        strategy=StrategyType.BASELINE,
        model_provider="ollama",
        executor_model="qwen3:8b",
        max_steps=6,
    )

    run = router.run(task, config, "exp_test")
    run.evaluation = evaluate_run(task, run)

    assert run.evaluation.success
    assert "fastapi-response-model" in run.citations


def test_text_evaluator_accepts_grounded_equivalent_answer(experiment_service) -> None:
    task = experiment_service.storage.get_task("TASK-SH-001")
    run = RunRecord(
        experiment_id="exp_test",
        task_id=task.task_id,
        config_id="baseline_ollama_live",
        strategy=StrategyType.BASELINE,
        status=RunStatus.COMPLETED,
        final_answer="In FastAPI, the response_model feature validates and filters the returned payload before the response is sent.",
        citations=["fastapi-response-model"],
        evaluation=EvalResult(success=False, score=0.0),
    )

    result = evaluate_run(task, run)

    assert result.success
    assert result.citation_match
