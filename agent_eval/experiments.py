from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent_eval.config import Settings
from agent_eval.evaluator import evaluate_run
from agent_eval.llm import DashScopeChatClient, OllamaChatClient
from agent_eval.models import ExperimentRunRequest, ExperimentSummary, TaskSpec
from agent_eval.presets import config_presets, resolve_config_ids, resolve_task_ids, task_presets
from agent_eval.runners import RunnerRouter
from agent_eval.seed import build_seed_payload
from agent_eval.storage import Storage
from agent_eval.tools import ToolRegistry


@dataclass
class PlatformServices:
    settings: Settings
    storage: Storage
    tools: ToolRegistry
    runner: RunnerRouter


def build_services(
    settings: Settings,
    dashscope_client: DashScopeChatClient | None = None,
    ollama_client: OllamaChatClient | None = None,
) -> PlatformServices:
    storage = Storage(settings.db_path)
    storage.init_db()
    if settings.auto_seed and not storage.has_seed_data():
        payload = build_seed_payload(
            settings.corpus_dir,
            include_live_configs=settings.live_qwen_enabled,
            include_ollama_configs=settings.live_ollama_enabled,
            planner_model=settings.planner_model,
            executor_model=settings.executor_model,
            verifier_model=settings.verifier_model,
        )
        storage.seed_reference_data(**payload)
    tools = ToolRegistry(storage, settings)
    runner = RunnerRouter(
        settings,
        tools,
        dashscope_client=dashscope_client,
        ollama_client=ollama_client,
    )
    return PlatformServices(settings=settings, storage=storage, tools=tools, runner=runner)


class ExperimentService:
    def __init__(self, services: PlatformServices) -> None:
        self.services = services

    @property
    def storage(self) -> Storage:
        return self.services.storage

    def bootstrap(self, force_seed: bool = False) -> None:
        self.storage.init_db()
        if force_seed or not self.storage.has_seed_data():
            payload = build_seed_payload(
                self.services.settings.corpus_dir,
                include_live_configs=self.services.settings.live_qwen_enabled,
                include_ollama_configs=self.services.settings.live_ollama_enabled,
                planner_model=self.services.settings.planner_model,
                executor_model=self.services.settings.executor_model,
                verifier_model=self.services.settings.verifier_model,
            )
            self.storage.seed_reference_data(**payload)

    def run_experiment(self, request: ExperimentRunRequest | None = None) -> dict[str, Any]:
        request = request or ExperimentRunRequest()
        configs = self.storage.list_agent_configs()
        tasks = self.storage.list_tasks()
        config_ids = resolve_config_ids(configs, request.config_preset, request.config_ids)
        task_ids = resolve_task_ids(tasks, request.task_preset, request.task_ids)
        if request.limit:
            task_ids = task_ids[: request.limit]
        if not config_ids:
            raise ValueError("No agent configs selected for this experiment")
        if not task_ids:
            raise ValueError("No tasks selected for this experiment")

        task_map = {task.task_id: task for task in tasks}
        config_map = {config.config_id: config for config in configs}
        selected_tasks = [task_map[task_id] for task_id in task_ids]
        selected_configs = [config_map[config_id] for config_id in config_ids]

        experiment_id = self.storage.create_experiment(
            config_ids,
            task_ids,
            config_preset=request.config_preset,
            task_preset=request.task_preset,
        )
        run_records = []
        for config in selected_configs:
            for task in selected_tasks:
                run = self.services.runner.run(task=task, config=config, experiment_id=experiment_id)
                run.evaluation = evaluate_run(task, run)
                self.storage.save_run(run)
                run_records.append(run)

        metrics = self._aggregate_metrics(run_records)
        self.storage.update_experiment_metrics(experiment_id, metrics)
        summary = ExperimentSummary(
            experiment_id=experiment_id,
            config_preset=request.config_preset,
            task_preset=request.task_preset,
            config_ids=config_ids,
            task_ids=task_ids,
            total_runs=len(run_records),
            metrics=metrics,
        )
        return summary.model_dump(mode="json")

    def leaderboard(self) -> list[dict[str, Any]]:
        return [entry.model_dump(mode="json") for entry in self.storage.list_leaderboard()]

    def experiments(self, limit: int = 20) -> list[dict[str, Any]]:
        return self.storage.list_experiments(limit=limit)

    def tasks(self) -> list[dict[str, Any]]:
        return [task.model_dump(mode="json") for task in self.storage.list_tasks()]

    def failures(self) -> list[dict[str, Any]]:
        return [failure.model_dump(mode="json") for failure in self.storage.list_failures()]

    def run_detail(self, run_id: str) -> dict[str, Any]:
        return self.storage.get_run(run_id)

    def experiment_detail(self, experiment_id: str) -> dict[str, Any]:
        payload = self.storage.get_experiment(experiment_id)
        payload["task_presets"] = task_presets(self.storage.list_tasks())
        payload["config_presets"] = config_presets(self.storage.list_agent_configs())
        return payload

    def dashboard(self) -> dict[str, Any]:
        summary = self.storage.dashboard_summary()
        latest_experiment = self.storage.latest_experiment()
        recent_experiments = self.experiments(limit=5)
        leaderboard = self.leaderboard()
        failures = self.failures()
        tasks = self.storage.list_tasks()
        configs = self.storage.list_agent_configs()
        return {
            "summary": summary,
            "latest_experiment": latest_experiment,
            "recent_experiments": recent_experiments,
            "leaderboard": leaderboard,
            "failures": failures,
            "task_presets": task_presets(tasks),
            "config_presets": config_presets(configs),
            "runtime": {
                "live_qwen_enabled": self.services.settings.live_qwen_enabled,
                "live_ollama_enabled": self.services.settings.live_ollama_enabled,
                "planner_model": self.services.settings.planner_model,
                "executor_model": self.services.settings.executor_model,
                "verifier_model": self.services.settings.verifier_model,
                "ollama_base_url": self.services.settings.ollama_base_url,
            },
        }

    def _aggregate_metrics(self, runs: list[Any]) -> dict[str, Any]:
        if not runs:
            return {}
        total = len(runs)
        success_count = sum(1 for run in runs if run.evaluation.success)
        avg_latency = sum(run.total_latency_ms for run in runs) / total
        avg_tokens = sum(run.total_tokens for run in runs) / total
        avg_cost = sum(run.total_cost_estimate for run in runs) / total
        avg_recovery = sum(run.evaluation.recovery_rate for run in runs) / total
        return {
            "success_rate": round(success_count / total, 4),
            "avg_latency_ms": round(avg_latency, 2),
            "avg_tokens": round(avg_tokens, 2),
            "avg_cost": round(avg_cost, 6),
            "recovery_rate": round(avg_recovery, 4),
        }
