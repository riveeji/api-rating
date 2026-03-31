from __future__ import annotations

import argparse
import json

from agent_eval.config import get_settings
from agent_eval.experiments import ExperimentService, build_services
from agent_eval.models import ExperimentRunRequest


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent evaluation platform CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init-db", help="Initialize the SQLite schema")
    subparsers.add_parser("seed-demo", help="Seed the demo corpus, cases, tasks, and agent configs")

    benchmark = subparsers.add_parser("run-benchmark", help="Run a benchmark and store experiment results")
    benchmark.add_argument("--limit", type=int, default=None, help="Limit the number of tasks after preset resolution")
    benchmark.add_argument(
        "--task-preset",
        choices=["all", "single_hop", "multi_step", "recovery"],
        default=None,
        help="Run a predefined task slice",
    )
    benchmark.add_argument(
        "--config-preset",
        choices=["all", "heuristic", "dashscope_live", "ollama_live"],
        default=None,
        help="Run a predefined config slice",
    )
    benchmark.add_argument("--task-id", action="append", default=[], help="Append an explicit task id")
    benchmark.add_argument("--config-id", action="append", default=[], help="Append an explicit config id")

    args = parser.parse_args()
    settings = get_settings()
    experiment_service = ExperimentService(build_services(settings))

    if args.command == "init-db":
        experiment_service.bootstrap(force_seed=False)
        print(f"Initialized database at {settings.db_path}")
        return

    if args.command == "seed-demo":
        experiment_service.bootstrap(force_seed=True)
        print(f"Seeded demo data into {settings.db_path}")
        return

    if args.command == "run-benchmark":
        summary = experiment_service.run_experiment(
            ExperimentRunRequest(
                config_preset=args.config_preset,
                task_preset=args.task_preset,
                config_ids=args.config_id,
                task_ids=args.task_id,
                limit=args.limit,
            )
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
