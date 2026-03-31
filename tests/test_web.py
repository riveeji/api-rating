from __future__ import annotations

from fastapi.testclient import TestClient

from agent_eval.web import create_app


def test_web_routes_and_benchmark(settings) -> None:
    client = TestClient(create_app(settings))

    root = client.get("/")
    tasks = client.get("/tasks")
    benchmark = client.post("/experiments/run", json={"task_preset": "single_hop", "limit": 2})
    leaderboard = client.get("/leaderboard?format=json")
    experiments = client.get("/experiments?format=json")

    assert root.status_code == 200
    assert tasks.status_code == 200
    assert len(tasks.json()) == 50
    assert benchmark.status_code == 200
    assert benchmark.json()["task_preset"] == "single_hop"
    assert benchmark.json()["total_runs"] == 6
    assert leaderboard.status_code == 200
    assert len(leaderboard.json()) == 3
    assert experiments.status_code == 200
    assert len(experiments.json()) == 1


def test_experiment_detail_and_invalid_preset(settings) -> None:
    client = TestClient(create_app(settings))
    benchmark = client.post(
        "/experiments/run",
        json={"task_preset": "recovery", "config_preset": "heuristic", "limit": 1},
    )

    assert benchmark.status_code == 200
    experiment_id = benchmark.json()["experiment_id"]

    detail = client.get(f"/experiments/{experiment_id}?format=json")
    assert detail.status_code == 200
    assert detail.json()["experiment"]["task_preset"] == "recovery"
    assert detail.json()["experiment"]["config_preset"] == "heuristic"
    assert detail.json()["run_count"] == 3

    invalid = client.post("/experiments/run", json={"task_preset": "does_not_exist"})
    assert invalid.status_code == 400

    explicit = client.post(
        "/experiments/run",
        json={"task_ids": ["TASK-SH-001"], "config_ids": ["baseline_heuristic"]},
    )
    assert explicit.status_code == 200
    assert explicit.json()["total_runs"] == 1
