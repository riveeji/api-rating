from __future__ import annotations

from agent_eval.seed import build_seed_payload


def test_seed_payload_counts(settings) -> None:
    payload = build_seed_payload(settings.corpus_dir)
    assert len(payload["documents"]) == 3
    assert len(payload["chunks"]) == 18
    assert len(payload["tasks"]) == 50
    assert len(payload["agent_configs"]) == 3


def test_seed_payload_can_include_live_configs(settings) -> None:
    payload = build_seed_payload(
        settings.corpus_dir,
        include_live_configs=True,
        include_ollama_configs=True,
        planner_model="qwen-plus",
        executor_model="qwen-plus",
        verifier_model="qwen-max",
    )
    assert len(payload["agent_configs"]) == 9
