from __future__ import annotations

from pathlib import Path

import pytest

from agent_eval.config import Settings
from agent_eval.experiments import ExperimentService, build_services


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    package_dir = Path(__file__).resolve().parents[1] / "agent_eval"
    return Settings(
        data_dir=tmp_path / "data",
        db_path=tmp_path / "data" / "agent_eval.db",
        corpus_dir=package_dir / "assets" / "corpus",
        templates_dir=package_dir / "templates",
        static_dir=package_dir / "static",
        auto_seed=True,
        enable_web_lookup=False,
    )


@pytest.fixture()
def experiment_service(settings: Settings) -> ExperimentService:
    service = ExperimentService(build_services(settings))
    service.bootstrap(force_seed=True)
    return service
