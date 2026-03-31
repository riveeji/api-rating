from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "LLM Agent 评测与优化平台"
    debug: bool = False
    data_dir: Path = Field(default_factory=lambda: Path("data"))
    db_path: Path = Field(default_factory=lambda: Path("data") / "agent_eval.db")
    corpus_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent / "assets" / "corpus"
    )
    templates_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent / "templates"
    )
    static_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent / "static")
    dashscope_api_key: str | None = None
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ollama_base_url: str = "http://127.0.0.1:11434/v1"
    ollama_api_key: str = "ollama"
    planner_model: str = "qwen-plus"
    executor_model: str = "qwen-plus"
    verifier_model: str = "qwen-max"
    use_llm_default: bool = False
    include_live_qwen_configs: bool = False
    include_live_ollama_configs: bool = False
    enable_web_lookup: bool = True
    auto_seed: bool = True
    request_timeout_seconds: float = 20.0
    ollama_request_timeout_seconds: float = 180.0

    model_config = SettingsConfigDict(
        env_prefix="AGENT_EVAL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def live_qwen_enabled(self) -> bool:
        return bool(self.dashscope_api_key) and self.include_live_qwen_configs

    @property
    def live_ollama_enabled(self) -> bool:
        return bool(self.include_live_ollama_configs)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
