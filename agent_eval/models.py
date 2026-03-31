from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(UTC)


class TaskCategory(str, Enum):
    SINGLE_HOP = "single_hop_qa"
    MULTI_STEP = "multi_step"
    RECOVERY = "recovery"


class StrategyType(str, Enum):
    BASELINE = "baseline_tool_calling"
    PLANNER = "planner_executor"
    VERIFIER = "planner_executor_verifier"


class AnswerFormat(str, Enum):
    TEXT = "text"
    JSON = "json"


class ValidationMode(str, Enum):
    EXACT_TEXT = "exact_text"
    EXACT_JSON = "exact_json"
    FIELD_MATCH = "field_match"
    SQL_RESULT_MATCH = "sql_result_match"


class RunStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class ToolCallStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    EMPTY = "empty"
    INVALID = "invalid"


class FailureType(str, Enum):
    WRONG_RETRIEVAL = "wrong_retrieval"
    BAD_TOOL_CHOICE = "bad_tool_choice"
    TOOL_ERROR_NOT_RECOVERED = "tool_error_not_recovered"
    FORMAT_VIOLATION = "format_violation"
    HALLUCINATED_ANSWER = "hallucinated_answer"
    NONE = "none"


class FaultMode(str, Enum):
    ERROR_ONCE = "error_once"
    EMPTY_ONCE = "empty_once"
    TIMEOUT_ONCE = "timeout_once"


class FaultInjection(BaseModel):
    tool_name: str
    mode: FaultMode
    message: str | None = None


class ValidationSpec(BaseModel):
    mode: ValidationMode
    required_fields: list[str] = Field(default_factory=list)
    citation_chunk_ids: list[str] = Field(default_factory=list)
    sql_result: list[dict[str, Any]] | None = None
    accepted_answers: list[str] = Field(default_factory=list)
    required_term_groups: list[list[str]] = Field(default_factory=list)


class TaskSpec(BaseModel):
    task_id: str
    title: str
    category: TaskCategory
    prompt: str
    answer_format: AnswerFormat
    expected_answer: Any
    required_tools: list[str] = Field(default_factory=list)
    reference_chunk_ids: list[str] = Field(default_factory=list)
    search_hints: list[str] = Field(default_factory=list)
    validation: ValidationSpec
    faults: list[FaultInjection] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentConfig(BaseModel):
    config_id: str
    display_name: str
    strategy: StrategyType
    model_provider: str = "heuristic"
    planner_model: str | None = None
    executor_model: str | None = None
    verifier_model: str | None = None
    max_steps: int = 8
    enable_verifier: bool = False
    allow_replan: bool = False
    enable_web_lookup: bool = False
    prompt_variant: str = "default"
    settings: dict[str, Any] = Field(default_factory=dict)


class ToolCallRecord(BaseModel):
    tool_call_id: str = Field(default_factory=lambda: f"tool_{uuid4().hex}")
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    status: ToolCallStatus
    output: Any = None
    error: str | None = None
    latency_ms: float = 0.0
    created_at: datetime = Field(default_factory=utc_now)


class StepRecord(BaseModel):
    step_id: str = Field(default_factory=lambda: f"step_{uuid4().hex}")
    step_index: int
    phase: str
    thought: str
    action_type: str
    action_payload: dict[str, Any] = Field(default_factory=dict)
    observation: Any = None
    status: str = "ok"
    latency_ms: float = 0.0
    tool_call: ToolCallRecord | None = None
    created_at: datetime = Field(default_factory=utc_now)


class EvalResult(BaseModel):
    success: bool
    score: float
    exact_match: bool = False
    field_match: bool = False
    citation_match: bool = False
    sql_result_match: bool = False
    tool_usage_accuracy: float = 0.0
    invalid_action_rate: float = 0.0
    tool_error_rate: float = 0.0
    recovery_rate: float = 0.0
    failure_type: FailureType = FailureType.NONE
    details: dict[str, Any] = Field(default_factory=dict)


class RunRecord(BaseModel):
    run_id: str = Field(default_factory=lambda: f"run_{uuid4().hex}")
    experiment_id: str
    task_id: str
    config_id: str
    strategy: StrategyType
    status: RunStatus
    final_answer: Any = None
    citations: list[str] = Field(default_factory=list)
    steps: list[StepRecord] = Field(default_factory=list)
    evaluation: EvalResult
    total_latency_ms: float = 0.0
    total_tokens: int = 0
    total_cost_estimate: float = 0.0
    metrics: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime = Field(default_factory=utc_now)


class ExperimentRunRequest(BaseModel):
    config_preset: str | None = None
    task_preset: str | None = None
    config_ids: list[str] = Field(default_factory=list)
    task_ids: list[str] = Field(default_factory=list)
    limit: int | None = None


class AssistantAskRequest(BaseModel):
    prompt: str
    session_id: str | None = None
    config_id: str | None = None


class ExperimentSummary(BaseModel):
    experiment_id: str
    created_at: datetime = Field(default_factory=utc_now)
    config_preset: str | None = None
    task_preset: str | None = None
    config_ids: list[str] = Field(default_factory=list)
    task_ids: list[str] = Field(default_factory=list)
    total_runs: int = 0
    metrics: dict[str, Any] = Field(default_factory=dict)


class LeaderboardEntry(BaseModel):
    config_id: str
    display_name: str
    strategy: StrategyType
    success_rate: float
    avg_latency_ms: float
    avg_tokens: float
    avg_cost: float
    tool_error_rate: float
    invalid_action_rate: float
    recovery_rate: float
    avg_steps: float
    total_runs: int


class FailureRecord(BaseModel):
    run_id: str
    task_id: str
    config_id: str
    strategy: StrategyType
    failure_type: FailureType
    title: str
    prompt: str
    details: dict[str, Any] = Field(default_factory=dict)


class AssistantSessionSummary(BaseModel):
    session_id: str
    title: str
    config_id: str | None = None
    message_count: int = 0
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    last_message_preview: str | None = None


class AssistantMessageRecord(BaseModel):
    message_id: str = Field(default_factory=lambda: f"msg_{uuid4().hex}")
    session_id: str
    role: str
    content: str
    config_id: str | None = None
    citations: list[str] = Field(default_factory=list)
    run_payload: dict[str, Any] | None = None
    created_at: datetime = Field(default_factory=utc_now)
