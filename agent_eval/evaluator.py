from __future__ import annotations

from typing import Any

from agent_eval.models import EvalResult, FailureType, RunRecord, TaskSpec, ValidationMode
from agent_eval.utils import normalize_text


def evaluate_run(task: TaskSpec, run: RunRecord) -> EvalResult:
    tool_calls = [step.tool_call for step in run.steps if step.tool_call]
    used_tools = [tool.tool_name for tool in tool_calls]
    unique_used_tools = set(used_tools)
    required_tools = set(task.required_tools)
    expected_citations = set(task.validation.citation_chunk_ids or task.reference_chunk_ids)
    actual_citations = set(run.citations)

    exact_match = False
    field_match = False
    sql_result_match = False
    text_rule_match = False

    if task.answer_format.value == "text":
        exact_match = _text_match(task.expected_answer, run.final_answer, task.validation.accepted_answers)
        text_rule_match = _text_rule_match(run.final_answer, task.validation.required_term_groups)
    else:
        field_match = _field_match(task.expected_answer, run.final_answer, task.validation.required_fields)
        if task.validation.mode == ValidationMode.EXACT_JSON:
            exact_match = run.final_answer == task.expected_answer
        else:
            exact_match = field_match

    citation_match = expected_citations.issubset(actual_citations) if expected_citations else True
    sql_tool_calls = [tool for tool in tool_calls if tool.tool_name == "sql_query" and tool.status.value == "success"]
    sql_result_match = True if not any(tool.tool_name == "sql_query" for tool in tool_calls) else bool(sql_tool_calls)
    tool_usage_accuracy = round(
        len(required_tools.intersection(unique_used_tools)) / max(1, len(required_tools)),
        4,
    )
    invalid_action_rate = round(
        sum(1 for tool in tool_calls if tool.tool_name not in required_tools) / max(1, len(tool_calls)),
        4,
    )
    tool_error_rate = round(
        sum(1 for tool in tool_calls if tool.status.value == "error") / max(1, len(tool_calls)),
        4,
    )
    recovery_rate = _recovery_rate(task, tool_calls, run)
    primary_match = (exact_match or text_rule_match) if task.answer_format.value == "text" else field_match
    success = primary_match and citation_match
    failure_type = FailureType.NONE if success else _classify_failure(
        task=task,
        run=run,
        primary_match=primary_match,
        citation_match=citation_match,
        invalid_action_rate=invalid_action_rate,
        tool_error_rate=tool_error_rate,
    )

    score = round(
        (0.55 * float(primary_match))
        + (0.15 * float(citation_match))
        + (0.15 * tool_usage_accuracy)
        + (0.15 * recovery_rate),
        4,
    )
    return EvalResult(
        success=success,
        score=score,
        exact_match=exact_match,
        field_match=field_match,
        citation_match=citation_match,
        sql_result_match=sql_result_match,
        tool_usage_accuracy=tool_usage_accuracy,
        invalid_action_rate=invalid_action_rate,
        tool_error_rate=tool_error_rate,
        recovery_rate=recovery_rate,
        failure_type=failure_type,
        details={
            "required_tools": task.required_tools,
            "used_tools": used_tools,
            "required_citations": sorted(expected_citations),
            "actual_citations": sorted(actual_citations),
            "text_rule_match": text_rule_match,
            "required_term_groups": task.validation.required_term_groups,
        },
    )


def _text_match(expected: Any, actual: Any, accepted_answers: list[str] | None = None) -> bool:
    if not isinstance(actual, str):
        return False
    candidates = []
    if isinstance(expected, str):
        candidates.append(expected)
    candidates.extend(accepted_answers or [])
    normalized_actual = normalize_text(actual)
    for candidate in candidates:
        normalized_candidate = normalize_text(candidate)
        if normalized_candidate == normalized_actual:
            return True
        if normalized_candidate and normalized_candidate in normalized_actual:
            return True
    return False


def _text_rule_match(actual: Any, required_term_groups: list[list[str]]) -> bool:
    if not isinstance(actual, str):
        return False
    if not required_term_groups:
        return False
    normalized_actual = normalize_text(actual).replace("`", "")
    for group in required_term_groups:
        normalized_group = [normalize_text(term).replace("`", "") for term in group]
        if not any(term in normalized_actual for term in normalized_group):
            return False
    return True


def _field_match(expected: Any, actual: Any, required_fields: list[str]) -> bool:
    if not isinstance(expected, dict) or not isinstance(actual, dict):
        return False
    for field in required_fields:
        if field not in actual:
            return False
        if actual[field] != expected[field]:
            return False
    return True


def _recovery_rate(task: TaskSpec, tool_calls: list[Any], run: RunRecord) -> float:
    if not task.faults:
        return 1.0 if run.status.value == "completed" else 0.0
    recovered = 0
    for fault in task.faults:
        fault_calls = [tool for tool in tool_calls if tool.tool_name == fault.tool_name]
        if len(fault_calls) >= 2 and any(call.status.value == "success" for call in fault_calls[1:]):
            recovered += 1
        elif run.status.value == "completed" and run.final_answer:
            recovered += 1
    return round(recovered / max(1, len(task.faults)), 4)


def _classify_failure(
    task: TaskSpec,
    run: RunRecord,
    primary_match: bool,
    citation_match: bool,
    invalid_action_rate: float,
    tool_error_rate: float,
) -> FailureType:
    if tool_error_rate > 0:
        return FailureType.TOOL_ERROR_NOT_RECOVERED
    if invalid_action_rate > 0:
        return FailureType.BAD_TOOL_CHOICE
    if not citation_match:
        return FailureType.WRONG_RETRIEVAL
    if task.answer_format.value == "json" and not primary_match:
        return FailureType.FORMAT_VIOLATION
    if task.answer_format.value == "text" and not primary_match:
        return FailureType.HALLUCINATED_ANSWER
    return FailureType.HALLUCINATED_ANSWER
