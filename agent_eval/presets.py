from __future__ import annotations

from collections.abc import Iterable

from agent_eval.models import AgentConfig, TaskCategory, TaskSpec


TASK_PRESET_DEFS = {
    "all": {
        "label": "全部任务",
        "description": "运行当前数据库中的全部任务。",
    },
    "single_hop": {
        "label": "单跳问答",
        "description": "仅运行文档检索问答任务。",
    },
    "multi_step": {
        "label": "多步工具调用",
        "description": "仅运行需要多工具协作的结构化任务。",
    },
    "recovery": {
        "label": "恢复与回退",
        "description": "仅运行带故障注入的恢复任务。",
    },
}

CONFIG_PRESET_DEFS = {
    "all": {
        "label": "全部配置",
        "description": "运行当前数据库中的全部 Agent 配置。",
    },
    "heuristic": {
        "label": "启发式策略",
        "description": "仅运行本地可复现实验用的启发式配置。",
    },
    "dashscope_live": {
        "label": "DashScope 实时模型",
        "description": "仅运行接入 DashScope/Qwen 的实时配置。",
    },
    "ollama_live": {
        "label": "Ollama 实时模型",
        "description": "仅运行本地 Ollama 实时配置。",
    },
}


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def resolve_task_ids(tasks: list[TaskSpec], preset: str | None, explicit_ids: list[str] | None = None) -> list[str]:
    explicit_ids = explicit_ids or []
    task_map = {task.task_id: task for task in tasks}
    if preset and preset not in TASK_PRESET_DEFS:
        raise ValueError(f"Unknown task preset: {preset}")

    if preset is None:
        selected = [] if explicit_ids else [task.task_id for task in tasks]
    elif preset == "all":
        selected = [task.task_id for task in tasks]
    elif preset == "single_hop":
        selected = [task.task_id for task in tasks if task.category == TaskCategory.SINGLE_HOP]
    elif preset == "multi_step":
        selected = [task.task_id for task in tasks if task.category == TaskCategory.MULTI_STEP]
    else:
        selected = [task.task_id for task in tasks if task.category == TaskCategory.RECOVERY]

    if explicit_ids:
        unknown_ids = [task_id for task_id in explicit_ids if task_id not in task_map]
        if unknown_ids:
            raise ValueError(f"Unknown task ids: {', '.join(unknown_ids)}")
    return _unique([*selected, *explicit_ids])


def resolve_config_ids(
    configs: list[AgentConfig],
    preset: str | None,
    explicit_ids: list[str] | None = None,
) -> list[str]:
    explicit_ids = explicit_ids or []
    config_map = {config.config_id: config for config in configs}
    if preset and preset not in CONFIG_PRESET_DEFS:
        raise ValueError(f"Unknown config preset: {preset}")

    if preset is None:
        selected = [] if explicit_ids else [config.config_id for config in configs]
    elif preset == "all":
        selected = [config.config_id for config in configs]
    elif preset == "heuristic":
        selected = [config.config_id for config in configs if config.model_provider == "heuristic"]
    elif preset == "dashscope_live":
        selected = [config.config_id for config in configs if config.model_provider == "dashscope"]
    else:
        selected = [config.config_id for config in configs if config.model_provider == "ollama"]

    if explicit_ids:
        unknown_ids = [config_id for config_id in explicit_ids if config_id not in config_map]
        if unknown_ids:
            raise ValueError(f"Unknown config ids: {', '.join(unknown_ids)}")
    return _unique([*selected, *explicit_ids])


def task_presets(tasks: list[TaskSpec]) -> list[dict[str, object]]:
    counts = {
        "all": len(tasks),
        "single_hop": sum(task.category == TaskCategory.SINGLE_HOP for task in tasks),
        "multi_step": sum(task.category == TaskCategory.MULTI_STEP for task in tasks),
        "recovery": sum(task.category == TaskCategory.RECOVERY for task in tasks),
    }
    return [
        {"preset": preset, **meta, "count": counts[preset]}
        for preset, meta in TASK_PRESET_DEFS.items()
    ]


def config_presets(configs: list[AgentConfig]) -> list[dict[str, object]]:
    counts = {
        "all": len(configs),
        "heuristic": sum(config.model_provider == "heuristic" for config in configs),
        "dashscope_live": sum(config.model_provider == "dashscope" for config in configs),
        "ollama_live": sum(config.model_provider == "ollama" for config in configs),
    }
    return [
        {"preset": preset, **meta, "count": counts[preset]}
        for preset, meta in CONFIG_PRESET_DEFS.items()
    ]
