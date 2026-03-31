from __future__ import annotations

import ast
import operator
import time
from dataclasses import dataclass
from typing import Any

import httpx

from agent_eval.config import Settings
from agent_eval.models import FaultInjection, FaultMode, ToolCallRecord, ToolCallStatus
from agent_eval.storage import Storage


class ToolExecutionError(RuntimeError):
    pass


@dataclass
class ToolContext:
    storage: Storage
    settings: Settings
    faults: list[FaultInjection]


class FaultController:
    def __init__(self, faults: list[FaultInjection]) -> None:
        self._faults = list(faults)
        self._consumed: set[tuple[str, FaultMode]] = set()

    def maybe_raise(self, tool_name: str) -> str | None:
        for fault in self._faults:
            key = (fault.tool_name, fault.mode)
            if fault.tool_name != tool_name or key in self._consumed:
                continue
            self._consumed.add(key)
            if fault.mode == FaultMode.ERROR_ONCE:
                raise ToolExecutionError(fault.message or f"{tool_name} transient error")
            if fault.mode == FaultMode.TIMEOUT_ONCE:
                raise TimeoutError(fault.message or f"{tool_name} timeout")
            if fault.mode == FaultMode.EMPTY_ONCE:
                return "empty"
        return None

    @property
    def consumed(self) -> int:
        return len(self._consumed)


class ToolRegistry:
    def __init__(self, storage: Storage, settings: Settings) -> None:
        self.storage = storage
        self.settings = settings

    def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        fault_controller: FaultController,
    ) -> ToolCallRecord:
        started = time.perf_counter()
        try:
            maybe_empty = fault_controller.maybe_raise(tool_name)
            if maybe_empty == "empty":
                return ToolCallRecord(
                    tool_name=tool_name,
                    arguments=arguments,
                    status=ToolCallStatus.EMPTY,
                    output=[],
                    latency_ms=(time.perf_counter() - started) * 1000,
                )
            handler = getattr(self, f"_tool_{tool_name}", None)
            if handler is None:
                raise ToolExecutionError(f"Unknown tool: {tool_name}")
            output = handler(arguments)
            status = ToolCallStatus.EMPTY if output in (None, [], {}) else ToolCallStatus.SUCCESS
            return ToolCallRecord(
                tool_name=tool_name,
                arguments=arguments,
                status=status,
                output=output,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        except Exception as exc:
            return ToolCallRecord(
                tool_name=tool_name,
                arguments=arguments,
                status=ToolCallStatus.ERROR,
                error=str(exc),
                latency_ms=(time.perf_counter() - started) * 1000,
            )

    def list_tools(self) -> list[str]:
        return [
            "doc_search",
            "doc_read",
            "sql_query",
            "case_api",
            "calculator",
            "web_lookup",
        ]

    def tool_schemas(self, enable_web_lookup: bool = False) -> list[dict[str, Any]]:
        specs = [
            {
                "type": "function",
                "function": {
                    "name": "doc_search",
                    "description": "搜索本地知识库文档切块，返回候选 chunk_id、标题、摘要和分数。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "检索查询"},
                            "limit": {"type": "integer", "description": "返回条数", "default": 5},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "doc_read",
                    "description": "读取指定 chunk_id 的完整文档片段内容。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "chunk_id": {"type": "string", "description": "文档切块唯一标识"},
                        },
                        "required": ["chunk_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "sql_query",
                    "description": "执行只读 SQLite 查询。仅允许 SELECT / WITH / PRAGMA。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sql": {"type": "string", "description": "只读 SQL 语句"},
                        },
                        "required": ["sql"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "case_api",
                    "description": "读取模拟企业 Case API，返回 case_id 对应的结构化元数据。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "case_id": {"type": "string", "description": "案例编号，例如 CASE-001"},
                        },
                        "required": ["case_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "执行基础算术表达式计算，只支持加减乘除。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string", "description": "算术表达式，例如 1200 / 3"},
                        },
                        "required": ["expression"],
                    },
                },
            },
        ]
        if enable_web_lookup:
            specs.append(
                {
                    "type": "function",
                    "function": {
                        "name": "web_lookup",
                        "description": "联网查询公开网页摘要。该工具结果不稳定，只能作为辅助信息。",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "联网搜索查询"},
                            },
                            "required": ["query"],
                        },
                    },
                }
            )
        return specs

    def _tool_doc_search(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        query = str(arguments.get("query", "")).strip()
        limit = int(arguments.get("limit", 5))
        if not query:
            raise ToolExecutionError("doc_search requires a non-empty query")
        return self.storage.search_chunks(query, limit=limit)

    def _tool_doc_read(self, arguments: dict[str, Any]) -> dict[str, Any]:
        chunk_id = str(arguments.get("chunk_id", "")).strip()
        if not chunk_id:
            raise ToolExecutionError("doc_read requires chunk_id")
        return self.storage.get_chunk(chunk_id)

    def _tool_sql_query(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        sql = str(arguments.get("sql", "")).strip()
        if not sql:
            raise ToolExecutionError("sql_query requires sql")
        return self.storage.run_readonly_query(sql)

    def _tool_case_api(self, arguments: dict[str, Any]) -> dict[str, Any]:
        case_id = str(arguments.get("case_id", "")).strip()
        if not case_id:
            raise ToolExecutionError("case_api requires case_id")
        return self.storage.get_case(case_id)

    def _tool_calculator(self, arguments: dict[str, Any]) -> dict[str, Any]:
        expression = str(arguments.get("expression", "")).strip()
        if not expression:
            raise ToolExecutionError("calculator requires expression")
        value = _safe_eval(expression)
        return {"expression": expression, "value": round(value, 2)}

    def _tool_web_lookup(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = str(arguments.get("query", "")).strip()
        if not query:
            raise ToolExecutionError("web_lookup requires query")
        response = httpx.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"},
            timeout=self.settings.request_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        related_topics = payload.get("RelatedTopics", [])
        top_result = related_topics[0] if related_topics else {}
        return {
            "query": query,
            "heading": payload.get("Heading"),
            "abstract": payload.get("AbstractText"),
            "url": payload.get("AbstractURL") or top_result.get("FirstURL"),
        }


_ALLOWED_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_ALLOWED_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval(expression: str) -> float:
    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BIN_OPS:
            return _ALLOWED_BIN_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY_OPS:
            return _ALLOWED_UNARY_OPS[type(node.op)](_eval(node.operand))
        raise ToolExecutionError("calculator only supports basic arithmetic")

    tree = ast.parse(expression, mode="eval")
    return _eval(tree)
