from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import httpx

from agent_eval.config import Settings


class LLMClientError(RuntimeError):
    pass


@dataclass
class LLMResponse:
    message: dict[str, Any]
    usage: dict[str, Any]
    raw: dict[str, Any]


@dataclass
class ParsedToolCall:
    tool_call_id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str


class OpenAICompatibleChatClient:
    def __init__(
        self,
        *,
        settings: Settings,
        base_url: str,
        api_key: str | None,
        provider_name: str,
        inject_result_format: bool = False,
        timeout_seconds: float | None = None,
    ) -> None:
        self.settings = settings
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.provider_name = provider_name
        self.inject_result_format = inject_result_format
        self.timeout_seconds = timeout_seconds or settings.request_timeout_seconds

    def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        temperature: float = 0.0,
        parallel_tool_calls: bool = False,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": model or self.settings.executor_model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"
            payload["parallel_tool_calls"] = parallel_tool_calls
            if self.inject_result_format:
                payload["result_format"] = "message"
        if response_format:
            payload["response_format"] = response_format

        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        raw = response.json()
        try:
            message = raw["choices"][0]["message"]
        except (KeyError, IndexError) as exc:
            raise LLMClientError(f"Unexpected {self.provider_name} response: {raw}") from exc
        usage = raw.get("usage", {})
        return LLMResponse(message=message, usage=usage, raw=raw)

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        temperature: float = 0.0,
        json_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response_format = None
        if json_schema:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "agent_eval_response",
                    "schema": json_schema,
                },
            }
        result = self.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=response_format,
            temperature=temperature,
        )
        return parse_json_content(message_text(result.message))

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


class DashScopeChatClient(OpenAICompatibleChatClient):
    def __init__(self, settings: Settings) -> None:
        if not settings.dashscope_api_key:
            raise LLMClientError("AGENT_EVAL_DASHSCOPE_API_KEY is not configured")
        super().__init__(
            settings=settings,
            base_url=settings.dashscope_base_url,
            api_key=settings.dashscope_api_key,
            provider_name="DashScope",
            inject_result_format=True,
            timeout_seconds=settings.request_timeout_seconds,
        )


class OllamaChatClient(OpenAICompatibleChatClient):
    def __init__(self, settings: Settings) -> None:
        super().__init__(
            settings=settings,
            base_url=settings.ollama_base_url,
            api_key=settings.ollama_api_key,
            provider_name="Ollama",
            inject_result_format=False,
            timeout_seconds=settings.ollama_request_timeout_seconds,
        )


def message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts).strip()
    if content is None:
        return ""
    return str(content).strip()


def parse_tool_calls(message: dict[str, Any]) -> list[ParsedToolCall]:
    calls = []
    for item in message.get("tool_calls", []) or []:
        function = item.get("function", {})
        raw_arguments_value = function.get("arguments", "") or ""
        if isinstance(raw_arguments_value, dict):
            raw_arguments = json.dumps(raw_arguments_value, ensure_ascii=False)
            arguments = raw_arguments_value
        else:
            raw_arguments = str(raw_arguments_value)
            try:
                arguments = json.loads(raw_arguments) if raw_arguments else {}
            except json.JSONDecodeError:
                arguments = {"_raw": raw_arguments}
        calls.append(
            ParsedToolCall(
                tool_call_id=item.get("id", ""),
                name=function.get("name", ""),
                arguments=arguments,
                raw_arguments=raw_arguments,
            )
        )
    return calls


def parse_json_content(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
        if not isinstance(value, dict):
            raise LLMClientError(f"Expected JSON object, got: {value!r}")
        return value
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise LLMClientError(f"Expected JSON output, received: {text}")
        value = json.loads(match.group(0))
        if not isinstance(value, dict):
            raise LLMClientError(f"Expected JSON object, got: {value!r}")
        return value
