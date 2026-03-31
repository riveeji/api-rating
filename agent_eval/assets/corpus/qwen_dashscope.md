# Qwen DashScope Snapshot
Source: https://www.alibabacloud.com/help/en/model-studio/qwen-api-via-dashscope
Reference-Date: 2026-03-31

## function-calling
DashScope supports function calling through the `tools` parameter. Each tool definition includes a function name, a description, and optionally a JSON Schema parameter object. Providing accurate parameter schemas improves the model's ability to choose and call the right tool.

## result-format
When tool calling is enabled, DashScope expects `result_format` to be set to `message`. In message mode, responses return structured fields such as `choices[].message.content` and `choices[].message.tool_calls`, which is more suitable for multi-turn agent loops than plain text mode.

## tool-choice
The `tool_choice` parameter controls whether the model selects tools automatically, disables tools, or is forced toward a specific function. Supported choices include `auto`, `none`, and an object that names the exact function to call. Forcing a specific tool is useful when the workflow must stay within a known path.

## parallel-tool-calls
DashScope exposes `parallel_tool_calls` as a boolean flag. When enabled, the model can request more than one tool call in a single step. For an evaluation platform, keeping this off by default simplifies trajectory analysis and makes comparison across agent strategies easier.

## structured-output
DashScope also supports structured output with `response_format`. The API can return plain text, a generic JSON object, or JSON that must conform to a provided schema. When using JSON output, the prompt should still clearly instruct the model to answer in JSON.

## validate-arguments
Tool arguments returned by the model should always be validated before execution. Model outputs can still contain malformed fields, missing required keys, or unsafe values. A robust agent runner should treat tool arguments as untrusted input and fail safely when validation does not pass.
