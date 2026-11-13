"""Shared helper-call adapter backed by resolved runtime metadata."""

from __future__ import annotations

import json as _json
import os
from dataclasses import dataclass
from typing import Any, Literal

from .model_config import ResolvedModel, call_with_retry_budget, resolve_model
from .usage import record_helper_usage


@dataclass(frozen=True)
class HelperCallResult:
    text: str
    tool_calls: list[dict[str, Any]]


async def call_helper_model(
    model: str | ResolvedModel,
    *,
    purpose: Literal["memory_flush", "compaction", "vision"],
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    max_tokens: int,
    temperature: float,
    timeout: int | None = None,
    thinking_params: dict[str, Any] | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
) -> HelperCallResult:
    """Call a helper model using the resolved transport defaults for the purpose."""
    import litellm

    resolved = resolve_model(model)
    transport = resolved.helper_transport_defaults.for_purpose(purpose)
    connection_kwargs: dict[str, Any] = {}
    if api_key is not None:
        connection_kwargs["api_key"] = api_key
    if api_base is not None:
        connection_kwargs["api_base"] = api_base

    if transport == "responses":
        kwargs: dict[str, Any] = {
            "model": resolved.model,
            "input": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **(thinking_params or {}),
            **connection_kwargs,
        }
        kwargs["timeout"] = (
            timeout
            if timeout is not None
            else float(os.environ.get("LITELLM_REQUEST_TIMEOUT", "1800.0"))
        )
        kwargs["num_retries"] = int(
            os.environ.get("LITELLM_MAX_RETRIES", "2")
        )
        if tools:
            kwargs["tools"] = [_to_responses_function_tool(tool) for tool in tools]
        response = await call_with_retry_budget(
            lambda: litellm.aresponses(**kwargs),
            label=f"helper:{purpose}({resolved.model})",
        )
        record_helper_usage(response, model=resolved.model, purpose=purpose)
        payload = response.model_dump()
        return HelperCallResult(
            text=_extract_responses_text(payload.get("output", [])),
            tool_calls=_extract_responses_tool_calls(payload.get("output", [])),
        )

    kwargs = {
        "model": resolved.model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        **(thinking_params or {}),
        **connection_kwargs,
    }
    kwargs["timeout"] = (
        timeout
        if timeout is not None
        else float(os.environ.get("LITELLM_REQUEST_TIMEOUT", "1800.0"))
    )
    kwargs["num_retries"] = int(os.environ.get("LITELLM_MAX_RETRIES", "2"))
    if tools:
        kwargs["tools"] = tools
    response = await call_with_retry_budget(
        lambda: litellm.acompletion(**kwargs),
        label=f"helper:{purpose}({resolved.model})",
    )
    record_helper_usage(response, model=resolved.model, purpose=purpose)
    choices = getattr(response, "choices", None) or []
    if not choices:
        return HelperCallResult(text="", tool_calls=[])
    choice = choices[0]
    msg = getattr(choice, "message", None)
    if msg is None:
        return HelperCallResult(text="", tool_calls=[])
    text_val = (getattr(msg, "content", None) or "").strip()
    if not text_val and purpose == "compaction":
        text_val = (getattr(msg, "reasoning_content", None) or "").strip()
    return HelperCallResult(
        text=text_val,
        tool_calls=[
            {
                "name": tool_call.function.name,
                "arguments": tool_call.function.arguments,
            }
            for tool_call in (getattr(msg, "tool_calls", None) or [])
        ],
    )


def _to_responses_function_tool(tool: dict[str, Any]) -> dict[str, Any]:
    function = tool.get("function", {})
    return {"type": "function", **function}


def _extract_responses_text(output_items: Any) -> str:
    if not isinstance(output_items, list):
        return ""

    parts: list[str] = []
    for item in output_items:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content", [])
        if isinstance(content, str):
            if content:
                parts.append(content)
            continue
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") in {"output_text", "text"} and block.get("text"):
                parts.append(block["text"])
    return "\n".join(parts).strip()


def _extract_responses_tool_calls(output_items: Any) -> list[dict[str, Any]]:
    if not isinstance(output_items, list):
        return []

    tool_calls: list[dict[str, Any]] = []
    for item in output_items:
        if not isinstance(item, dict) or item.get("type") != "function_call":
            continue
        tool_calls.append(
            {
                "name": item.get("name", ""),
                "arguments": item.get("arguments", _json.dumps({})),
            }
        )
    return tool_calls
