"""Model configuration registry — declarative format metadata per model variant.

Maps model identifiers to format metadata (tool schema type, screenshot format,
safety check support, action format, adapter target) so that adding a new model
variant requires one config entry and zero code changes.

Design reference:
  - OpenClaw's model.ts resolveModel pattern (provider catalog with format metadata)
  - CUA's @register_agent decorator (regex-based model matching)

"""

from __future__ import annotations

import asyncio
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, List, Literal, Tuple, TypeVar

import litellm

from ale_run.base_interface.token_usage import usage_scope

# Set LiteLLM global request timeout and max retries from environment variables
litellm.request_timeout = float(os.environ.get("LITELLM_REQUEST_TIMEOUT", "600.0"))
litellm.num_retries = int(os.environ.get("LITELLM_MAX_RETRIES", "1000"))

DEFAULT_RETRY_BUDGET_S = 86400.0  # 24 hours

_T = TypeVar("_T")

# Patterns indicating context window overflow (must NOT be retried as transient
# infra errors so that reactive context compaction can run immediately).
_CONTEXT_OVERFLOW_PATTERNS: tuple[str, ...] = (
    "context_length_exceeded",
    "context length",
    "maximum context length",
    "prompt is too long",
    "input is too long",
    "request_too_large",
    "exceeds the model's maximum context",
    "exceeds the maximum number of tokens",
    "context window",
    "too many tokens",
    "payload too large",
    "request entity too large",
    "exceeds maximum allowed limit",
    "status code: 413",
    "http 413",
)

# Substrings in exception messages or types that indicate a transient
# infrastructure, overload, preemption, timeout, or network failure.
_TRANSIENT_ERROR_PATTERNS: tuple[str, ...] = (
    "timeout",
    "timed out",
    "sockettimeout",
    "rate limit",
    "ratelimit",
    "too many requests",
    "429",
    "500",
    "502",
    "503",
    "504",
    "520",
    "522",
    "524",
    "529",
    "overloaded",
    "preempted",
    "resource_exhausted",
    "service unavailable",
    "unavailable",
    "bad gateway",
    "gateway timeout",
    "internal server error",
    "internal error",
    "backend error",
    "connection",
    "disconnected",
    "reset by peer",
    "broken pipe",
    "unexpected eof",
    "eof occurred",
    "server closed",
    "remoteprotocolerror",
    "clientoserror",
    "clientpayloaderror",
    "no healthy",
    "try again",
)

_TRANSIENT_STATUS_CODES: frozenset[int] = frozenset(
    {408, 429, 500, 502, 503, 504, 520, 522, 524, 529}
)


def get_retry_budget_s() -> float:
    """Return the time budget in seconds for retrying transient LLM failures.

    Defaults to 86400.0s (24 hours) in normal execution, or 0.0s inside pytest
    unless ``LITELLM_RETRY_BUDGET_S`` is explicitly set in the environment.
    """
    env_val = os.environ.get("LITELLM_RETRY_BUDGET_S")
    if env_val is not None:
        return float(env_val)
    if "PYTEST_CURRENT_TEST" in os.environ:
        return 0.0
    return DEFAULT_RETRY_BUDGET_S


def is_transient_llm_error(exc: BaseException) -> bool:
    """Return True if *exc* is a transient infra/network/timeout/overload error."""
    if isinstance(exc, (KeyboardInterrupt, SystemExit, asyncio.CancelledError)):
        return False

    msg_lower = f"{type(exc).__name__}: {exc}".lower()
    if any(p in msg_lower for p in _CONTEXT_OVERFLOW_PATTERNS):
        return False

    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        if status_code in _TRANSIENT_STATUS_CODES:
            return True
        if 400 <= status_code < 500 and not any(
            p in msg_lower for p in ("overloaded", "preempted", "resource_exhausted", "rate limit", "timeout", "timed out")
        ):
            return False

    if isinstance(
        exc,
        (
            TimeoutError,
            asyncio.TimeoutError,
            ConnectionError,
            OSError,
            litellm.Timeout,
            litellm.RateLimitError,
            litellm.InternalServerError,
            litellm.ServiceUnavailableError,
            litellm.APIConnectionError,
        ),
    ):
        return True

    bad_gw = getattr(litellm, "BadGatewayError", None)
    gw_timeout = getattr(litellm, "GatewayTimeoutError", None)
    extra_types = tuple(t for t in (bad_gw, gw_timeout) if isinstance(t, type))
    if extra_types and isinstance(exc, extra_types):
        return True

    if any(p in msg_lower for p in _TRANSIENT_ERROR_PATTERNS):
        return True

    cause = exc.__cause__ or exc.__context__
    if cause is not None and cause is not exc:
        return is_transient_llm_error(cause)

    return False


async def call_with_retry_budget(
    coro_fn: Callable[[], Awaitable[_T]],
    *,
    label: str = "LLM call",
    budget_s: float | None = None,
) -> _T:
    """Execute *coro_fn* with exponential backoff for up to *budget_s* seconds on transient errors."""
    effective_budget_s = get_retry_budget_s() if budget_s is None else budget_s
    deadline = time.monotonic() + max(0.0, effective_budget_s)
    attempt = 0

    while True:
        attempt += 1
        try:
            with usage_scope(call_site=label):
                return await coro_fn()
        except BaseException as exc:
            if getattr(exc, "_retry_budget_exhausted", False):
                raise
            if not is_transient_llm_error(exc):
                raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                try:
                    setattr(exc, "_retry_budget_exhausted", True)
                except Exception:
                    pass
                raise
            base_delay = min(60.0, 5.0 * (2 ** min(attempt - 1, 4)))
            jitter = random.uniform(0.0, min(5.0, base_delay * 0.25))
            sleep_s = min(remaining, base_delay + jitter)
            print(
                f"[RetryBudget] {label} failed on attempt {attempt} "
                f"({type(exc).__name__}: {exc}). "
                f"Retrying in {sleep_s:.1f}s (remaining budget: {remaining:.0f}s)...",
                flush=True,
            )
            await asyncio.sleep(sleep_s)


@dataclass(frozen=True)
class ModelConfig:
    """Declarative format metadata for a model variant.

    Fields:
        tool_schema_type: OpenAI tool schema type sent to the API.
            "computer" (GPT 5.4) or "computer_use_preview" (legacy).
        screenshot_output_type: Image type in computer_call_output items.
            "computer_screenshot" (GPT 5.4, with detail="original") or
            "input_image" (computer-use-preview and Anthropic).
        supports_safety_checks: Whether to include acknowledged_safety_checks
            in computer_call_output. False for GPT 5.4.
        action_format: "batched" (GPT 5.4 actions array) or "single"
            (computer-use-preview singular action).
        adapter_target: Provider format for sanitize_items() conversion.
            "openai-responses" or "anthropic".
    """

    tool_schema_type: str
    screenshot_output_type: str
    supports_safety_checks: bool
    action_format: str
    adapter_target: str
    provider: str | None = None
    model_api: str | None = None
    transcript_api_label: str | None = None
    helper_transport_defaults: "HelperTransportDefaults | None" = None
    context_window: int | None = None


@dataclass(frozen=True)
class HelperTransportDefaults:
    """Default helper transport modes for a resolved model."""

    memory_flush: Literal["responses", "chat"] = "chat"
    compaction: Literal["responses", "chat"] = "chat"
    vision: Literal["responses", "chat"] = "chat"

    def for_purpose(
        self,
        purpose: Literal["memory_flush", "compaction", "vision"],
    ) -> Literal["responses", "chat"]:
        if purpose == "memory_flush":
            return self.memory_flush
        if purpose == "vision":
            return self.vision
        return self.compaction


@dataclass(frozen=True)
class ResolvedModel:
    """Capability-aware resolved model metadata for one runtime model string."""

    model: str
    model_id: str
    provider: str
    model_api: str
    adapter_target: str
    tool_schema_type: str
    screenshot_output_type: str
    supports_safety_checks: bool
    action_format: str
    transcript_api_label: str
    helper_transport_defaults: HelperTransportDefaults
    context_window: int | None = None


def model_credentials(
    primary_model: str, target_model: str, api_key: str | None, api_base: str | None
) -> dict[str, str]:
    """Use explicit gateway credentials only for models on the same provider."""
    def provider(model: str) -> str:
        return model.split("/", 1)[0] if "/" in model else resolve_model(model).provider

    primary_provider, target_provider = provider(primary_model), provider(target_model)
    if primary_provider != target_provider or (
        primary_provider == "unknown" and primary_model != target_model
    ):
        return {}
    return {key: value for key, value in (("api_key", api_key), ("api_base", api_base)) if value is not None}


def model_generation_kwargs(
    primary_model: str, target_model: str, kwargs: dict[str, Any] | None
) -> dict[str, Any]:
    """Select generation options applicable to an auxiliary model call."""
    primary_provider = primary_model.split("/", 1)[0] if "/" in primary_model else resolve_model(primary_model).provider
    target_provider = target_model.split("/", 1)[0] if "/" in target_model else resolve_model(target_model).provider
    same_provider = primary_provider == target_provider and (
        primary_provider != "unknown" or primary_model == target_model
    )
    generic = {"temperature", "top_p", "seed", "presence_penalty", "frequency_penalty"}
    reserved = {"model", "messages", "input", "tools", "stream", "api_key", "api_base", "timeout", "max_tokens", "max_turns", "truncation"}
    return {
        key: value for key, value in (kwargs or {}).items()
        if key not in reserved and (same_provider or key in generic)
    }


_THINKING_OPTIONS = frozenset({
    "reasoning", "reasoning_effort", "thinking", "thinking_level", "include_reasoning",
})


def merge_helper_generation_kwargs(
    generation_kwargs: dict[str, Any] | None,
    thinking_params: dict[str, Any] | None,
    *,
    override_thinking: bool = False,
) -> dict[str, Any]:
    """Merge inherited options with an optional helper-specific thinking level.

    Main-loop extra kwargs override its default thinking. Helpers inherit that
    effective setting unless their own thinking level is explicitly configured.
    In particular, an explicit ``off`` must remove inherited thinking options
    even though its resolved parameter dictionary is empty.
    """
    options = dict(generation_kwargs or {})
    extra_body = options.get("extra_body")
    nested_thinking = isinstance(extra_body, dict) and any(k in extra_body for k in _THINKING_OPTIONS)
    if override_thinking:
        options = {k: v for k, v in options.items() if k not in _THINKING_OPTIONS}
        if nested_thinking:
            options["extra_body"] = {k: v for k, v in extra_body.items() if k not in _THINKING_OPTIONS}
        return {**options, **(thinking_params or {})}
    if nested_thinking or any(k in options for k in _THINKING_OPTIONS):
        return options
    return {**(thinking_params or {}), **options}


# ---------------------------------------------------------------------------
# Registry: ordered list of (compiled_regex, ModelConfig) — first match wins.
# ---------------------------------------------------------------------------

_MODEL_CONFIGS: List[Tuple[re.Pattern, ModelConfig]] = [
    (
        re.compile(r"gpt-5\.4", re.IGNORECASE),
        ModelConfig(
            tool_schema_type="computer",
            screenshot_output_type="computer_screenshot",
            supports_safety_checks=False,
            action_format="batched",
            adapter_target="openai-responses",
        ),
    ),
    (
        re.compile(r"computer-use-preview", re.IGNORECASE),
        ModelConfig(
            tool_schema_type="computer_use_preview",
            screenshot_output_type="input_image",
            supports_safety_checks=True,
            action_format="single",
            adapter_target="openai-responses",
        ),
    ),
]

# Default config for models that don't match any pattern (Anthropic, etc.)
_DEFAULT_CONFIG = ModelConfig(
    tool_schema_type="computer_use_preview",
    screenshot_output_type="input_image",
    supports_safety_checks=True,
    action_format="single",
    adapter_target="anthropic",
)


def get_model_config(model: str) -> ModelConfig:
    """Look up model config by matching model string against registry patterns.

    Searches ``_MODEL_CONFIGS`` in order; returns the first match.  Falls back
    to ``_DEFAULT_CONFIG`` (Anthropic-compatible) if no pattern matches.

    Args:
        model: litellm model identifier (e.g. "openai/gpt-5.4", "anthropic/claude-sonnet-4-20250514").
    """
    for pattern, config in _MODEL_CONFIGS:
        if pattern.search(model):
            return config
    return _DEFAULT_CONFIG


def resolve_model(model: str | ResolvedModel) -> ResolvedModel:
    """Resolve a model string into structured runtime metadata."""
    if isinstance(model, ResolvedModel):
        return model

    config = get_model_config(model)
    provider = config.provider or _infer_provider(model)
    model_api = config.model_api or _infer_model_api(config, provider)
    helper_transport_defaults = (
        config.helper_transport_defaults
        or _default_helper_transports(provider, model)
    )
    transcript_api_label = (
        config.transcript_api_label
        or _default_transcript_api_label(provider, model_api)
    )

    return ResolvedModel(
        model=model,
        model_id=model.split("/", 1)[-1] if "/" in model else model,
        provider=provider,
        model_api=model_api,
        adapter_target=config.adapter_target,
        tool_schema_type=config.tool_schema_type,
        screenshot_output_type=config.screenshot_output_type,
        supports_safety_checks=config.supports_safety_checks,
        action_format=config.action_format,
        transcript_api_label=transcript_api_label,
        helper_transport_defaults=helper_transport_defaults,
        context_window=config.context_window or _lookup_context_window(model),
    )


def register_model_config(pattern: str, config: ModelConfig) -> None:
    """Register a new model config at the front of the registry.

    New entries take priority over existing ones (prepended to list).
    Useful for adding model support at runtime or in tests.

    Args:
        pattern: Regex pattern to match model strings.
        config: ModelConfig for matching models.
    """
    _MODEL_CONFIGS.insert(0, (re.compile(pattern, re.IGNORECASE), config))


def _infer_provider(model: str) -> str:
    model_lower = model.lower()
    if model_lower.startswith("anthropic/") or "claude" in model_lower:
        return "anthropic"
    if (
        "openai" in model_lower
        or "gpt" in model_lower
        or model_lower.startswith(("o1", "o3", "o4"))
    ):
        return "openai"
    if "gemini" in model_lower or "google" in model_lower:
        return "google"
    if "vertex" in model_lower:
        return "vertex"
    return "unknown"


def _infer_model_api(config: ModelConfig, provider: str) -> str:
    if config.adapter_target == "openai-responses" or provider == "openai":
        return "responses"
    return "chat"


def _default_helper_transports(
    provider: str, model: str = ""
) -> HelperTransportDefaults:
    # OpenRouter routes everything through Chat Completions — always use "chat".
    if model.lower().startswith("openrouter/"):
        return HelperTransportDefaults()
    if provider == "openai":
        return HelperTransportDefaults(memory_flush="responses")
    return HelperTransportDefaults()


def _default_transcript_api_label(provider: str, model_api: str) -> str:
    if provider == "openai" and model_api == "responses":
        return "openai-responses"
    if provider in {"anthropic", "google", "vertex"}:
        return provider
    return provider if provider != "unknown" else model_api


def _lookup_context_window(model: str) -> int | None:
    env_override = os.environ.get("CONTEXT_WINDOW_OVERRIDE")
    if env_override:
        try:
            val = int(env_override)
            if val > 0:
                return val
        except ValueError:
            pass
    for candidate in _model_candidates(model):
        try:
            import litellm

            info = litellm.get_model_info(candidate)
            max_input = info.get("max_input_tokens")
            if max_input and max_input > 0:
                return int(max_input)
        except Exception:
            continue
    if "gemini" in (model or "").lower():
        return 1_000_000
    return None


def _model_candidates(model: str) -> list[str]:
    candidates = [model]
    if "/" in model:
        candidates.append(model.split("/", 1)[1])
    return candidates
