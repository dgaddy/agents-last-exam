"""LiteLLM callback that records every model call into the token-usage ledger.

Registered once per harness process. Every ``litellm.acompletion`` /
``litellm.aresponses`` call in the process (main loop, helpers, subagents,
tool-internal VLM calls, CUA loops) reaches this logger, which appends one
record per call to ``<work_dir>/token_usage.jsonl``.

Attribution comes from :mod:`ale_run.base_interface.token_usage` scopes:
``call_with_retry_budget`` labels each call site and subagents set
``role="subagent"``. The scope is captured synchronously in
``log_pre_api_call`` (caller context) so it does not depend on how LiteLLM
schedules its async success/failure handlers.
"""

from __future__ import annotations

import datetime as _dt
import logging
import threading
from typing import Any

from litellm.integrations.custom_logger import CustomLogger

from ale_run.base_interface.token_usage import (
    TokenUsageLedger,
    build_record,
    current_scope,
    normalize_usage,
)

logger = logging.getLogger(__name__)

_MAX_PENDING = 4096


def role_for_call_site(call_site: str | None) -> str | None:
    """Map a ``call_with_retry_budget`` label to a ledger role."""
    if not call_site:
        return None
    label = call_site.lower()
    if label.startswith(("helper:compaction", "helper:memory_flush")):
        return "compaction"
    if label.startswith(("helper:vision", "analyze_image", "predict_click")):
        return "tool_llm"
    if label.startswith("subagent"):
        return "subagent"
    if "predict_step" in label:
        return "agent"
    return "other"


def resolve_role(scope_role: str | None, call_site: str | None) -> str:
    site_role = role_for_call_site(call_site)
    if site_role in (None, "agent"):
        return scope_role or "agent"
    return site_role


def _iso(value: Any) -> str | None:
    if isinstance(value, _dt.datetime):
        if value.tzinfo is None:
            value = value.astimezone()
        return value.astimezone(_dt.UTC).isoformat(timespec="milliseconds")
    return None


def _latency_ms(start: Any, end: Any) -> float | None:
    if isinstance(start, _dt.datetime) and isinstance(end, _dt.datetime):
        return round((end - start).total_seconds() * 1000.0, 1)
    return None


def _get(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


class TokenUsageCallback(CustomLogger):
    """Append one ledger record per LiteLLM call (success or failure)."""

    def __init__(self, ledger: TokenUsageLedger, *, agent_harness: str = "ale_claw"):
        super().__init__()
        self._ledger = ledger
        self._agent_harness = agent_harness
        self._lock = threading.Lock()
        self._pending: dict[str, tuple[str | None, str | None]] = {}
        self._agent_calls = 0

    def retarget(self, ledger: TokenUsageLedger) -> None:
        """Start writing to ``ledger`` with fresh per-attempt counters."""
        with self._lock:
            self._ledger = ledger
            self._pending.clear()
            self._agent_calls = 0

    def log_pre_api_call(self, model, messages, kwargs):
        try:
            call_id = kwargs.get("litellm_call_id")
            if not call_id:
                return
            with self._lock:
                if len(self._pending) >= _MAX_PENDING:
                    self._pending.pop(next(iter(self._pending)))
                self._pending[call_id] = current_scope()
        except Exception as exc:                                # noqa: BLE001
            logger.debug("token_usage pre-call hook failed: %s", exc)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._record(kwargs, response_obj, start_time, end_time, ok=True)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._record(kwargs, response_obj, start_time, end_time, ok=False)

    def _record(self, kwargs, response_obj, start_time, end_time, *, ok: bool) -> None:
        try:
            kwargs = kwargs or {}
            call_id = kwargs.get("litellm_call_id")
            with self._lock:
                scope = (
                    self._pending.pop(call_id, None) if ok else self._pending.get(call_id)
                ) if call_id else None
            scope_role, call_site = scope if scope is not None else current_scope()
            role = resolve_role(scope_role, call_site)
            with self._lock:
                if role == "agent" and ok:
                    self._agent_calls += 1
                step = self._agent_calls if ok or role != "agent" else self._agent_calls + 1

            litellm_params = kwargs.get("litellm_params") or {}
            provider = kwargs.get("custom_llm_provider") or litellm_params.get(
                "custom_llm_provider"
            )
            raw_usage = _get(response_obj, "usage") if ok else None
            http_status = 200 if ok else None
            if not ok:
                exc = kwargs.get("exception")
                status_code = getattr(exc, "status_code", None)
                http_status = status_code if isinstance(status_code, int) else None
            cost = kwargs.get("response_cost")
            self._ledger.append(build_record(
                role=role,
                model=kwargs.get("model"),
                usage=normalize_usage(raw_usage, provider=provider) if ok else None,
                raw_usage=raw_usage,
                status="ok" if ok else "error",
                provider=provider,
                api=kwargs.get("call_type"),
                response_model=_get(response_obj, "model") if ok else None,
                http_status=http_status,
                response_id=_get(response_obj, "id") if ok else None,
                started_at=_iso(start_time),
                latency_ms=_latency_ms(start_time, end_time),
                cost_usd=float(cost) if isinstance(cost, (int, float)) else None,
                step=step,
                call_site=call_site,
                agent_harness=self._agent_harness,
            ))
        except Exception as exc:                                # noqa: BLE001
            logger.debug("token_usage callback failed: %s", exc)


_INSTALL_LOCK = threading.Lock()


def install_token_usage_callback(ledger_path, *, agent_harness: str = "ale_claw") -> None:
    """Register a :class:`TokenUsageCallback` writing to ``ledger_path``.

    One callback per process; a later call retargets it to the new ledger.
    Never raises; a failure only disables usage recording.
    """
    try:
        import litellm

        with _INSTALL_LOCK:
            ledger = TokenUsageLedger(ledger_path)
            for existing in litellm.callbacks or []:
                if isinstance(existing, TokenUsageCallback):
                    existing.retarget(ledger)
                    return
            callback = TokenUsageCallback(ledger, agent_harness=agent_harness)
            litellm.callbacks = [*(litellm.callbacks or []), callback]
    except Exception as exc:                                    # noqa: BLE001
        logger.warning("token_usage: could not register LiteLLM callback: %s", exc)
