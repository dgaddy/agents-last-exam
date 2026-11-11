"""Per-model-call token usage ledger (``token_usage.jsonl``).

Each line is one JSON object describing one model API call, schema
``gceval_token_usage/v1``. Records carry token counts and call metadata only:
never prompts, responses, images, or credentials.

Two producers feed the ledger:

* In-process harnesses (e.g. ``ale_claw``) append a record per call to
  ``<work_dir>/token_usage.jsonl`` as calls complete (:class:`TokenUsageLedger`).
* CLI harnesses that run inside the task VM expose
  ``BaseAgentDeployer.token_usage_records``, which rebuilds records from the
  CLI's native event stream on the host.

After the agent phase, the lifecycle merges both into the canonical
``<attempt_dir>/token_usage.jsonl`` next to ``run.json``
(:func:`finalize_attempt_ledger`), stamping attempt identity and call order.

The ``usage`` object follows a 3-way input partition:
``prompt_tokens == cached_tokens + cache_write_tokens + uncached_prompt_tokens``;
``completion_tokens`` is all billed output including reasoning;
``reasoning_tokens`` / ``response_tokens`` are null when the provider does not
split them.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import datetime as _dt
import json
import logging
import os
import re
import sys
import threading
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA = "gceval_token_usage/v1"
LEDGER_FILENAME = "token_usage.jsonl"

ROLES = frozenset({
    "agent", "subagent", "compaction", "tool_llm",
    "user_simulator", "judge", "evaluator", "other",
})
NON_AGENT_ROLES = frozenset({"user_simulator", "judge", "evaluator"})

USAGE_FIELDS = (
    "prompt_tokens",
    "cached_tokens",
    "cache_write_tokens",
    "uncached_prompt_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "response_tokens",
)

_CAMEL_RE = re.compile(r"(?<!^)(?=[A-Z])")

# ---------------------------------------------------------------------------
# Attribution scope (role / call site) carried across awaits via contextvars.
# ---------------------------------------------------------------------------

_ROLE: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "ale_token_usage_role", default=None,
)
_CALL_SITE: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "ale_token_usage_call_site", default=None,
)


@contextlib.contextmanager
def usage_scope(*, role: str | None = None, call_site: str | None = None) -> Iterator[None]:
    """Attribute model calls made inside the block to ``role`` / ``call_site``.

    Unset arguments inherit the enclosing scope. Safe to nest and to use
    around ``await``; child tasks created inside the block inherit it.
    """
    tokens: list[tuple[contextvars.ContextVar, contextvars.Token]] = []
    if role is not None:
        tokens.append((_ROLE, _ROLE.set(role)))
    if call_site is not None:
        tokens.append((_CALL_SITE, _CALL_SITE.set(call_site)))
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


def current_scope() -> tuple[str | None, str | None]:
    """Return ``(role, call_site)`` for the current context."""
    return _ROLE.get(), _CALL_SITE.get()


# ---------------------------------------------------------------------------
# Usage normalization
# ---------------------------------------------------------------------------


def to_jsonable(value: Any) -> Any:
    """Convert pydantic objects / mappings / sequences into plain JSON data."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "model_dump"):
        try:
            return to_jsonable(value.model_dump())
        except Exception as exc:                                # noqa: BLE001
            logger.debug("token_usage: model_dump failed: %s", exc)
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "__dict__"):
        return {
            str(k): to_jsonable(v)
            for k, v in vars(value).items()
            if not str(k).startswith("_")
        }
    return str(value)


def _snake_keys(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            _CAMEL_RE.sub("_", str(k)).lower(): _snake_keys(v) for k, v in value.items()
        }
    return value


def _int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _sub(metrics: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = metrics.get(key)
    return value if isinstance(value, Mapping) else {}


def _finish(
    *,
    prompt: int,
    cached: int,
    cache_write: int,
    completion: int,
    reasoning: int | None,
) -> dict[str, int | None]:
    prompt = max(0, prompt)
    cache_write = min(max(0, cache_write), prompt)
    # Providers occasionally report more cached tokens than were sent.
    cached = min(max(0, cached), prompt - cache_write)
    completion = max(0, completion)
    if reasoning is not None:
        reasoning = min(max(0, reasoning), completion)
        response: int | None = completion - reasoning
    else:
        response = None
    return {
        "prompt_tokens": prompt,
        "cached_tokens": cached,
        "cache_write_tokens": cache_write,
        "uncached_prompt_tokens": prompt - cached - cache_write,
        "completion_tokens": completion,
        "reasoning_tokens": reasoning,
        "response_tokens": response,
    }


def _is_xai_chat(metrics: Mapping[str, Any], reasoning: int | None) -> bool:
    if not reasoning:
        return False
    total = _int(metrics.get("total_tokens"))
    if total is None:
        return False
    prompt = _int(metrics.get("prompt_tokens")) or 0
    completion = _int(metrics.get("completion_tokens")) or 0
    return total == prompt + completion + reasoning


def normalize_usage(raw: Any, *, provider: str | None = None) -> dict[str, int | None] | None:
    """Normalize a provider / LiteLLM usage object into the ledger ``usage`` shape.

    Recognized shapes, in order:

    * Chat Completions (OpenAI, LiteLLM-normalized for every provider):
      ``prompt_tokens`` includes cache reads and cache writes;
      ``prompt_tokens_details.cached_tokens`` / ``cache_creation_tokens``
      split them. xAI Chat Completions exclude reasoning from
      ``completion_tokens`` (detected via ``provider == "xai"`` or the
      ``total == prompt + completion + reasoning`` invariant) so it is added
      back. ``prompt_tokens_details.text_tokens`` is never read (LiteLLM can
      make it negative).
    * Gemini ``usageMetadata`` (``promptTokenCount`` includes cache hits;
      output = candidates + thoughts).
    * Anthropic Messages: ``input_tokens``, ``cache_read_input_tokens`` and
      ``cache_creation_input_tokens`` are disjoint and summed.
    * Responses API: ``input_tokens`` includes cache reads,
      ``output_tokens`` includes reasoning.

    Returns None when no recognized fields are present.
    """
    if raw is None:
        return None
    metrics = _snake_keys(to_jsonable(raw))
    if not isinstance(metrics, Mapping):
        return None
    provider = (provider or "").lower()

    if "prompt_tokens" in metrics:
        details = _sub(metrics, "prompt_tokens_details")
        cached = _int(details.get("cached_tokens"))
        if cached is None:
            cached = _int(metrics.get("cache_read_input_tokens")) or 0
        cache_write = _int(details.get("cache_creation_tokens"))
        if cache_write is None:
            cache_write = _int(metrics.get("cache_creation_input_tokens")) or 0
        completion = _int(metrics.get("completion_tokens")) or 0
        reasoning = _int(_sub(metrics, "completion_tokens_details").get("reasoning_tokens"))
        if reasoning is not None and (provider == "xai" or _is_xai_chat(metrics, reasoning)):
            completion += reasoning
        return _finish(
            prompt=_int(metrics.get("prompt_tokens")) or 0,
            cached=cached,
            cache_write=cache_write,
            completion=completion,
            reasoning=reasoning,
        )

    if "prompt_token_count" in metrics:
        candidates = _int(metrics.get("candidates_token_count")) or 0
        thoughts = _int(metrics.get("thoughts_token_count"))
        return _finish(
            prompt=_int(metrics.get("prompt_token_count")) or 0,
            cached=_int(metrics.get("cached_content_token_count")) or 0,
            cache_write=0,
            completion=candidates + (thoughts or 0),
            reasoning=thoughts,
        )

    if "input_tokens" in metrics and (
        "cache_read_input_tokens" in metrics or "cache_creation_input_tokens" in metrics
    ) and "input_tokens_details" not in metrics:
        uncached = _int(metrics.get("input_tokens")) or 0
        cached = _int(metrics.get("cache_read_input_tokens")) or 0
        cache_write = _int(metrics.get("cache_creation_input_tokens")) or 0
        return _finish(
            prompt=uncached + cached + cache_write,
            cached=cached,
            cache_write=cache_write,
            completion=_int(metrics.get("output_tokens")) or 0,
            reasoning=None,
        )

    if "input_tokens" in metrics or "output_tokens" in metrics:
        return _finish(
            prompt=_int(metrics.get("input_tokens")) or 0,
            cached=_int(_sub(metrics, "input_tokens_details").get("cached_tokens")) or 0,
            cache_write=0,
            completion=_int(metrics.get("output_tokens")) or 0,
            reasoning=_int(_sub(metrics, "output_tokens_details").get("reasoning_tokens")),
        )

    return None


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


def utc_now_iso() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat(timespec="milliseconds")


def build_record(
    *,
    role: str,
    model: str | None,
    usage: Mapping[str, Any] | None,
    raw_usage: Any = None,
    status: str = "ok",
    provider: str | None = None,
    api: str | None = None,
    response_model: str | None = None,
    http_status: int | None = None,
    retry_index: int | None = None,
    response_id: str | None = None,
    started_at: str | None = None,
    latency_ms: float | None = None,
    cost_usd: float | None = None,
    step: int | None = None,
    phase_index: int | None = None,
    call_site: str | None = None,
    agent_harness: str | None = None,
) -> dict[str, Any]:
    """Assemble one ledger record. Identity fields are stamped at finalize."""
    if role not in ROLES:
        role = "other"
    return {
        "schema": SCHEMA,
        "task_id": None,
        "attempt_id": None,
        "agent_harness": agent_harness,
        "phase_index": phase_index,
        "step": step,
        "call_index": None,
        "role": role,
        "counts_toward_agent": role not in NON_AGENT_ROLES,
        "call_site": call_site,
        "provider": provider,
        "api": api,
        "model": model,
        "response_model": response_model,
        "status": status,
        "http_status": http_status,
        "retry_index": retry_index,
        "response_id": response_id,
        "started_at": started_at,
        "latency_ms": latency_ms,
        "usage": dict(usage) if usage is not None else None,
        "cost_usd": cost_usd,
        "raw_usage": to_jsonable(raw_usage),
    }


class TokenUsageLedger:
    """Append-only JSONL writer; one record per model call, flushed per write.

    ``append`` never raises: ledger failures must not disturb the agent loop.
    """

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._count = 0

    def append(self, record: Mapping[str, Any]) -> None:
        try:
            with self._lock:
                self._count += 1
                line = json.dumps(
                    {**record, "call_index": self._count},
                    ensure_ascii=False,
                    default=str,
                )
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
                    fh.flush()
        except Exception as exc:                                # noqa: BLE001
            logger.warning("token_usage: failed to append to %s: %s", self.path, exc)


def read_ledger(path: str | os.PathLike[str]) -> list[dict[str, Any]] | None:
    """Read a ledger file. Returns None if the file does not exist.

    Undecodable lines (e.g. a line truncated by a kill) are skipped.
    """
    p = Path(path)
    if not p.is_file():
        return None
    records: list[dict[str, Any]] = []
    with p.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                records.append(obj)
    return records


def finalize_attempt_ledger(
    *,
    attempt_dir: str | os.PathLike[str],
    work_dir: str | os.PathLike[str] | None,
    stream_records: Iterable[Mapping[str, Any]] | None,
    task_id: str,
    attempt_id: str,
    agent_harness: str,
) -> Path | None:
    """Write ``<attempt_dir>/token_usage.jsonl`` from the live ledger and stream records.

    Writes nothing (returns None) when the harness produced neither a live
    ledger nor stream records, i.e. the harness does not record usage. Never
    raises.
    """
    try:
        live = read_ledger(Path(work_dir) / LEDGER_FILENAME) if work_dir else None
        stream = list(stream_records) if stream_records is not None else None
        if live is None and stream is None:
            return None
        merged = [dict(r) for r in (live or [])] + [dict(r) for r in (stream or [])]
        for index, record in enumerate(merged, start=1):
            record["schema"] = SCHEMA
            record["task_id"] = task_id
            record["attempt_id"] = attempt_id
            record["agent_harness"] = agent_harness
            record["call_index"] = index
        out = Path(attempt_dir) / LEDGER_FILENAME
        tmp = out.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for record in merged:
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        tmp.replace(out)
        return out
    except Exception as exc:                                    # noqa: BLE001
        logger.warning("token_usage: finalize failed for %s: %s", attempt_dir, exc)
        return None


async def flush_pending(timeout_s: float = 30.0) -> None:
    """Drain LiteLLM's async logging queue so queued usage records are written.

    No-op when LiteLLM is not loaded. Never raises.
    """
    if "litellm" not in sys.modules:
        return
    try:
        from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
    except Exception:                                           # noqa: BLE001
        return
    # ``GLOBAL_LOGGING_WORKER.flush()`` returns as soon as the queue is empty,
    # even while the last dequeued handler is still running; ``join`` waits for
    # every handler to finish.
    queue = getattr(GLOBAL_LOGGING_WORKER, "_queue", None)
    join = getattr(queue, "join", None)
    flush = join or getattr(GLOBAL_LOGGING_WORKER, "flush", None)
    if flush is None:
        return
    try:
        await asyncio.wait_for(flush(), timeout=timeout_s)
    except Exception as exc:                                    # noqa: BLE001
        logger.debug("token_usage: litellm logging flush failed: %s", exc)
