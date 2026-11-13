"""Probe shared LLM judges before provisioning any selected run units."""

from __future__ import annotations

import ast
import asyncio
import io
import logging
import math
import os
import random
from pathlib import Path
from typing import Iterable

from .experiment_spec import RunUnit

logger = logging.getLogger(__name__)

_TEXT_ONLY_TASKS = frozenset({
    "business_finance/digital_marketing_ab_test_analysis_1",
    "engineering/humanoid_wbc_policy_evaluation",
})
_TASKS_ROOT = Path(__file__).resolve().parents[2] / "tasks"

# A probe is tried once by default. Long unattended runs can raise the attempt
# count so a brief judge outage does not abort the whole experiment.
_ATTEMPTS_ENV = "ALE_JUDGE_PREFLIGHT_ATTEMPTS"
_RETRY_DELAY_ENV = "ALE_JUDGE_PREFLIGHT_RETRY_DELAY_S"
_DEFAULT_RETRY_DELAY_S = 30.0
_MAX_RETRY_DELAY_S = 300.0


def judge_requirements(task_paths: Iterable[str]) -> dict[str, list[str]]:
    """Discover task-local shared judge imports without executing task modules."""
    required: dict[str, list[str]] = {}
    for task_path in sorted(set(task_paths)):
        directory = _TASKS_ROOT / task_path
        sources = list(directory.glob("*.py")) + list((directory / "scripts").rglob("*.py"))
        uses_judge = False
        for source in sources:
            if "evidence" in source.relative_to(directory).parts:
                continue
            tree = ast.parse(source.read_text(encoding="utf-8-sig"), filename=str(source))
            if any(
                isinstance(node, ast.ImportFrom)
                and node.module == "tasks.utils.evaluation"
                and any(
                    alias.name.startswith("llm_") or alias.name == "compare_screenshots_game"
                    for alias in node.names
                )
                for node in ast.walk(tree)
            ):
                uses_judge = True
                break
        if uses_judge:
            required.setdefault("text", []).append(task_path)
            if task_path not in _TEXT_ONLY_TASKS:
                required.setdefault("vision", []).append(task_path)
    return required


def _env_number(name: str, default: float, *, minimum: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if not math.isfinite(value):
        logger.warning("Ignoring invalid %s=%r; using %g", name, raw, default)
        return default
    return max(minimum, value)


def _http_status(error: BaseException) -> int | None:
    """Return the HTTP status carried by ``error`` or one of its causes."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        status = getattr(current, "status_code", None)
        if isinstance(status, int) and not isinstance(status, bool):
            return status
        current = current.__cause__ or current.__context__
    return None


def _is_permanent(status: int | None) -> bool:
    """Whether retrying cannot help: the judge rejected the request itself."""
    return status is not None and 400 <= status < 500 and status not in (408, 409, 429)


async def check_judges(units: Iterable[RunUnit]) -> None:
    required = judge_requirements(unit.task_path for unit in units)
    if not required:
        return

    from tasks.utils.evaluation import (
        JudgeInfrastructureError,
        build_vision_content,
        llm_multimodal_json,
    )

    attempts = int(_env_number(_ATTEMPTS_ENV, 1, minimum=1))
    retry_delay_s = _env_number(_RETRY_DELAY_ENV, _DEFAULT_RETRY_DELAY_S, minimum=0)

    async def probe(capability: str) -> None:
        if capability == "vision":
            from PIL import Image

            stream = io.BytesIO()
            Image.new("RGB", (64, 64), (255, 0, 0)).save(stream, format="PNG")
            content = build_vision_content(
                'What is the dominant color in this image? Return JSON {"color":"color name"}.',
                [stream.getvalue()],
            )
        else:
            content = [{"type": "text", "text": 'Return JSON {"sum":7+8}, evaluating the sum.'}]
        for attempt in range(1, attempts + 1):
            try:
                result = await asyncio.wait_for(
                    llm_multimodal_json(content=content, max_tokens=128), timeout=150,
                )
                valid = (
                    str(result.get("color", "")).strip().lower() == "red"
                    if capability == "vision"
                    else result.get("sum") == 15
                )
                if not valid:
                    raise JudgeInfrastructureError("Judge probe returned an incorrect answer")
                break
            except Exception as error:
                # Name the failure by exception type and HTTP status only. The
                # message is left out on purpose: provider errors can echo
                # request details, and this text ends up in logs and summaries.
                status = _http_status(error)
                failure = type(error).__name__ + (f", HTTP {status}" if status is not None else "")
                if attempt < attempts and not _is_permanent(status):
                    # Jittered so runs that start together do not retry in step.
                    delay_s = min(
                        retry_delay_s * 2 ** (attempt - 1) * random.uniform(0.5, 1.5),
                        _MAX_RETRY_DELAY_S,
                    )
                    logger.warning(
                        "Judge %s preflight attempt %d/%d failed (%s); retrying in %.0fs",
                        capability, attempt, attempts, failure, delay_s,
                    )
                    await asyncio.sleep(delay_s)
                    continue
                tried = f" after {attempt} attempts" if attempt > 1 else ""
                raise JudgeInfrastructureError(
                    f"Judge {capability} preflight failed ({failure}){tried}; "
                    "no sandboxes were started. Check JUDGE_BASE_URL, JUDGE_API_KEY, "
                    "JUDGE_MODEL and retry the experiment."
                ) from None
        logger.info("Judge %s preflight passed for %d selected task(s)", capability, len(required[capability]))

    await asyncio.gather(*(probe(capability) for capability in required))
