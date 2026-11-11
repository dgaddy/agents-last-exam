"""ale_claw records every LiteLLM call into the token-usage ledger."""

from __future__ import annotations

import asyncio
from pathlib import Path

import litellm
import pytest

from ale_run.agents.ale_claw.harness.model.model_config import call_with_retry_budget
from ale_run.agents.ale_claw.harness.model.usage_ledger import (
    TokenUsageCallback,
    install_token_usage_callback,
    resolve_role,
)
from ale_run.base_interface.token_usage import (
    LEDGER_FILENAME,
    TokenUsageLedger,
    flush_pending,
    read_ledger,
    usage_scope,
)


@pytest.fixture
def ledger_path(tmp_path: Path):
    before = list(litellm.callbacks or [])
    path = tmp_path / LEDGER_FILENAME
    install_token_usage_callback(path)
    yield path
    litellm.callbacks = [cb for cb in before if not isinstance(cb, TokenUsageCallback)]
    for name in ("input_callback", "success_callback", "failure_callback",
                 "_async_success_callback", "_async_failure_callback"):
        lst = getattr(litellm, name, None)
        if isinstance(lst, list):
            lst[:] = [cb for cb in lst if not isinstance(cb, TokenUsageCallback)]


async def _mock_call(label: str, content: str = "ok"):
    return await call_with_retry_budget(
        lambda: litellm.acompletion(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "hello"}],
            mock_response=content,
        ),
        label=label,
    )


@pytest.mark.parametrize(
    ("scope_role", "call_site", "expected"),
    [
        (None, "agent_loop.predict_step(m)", "agent"),
        (None, "predict_step(m)", "agent"),
        (None, "helper:compaction(m)", "compaction"),
        (None, "helper:memory_flush(m)", "compaction"),
        (None, "helper:vision(m)", "tool_llm"),
        (None, "analyze_image(m)", "tool_llm"),
        (None, "subagent(m)", "subagent"),
        ("subagent", "predict_step(m)", "subagent"),
        ("subagent", "helper:compaction(m)", "compaction"),
        ("subagent", None, "subagent"),
        (None, None, "agent"),
    ],
)
def test_resolve_role(scope_role, call_site, expected):
    assert resolve_role(scope_role, call_site) == expected


async def test_every_litellm_call_lands_in_the_ledger_with_roles(ledger_path: Path):
    await _mock_call("agent_loop.predict_step(m)")
    await _mock_call("helper:memory_flush(m)")
    await _mock_call("analyze_image(m)")
    with usage_scope(role="subagent"):
        await _mock_call("predict_step(m)")
    await _mock_call("agent_loop.predict_step(m)")
    await flush_pending()

    records = read_ledger(ledger_path)
    assert [r["role"] for r in records] == [
        "agent", "compaction", "tool_llm", "subagent", "agent",
    ]
    assert [r["step"] for r in records] == [1, 1, 1, 1, 2]
    assert [r["call_index"] for r in records] == [1, 2, 3, 4, 5]
    for r in records:
        assert r["status"] == "ok"
        assert r["counts_toward_agent"] is True
        assert r["model"] == "gpt-4o-mini"
        assert r["usage"]["prompt_tokens"] > 0
        assert r["usage"]["completion_tokens"] > 0
        assert r["raw_usage"]["prompt_tokens"] == r["usage"]["prompt_tokens"]
        assert "messages" not in r and "content" not in r


async def test_failed_call_is_recorded_as_error(ledger_path: Path):
    with pytest.raises(litellm.BadRequestError):
        await litellm.acompletion(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "hello"}],
            mock_response=litellm.BadRequestError(
                message="bad", model="gpt-4o-mini", llm_provider="openai",
            ),
            num_retries=0,
        )
    await flush_pending()
    records = read_ledger(ledger_path)
    assert records and records[-1]["status"] == "error"
    assert records[-1]["usage"] is None


async def test_thinking_only_turn_with_empty_choices_is_recorded(tmp_path: Path):
    path = tmp_path / LEDGER_FILENAME
    callback = TokenUsageCallback(TokenUsageLedger(path))
    response = {
        "id": "resp-1",
        "model": "gemini-x",
        "choices": [],
        "usage": {
            "prompt_tokens": 40_000,
            "completion_tokens": 900,
            "prompt_tokens_details": {"cached_tokens": 38_000},
            "completion_tokens_details": {"reasoning_tokens": 900},
        },
    }
    kwargs = {"litellm_call_id": "c1", "model": "gemini-x", "call_type": "acompletion",
              "custom_llm_provider": "gemini"}
    callback.log_pre_api_call("gemini-x", [], kwargs)
    with usage_scope(call_site="predict_step(gemini-x)"):
        pass
    await callback.async_log_success_event(kwargs, response, None, None)
    (record,) = read_ledger(path)
    assert record["role"] == "agent"
    assert record["usage"]["completion_tokens"] == 900
    assert record["usage"]["reasoning_tokens"] == 900
    assert record["usage"]["response_tokens"] == 0
    assert record["usage"]["cached_tokens"] == 38_000
    assert record["response_id"] == "resp-1"


async def test_scope_is_captured_at_call_start(tmp_path: Path):
    path = tmp_path / LEDGER_FILENAME
    callback = TokenUsageCallback(TokenUsageLedger(path))
    kwargs = {"litellm_call_id": "c2", "model": "m"}
    with usage_scope(call_site="helper:compaction(m)"):
        callback.log_pre_api_call("m", [], kwargs)
    # The success handler runs later, outside the caller's context.
    await asyncio.sleep(0)
    await callback.async_log_success_event(
        kwargs, {"usage": {"prompt_tokens": 1, "completion_tokens": 1}}, None, None,
    )
    (record,) = read_ledger(path)
    assert record["role"] == "compaction"
    assert record["call_site"] == "helper:compaction(m)"


def test_install_retargets_single_callback(tmp_path: Path, ledger_path: Path):
    install_token_usage_callback(tmp_path / "second.jsonl")
    ours = [cb for cb in litellm.callbacks if isinstance(cb, TokenUsageCallback)]
    assert len(ours) == 1
