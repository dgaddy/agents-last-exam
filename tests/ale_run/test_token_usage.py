"""Tests for the per-call token usage ledger (``token_usage.jsonl``)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ale_run.agents.antigravity_cli.metrics import agy_usage, build_token_usage_records
from ale_run.agents.grok_cli.config import GrokCliConfig
from ale_run.agents.grok_cli.deployer import GrokCliDeployer
from ale_run.base_interface import AgentRunResult
from ale_run.base_interface.token_usage import (
    LEDGER_FILENAME,
    SCHEMA,
    TokenUsageLedger,
    build_record,
    current_scope,
    finalize_attempt_ledger,
    normalize_usage,
    read_ledger,
    usage_scope,
)


def _partition_ok(u: dict) -> bool:
    return u["prompt_tokens"] == (
        u["cached_tokens"] + u["cache_write_tokens"] + u["uncached_prompt_tokens"]
    )


# ---------------------------------------------------------------------------
# normalize_usage
# ---------------------------------------------------------------------------


def test_litellm_gemini_clamps_cached_above_prompt():
    u = normalize_usage({
        "prompt_tokens": 1000,
        "completion_tokens": 300,
        "total_tokens": 1300,
        "prompt_tokens_details": {"cached_tokens": 1200, "text_tokens": -200},
        "completion_tokens_details": {"reasoning_tokens": 120, "text_tokens": 180},
    })
    assert u == {
        "prompt_tokens": 1000,
        "cached_tokens": 1000,
        "cache_write_tokens": 0,
        "uncached_prompt_tokens": 0,
        "completion_tokens": 300,
        "reasoning_tokens": 120,
        "response_tokens": 180,
    }


def test_litellm_gemini_without_reasoning_split_is_null():
    u = normalize_usage({
        "prompt_tokens": 30180,
        "completion_tokens": 357,
        "total_tokens": 30537,
        "prompt_tokens_details": {"cached_tokens": 27615, "text_tokens": 2565},
    })
    assert u["cached_tokens"] == 27615
    assert u["uncached_prompt_tokens"] == 2565
    assert u["reasoning_tokens"] is None and u["response_tokens"] is None


def test_litellm_anthropic_cache_writes_are_separated():
    # LiteLLM's Anthropic adapter folds cache reads and writes into prompt_tokens.
    u = normalize_usage({
        "completion_tokens": 374,
        "prompt_tokens": 11355,
        "total_tokens": 11729,
        "prompt_tokens_details": {"cached_tokens": 0, "cache_creation_tokens": 11353},
        "cache_creation_input_tokens": 11353,
        "cache_read_input_tokens": 0,
    })
    assert u["prompt_tokens"] == 11355
    assert u["cache_write_tokens"] == 11353
    assert u["cached_tokens"] == 0
    assert u["uncached_prompt_tokens"] == 2
    assert _partition_ok(u)


def test_native_anthropic_fields_are_disjoint_and_summed():
    u = normalize_usage({
        "input_tokens": 50,
        "cache_read_input_tokens": 9000,
        "cache_creation_input_tokens": 400,
        "output_tokens": 210,
    })
    assert u["prompt_tokens"] == 9450
    assert u["cached_tokens"] == 9000
    assert u["cache_write_tokens"] == 400
    assert u["uncached_prompt_tokens"] == 50
    assert u["completion_tokens"] == 210


def test_openai_chat_cached_and_reasoning():
    u = normalize_usage({
        "prompt_tokens": 2000,
        "completion_tokens": 900,
        "total_tokens": 2900,
        "prompt_tokens_details": {"cached_tokens": 1536},
        "completion_tokens_details": {"reasoning_tokens": 640},
    })
    assert u["cached_tokens"] == 1536
    assert u["uncached_prompt_tokens"] == 464
    assert u["completion_tokens"] == 900
    assert u["reasoning_tokens"] == 640
    assert u["response_tokens"] == 260


def test_openai_responses_api_shape():
    u = normalize_usage({
        "input_tokens": 5000,
        "input_tokens_details": {"cached_tokens": 4096},
        "output_tokens": 700,
        "output_tokens_details": {"reasoning_tokens": 500},
        "total_tokens": 5700,
    })
    assert u["prompt_tokens"] == 5000
    assert u["cached_tokens"] == 4096
    assert u["completion_tokens"] == 700
    assert u["reasoning_tokens"] == 500


def test_xai_chat_reasoning_added_back_by_invariant():
    u = normalize_usage({
        "prompt_tokens": 1000,
        "completion_tokens": 100,
        "total_tokens": 1400,
        "completion_tokens_details": {"reasoning_tokens": 300},
    })
    assert u["completion_tokens"] == 400
    assert u["reasoning_tokens"] == 300
    assert u["response_tokens"] == 100


def test_xai_chat_reasoning_added_back_by_provider():
    u = normalize_usage(
        {"prompt_tokens": 1000, "completion_tokens": 100,
         "completion_tokens_details": {"reasoning_tokens": 300}},
        provider="xai",
    )
    assert u["completion_tokens"] == 400


def test_gemini_usage_metadata_camel_case():
    u = normalize_usage({
        "promptTokenCount": 12000,
        "cachedContentTokenCount": 8000,
        "candidatesTokenCount": 150,
        "thoughtsTokenCount": 350,
    })
    assert u["prompt_tokens"] == 12000
    assert u["cached_tokens"] == 8000
    assert u["completion_tokens"] == 500
    assert u["reasoning_tokens"] == 350


def test_long_context_prompt_is_preserved():
    u = normalize_usage({
        "prompt_tokens": 412_345,
        "completion_tokens": 2_000,
        "prompt_tokens_details": {"cached_tokens": 400_000},
    })
    assert u["prompt_tokens"] == 412_345
    assert u["uncached_prompt_tokens"] == 12_345


def test_unrecognized_usage_returns_none():
    assert normalize_usage(None) is None
    assert normalize_usage({"foo": 1}) is None


def test_normalize_accepts_pydantic_like_objects():
    class FakeUsage:
        def model_dump(self):
            return {"prompt_tokens": 10, "completion_tokens": 5}

    assert normalize_usage(FakeUsage())["prompt_tokens"] == 10


# ---------------------------------------------------------------------------
# scope / records / ledger
# ---------------------------------------------------------------------------


def test_usage_scope_nests_and_resets():
    assert current_scope() == (None, None)
    with usage_scope(role="subagent"):
        with usage_scope(call_site="helper:compaction(m)"):
            assert current_scope() == ("subagent", "helper:compaction(m)")
        assert current_scope() == ("subagent", None)
    assert current_scope() == (None, None)


def test_build_record_attribution():
    agent = build_record(role="agent", model="m", usage=None)
    judge = build_record(role="judge", model="m", usage=None)
    bogus = build_record(role="weird", model="m", usage=None)
    assert agent["counts_toward_agent"] is True
    assert judge["counts_toward_agent"] is False
    assert bogus["role"] == "other"
    assert agent["schema"] == SCHEMA


def test_ledger_appends_and_flushes_each_record(tmp_path: Path):
    ledger = TokenUsageLedger(tmp_path / LEDGER_FILENAME)
    ledger.append(build_record(role="agent", model="m", usage={"prompt_tokens": 1}))
    # Visible on disk before the writer is closed / the process exits.
    assert len(read_ledger(tmp_path / LEDGER_FILENAME)) == 1
    ledger.append(build_record(role="agent", model="m", usage={"prompt_tokens": 2}))
    records = read_ledger(tmp_path / LEDGER_FILENAME)
    assert [r["call_index"] for r in records] == [1, 2]


def test_ledger_append_never_raises(tmp_path: Path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    TokenUsageLedger(blocker / "sub" / LEDGER_FILENAME).append({"a": object()})


def test_read_ledger_skips_truncated_lines(tmp_path: Path):
    p = tmp_path / LEDGER_FILENAME
    p.write_text(json.dumps({"call_index": 1}) + "\n" + '{"call_index": 2, "us')
    assert read_ledger(p) == [{"call_index": 1}]
    assert read_ledger(tmp_path / "missing.jsonl") is None


def test_finalize_merges_live_and_stream_and_stamps_identity(tmp_path: Path):
    work = tmp_path / "origin_log" / "ale-claw"
    work.mkdir(parents=True)
    ledger = TokenUsageLedger(work / LEDGER_FILENAME)
    ledger.append(build_record(role="agent", model="m", usage={"prompt_tokens": 1}))
    stream = [build_record(role="agent", model="m", usage={"prompt_tokens": 2})]
    out = finalize_attempt_ledger(
        attempt_dir=tmp_path, work_dir=work, stream_records=stream,
        task_id="tasks/x/y", attempt_id="a__v0__20260101_000000", agent_harness="h",
    )
    assert out == tmp_path / LEDGER_FILENAME
    records = read_ledger(out)
    assert [r["call_index"] for r in records] == [1, 2]
    assert {r["task_id"] for r in records} == {"tasks/x/y"}
    assert {r["attempt_id"] for r in records} == {"a__v0__20260101_000000"}
    assert {r["agent_harness"] for r in records} == {"h"}


def test_finalize_writes_nothing_for_unsupported_harness(tmp_path: Path):
    assert finalize_attempt_ledger(
        attempt_dir=tmp_path, work_dir=tmp_path / "w", stream_records=None,
        task_id="t", attempt_id="a", agent_harness="claude_code",
    ) is None
    assert not (tmp_path / LEDGER_FILENAME).exists()


def test_finalize_writes_empty_ledger_when_supported_harness_made_no_calls(tmp_path: Path):
    out = finalize_attempt_ledger(
        attempt_dir=tmp_path, work_dir=None, stream_records=[],
        task_id="t", attempt_id="a", agent_harness="antigravity_cli",
    )
    assert out is not None and read_ledger(out) == []


# ---------------------------------------------------------------------------
# antigravity_cli stream → ledger
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, events: list[dict]) -> None:
    path.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")


def _step(index: int, usage: dict | None, *, state: str = "DONE", step_type="agent_response"):
    update = {"step_index": index, "state": state, "step_type": step_type}
    if usage is not None:
        update["usage"] = usage
    return {"event": "step_update", "step_update": update}


def test_agy_usage_cache_reads_are_disjoint_and_output_includes_thinking():
    u = agy_usage({
        "input_tokens": 2731, "output_tokens": 388, "thinking_tokens": 164,
        "cache_read_tokens": 8674, "total_tokens": 3119,
    })
    assert u == {
        "prompt_tokens": 11405,
        "cached_tokens": 8674,
        "cache_write_tokens": 0,
        "uncached_prompt_tokens": 2731,
        "completion_tokens": 388,
        "reasoning_tokens": 164,
        "response_tokens": 224,
    }


def test_agy_records_dedupe_repeated_step_updates_keep_max(tmp_path: Path):
    _write_jsonl(tmp_path / "transcript.jsonl", [
        {"event": "init", "init": {"model": "gemini-x"}},
        _step(0, None, step_type="user_input"),
        _step(1, {"input_tokens": 100, "cache_read_tokens": 0, "output_tokens": 5},
              state="ACTIVE"),
        _step(1, {"input_tokens": 100, "cache_read_tokens": 0, "output_tokens": 40,
                  "thinking_tokens": 10}),
        _step(1, {"input_tokens": 100, "cache_read_tokens": 0, "output_tokens": 40,
                  "thinking_tokens": 10}),
        _step(2, None, step_type="tool"),
        _step(3, {"input_tokens": 30, "cache_read_tokens": 120, "output_tokens": 7}),
        {"event": "result", "result": {"status": "SUCCESS", "usage": {
            "input_tokens": 130, "cache_read_tokens": 120, "output_tokens": 47}}},
    ])
    records = build_token_usage_records(tmp_path, model="cfg-model", provider="gemini")
    assert [r["step"] for r in records] == [1, 3]
    assert records[0]["usage"]["completion_tokens"] == 40
    assert records[0]["model"] == "gemini-x"
    assert records[1]["usage"]["prompt_tokens"] == 150
    assert records[1]["usage"]["cached_tokens"] == 120
    assert all(r["role"] == "agent" and r["status"] == "ok" for r in records)
    total_uncached = sum(r["usage"]["uncached_prompt_tokens"] for r in records)
    total_cached = sum(r["usage"]["cached_tokens"] for r in records)
    total_out = sum(r["usage"]["completion_tokens"] for r in records)
    assert (total_uncached, total_cached, total_out) == (130, 120, 47)


def test_agy_resumed_conversation_keeps_both_segments(tmp_path: Path):
    _write_jsonl(tmp_path / "transcript.jsonl", [
        {"event": "init", "init": {"model": "m"}},
        _step(1, {"input_tokens": 10, "cache_read_tokens": 0, "output_tokens": 1}),
        {"event": "init", "init": {"model": "m"}},
        _step(1, {"input_tokens": 20, "cache_read_tokens": 5, "output_tokens": 2}),
    ])
    records = build_token_usage_records(tmp_path, model="m", provider=None)
    assert [r["usage"]["uncached_prompt_tokens"] for r in records] == [10, 20]


def test_agy_error_step_marked_error(tmp_path: Path):
    _write_jsonl(tmp_path / "transcript.jsonl", [
        _step(1, {"input_tokens": 10, "cache_read_tokens": 0, "output_tokens": 0},
              state="ERROR"),
    ])
    (record,) = build_token_usage_records(tmp_path, model="m", provider=None)
    assert record["status"] == "error"


def test_agy_missing_transcript_returns_none(tmp_path: Path):
    assert build_token_usage_records(tmp_path, model="m", provider=None) is None


# ---------------------------------------------------------------------------
# grok_cli stream → ledger
# ---------------------------------------------------------------------------


def test_grok_records_have_unknown_cache_split(tmp_path: Path):
    _write_jsonl(tmp_path / "transcript.jsonl", [
        {"type": "step_start", "stepNumber": 0},
        {"type": "step_finish", "stepNumber": 0, "finishReason": "tool-calls",
         "usage": {"inputTokens": 900, "outputTokens": 80, "totalTokens": 980,
                   "costUsdTicks": 1500}},
        {"type": "step_finish", "stepNumber": 1, "finishReason": "stop", "usage": {}},
    ])
    records = GrokCliDeployer.token_usage_records(
        work_dir=tmp_path,
        config=GrokCliConfig(model="grok-x"),
        run_result=AgentRunResult(status="completed"),
    )
    assert len(records) == 1
    (r,) = records
    assert r["usage"]["prompt_tokens"] == 900
    assert r["usage"]["completion_tokens"] == 80
    assert r["usage"]["cached_tokens"] is None
    assert r["provider"] == "xai"
    assert r["step"] == 0


@pytest.mark.parametrize("missing", [True, False])
def test_grok_records_without_transcript(tmp_path: Path, missing: bool):
    if not missing:
        (tmp_path / "transcript.jsonl").write_text("")
    records = GrokCliDeployer.token_usage_records(
        work_dir=tmp_path,
        config=GrokCliConfig(model="grok-x"),
        run_result=AgentRunResult(status="failed"),
    )
    assert records is None if missing else records == []
