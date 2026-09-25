"""Tests for unified_loop message conversion, reasoning replay, and malformed tool calls."""

import builtins
import threading
from types import SimpleNamespace

from ale_run.agents.ale_claw.harness.model import unified_loop


class TestConvertInputToMessages:
    """Tests for _convert_input_to_messages reasoning and assistant turn merging."""

    def test_reasoning_and_function_call(self) -> None:
        """Prior turn with reasoning + function_call (no output_text) merges into one assistant message."""
        msgs = unified_loop._convert_input_to_messages([
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "Step 1 thought"}],
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "exec",
                "arguments": '{"command": "ls -la"}',
            },
        ])
        assert len(msgs) == 1
        assert msgs[0]["role"] == "assistant"
        assert msgs[0]["content"] is None
        assert msgs[0]["reasoning_content"] == "Step 1 thought"
        assert len(msgs[0]["tool_calls"]) == 1
        assert msgs[0]["tool_calls"][0]["id"] == "call_1"

    def test_reasoning_message_and_function_call_merged(self) -> None:
        """Prior turn with reasoning + message + function_call merges into a single assistant message."""
        msgs = unified_loop._convert_input_to_messages([
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "Step 2 thought"}],
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Running read tool"}],
            },
            {
                "type": "function_call",
                "call_id": "call_2",
                "name": "read",
                "arguments": '{"path": "/workspace/README.md"}',
            },
        ])
        assert len(msgs) == 1
        assert msgs[0]["role"] == "assistant"
        assert msgs[0]["content"] == "Running read tool"
        assert msgs[0]["reasoning_content"] == "Step 2 thought"
        assert len(msgs[0]["tool_calls"]) == 1
        assert msgs[0]["tool_calls"][0]["id"] == "call_2"

    def test_canonical_assistant_thinking_text_and_tool_use(self) -> None:
        """Canonical assistant message with thinking + text + tool_use populates reasoning_content."""
        msgs = unified_loop._convert_input_to_messages([
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Canonical thought"},
                    {"type": "text", "text": "Canonical text"},
                    {
                        "type": "tool_use",
                        "id": "call_3",
                        "name": "exec",
                        "input": {"command": "pwd"},
                    },
                ],
            },
        ])
        assert len(msgs) == 1
        assert msgs[0]["content"] == "Canonical text"
        assert msgs[0]["reasoning_content"] == "Canonical thought"
        assert len(msgs[0]["tool_calls"]) == 1
        assert msgs[0]["tool_calls"][0]["id"] == "call_3"

    def test_function_call_preserves_provider_specific_fields_and_extra_content(self) -> None:
        """_convert_response_to_output and _convert_input_to_messages preserve provider_specific_fields and extra_content."""
        tc = SimpleNamespace(
            id="call_sig_1",
            function=SimpleNamespace(name="exec", arguments='{"command": "ls"}'),
            provider_specific_fields={"thought_signature": "sig_abc"},
            extra_content={"google": {"thought_signature": "sig_abc"}},
        )
        msg = SimpleNamespace(
            reasoning_content=None,
            provider_specific_fields=None,
            content="Let me run ls.",
            tool_calls=[tc],
        )
        resp = SimpleNamespace(
            choices=[SimpleNamespace(message=msg, finish_reason="tool_calls")],
            usage=None,
        )
        out = unified_loop._convert_response_to_output(resp)
        assert out["malformed_tool_call"] is False
        fc_items = [x for x in out["output"] if x.get("type") == "function_call"]
        assert len(fc_items) == 1
        assert fc_items[0]["provider_specific_fields"] == {"thought_signature": "sig_abc"}
        assert fc_items[0]["extra_content"] == {"google": {"thought_signature": "sig_abc"}}

        rebuilt = unified_loop._convert_input_to_messages(out["output"])
        assert rebuilt[0]["tool_calls"][0]["provider_specific_fields"] == {
            "thought_signature": "sig_abc"
        }
        assert rebuilt[0]["tool_calls"][0]["extra_content"] == {
            "google": {"thought_signature": "sig_abc"}
        }

    def test_malformed_function_call_finish_reason_propagated(self) -> None:
        """_convert_response_to_output propagates MALFORMED_FUNCTION_CALL without modifying text."""
        if not hasattr(builtins, "_malformed_fc_cache"):
            builtins._malformed_fc_cache = {}
        builtins._malformed_fc_cache[threading.get_ident()] = (
            "Malformed function call: Failed to parse function call: exec"
        )
        msg = SimpleNamespace(
            reasoning_content=None,
            provider_specific_fields=None,
            content="Let me check the files.",
            tool_calls=None,
        )
        resp = SimpleNamespace(
            choices=[SimpleNamespace(message=msg, finish_reason="stop")],
            usage=None,
        )
        out = unified_loop._convert_response_to_output(resp)
        assert out["malformed_tool_call"] is True
        assert out["finish_reason"] == "malformed_function_call"
        assert "Failed to parse function call: exec" in out["malformed_finish_message"]
        msg_items = [x for x in out["output"] if x.get("type") == "message"]
        assert len(msg_items) == 1
        assert msg_items[0]["content"][0]["text"] == "Let me check the files."


class TestMaybeNudgeBareText:
    """Tests for OpenClawComputerAgent._maybe_nudge_bare_text malformed tool call feedback."""

    def test_malformed_with_zero_tool_calls(self) -> None:
        """When MALFORMED_FUNCTION_CALL occurs with 0 parsed tool calls, model gets explicit error."""
        from ale_run.agents.ale_claw.harness.agent_loop import OpenClawComputerAgent

        recorded: list[tuple[str, str]] = []
        dummy_agent = SimpleNamespace(
            session_mgr=SimpleNamespace(
                append_message=lambda role, text: recorded.append((role, text))
            )
        )
        new_items: list[dict] = []
        OpenClawComputerAgent._maybe_nudge_bare_text(
            dummy_agent,
            {
                "output": [{"type": "message", "role": "assistant", "content": []}],
                "malformed_tool_call": True,
                "finish_reason": "malformed_function_call",
                "malformed_finish_message": "Failed to parse function call: exec",
            },
            new_items,
        )
        assert len(new_items) == 1
        assert "finishReason=MALFORMED_FUNCTION_CALL" in new_items[0]["content"]
        assert "Failed to parse function call: exec" in new_items[0]["content"]
        assert "no tool was executed" in new_items[0]["content"]

    def test_malformed_alongside_valid_tool_call(self) -> None:
        """When MALFORMED_FUNCTION_CALL occurs alongside a valid tool call, model is informed."""
        from ale_run.agents.ale_claw.harness.agent_loop import OpenClawComputerAgent

        recorded: list[tuple[str, str]] = []
        dummy_agent = SimpleNamespace(
            session_mgr=SimpleNamespace(
                append_message=lambda role, text: recorded.append((role, text))
            )
        )
        new_items: list[dict] = []
        OpenClawComputerAgent._maybe_nudge_bare_text(
            dummy_agent,
            {
                "output": [{"type": "function_call", "call_id": "call_1", "name": "exec"}],
                "malformed_tool_call": True,
                "finish_reason": "malformed_function_call",
                "malformed_finish_message": "Failed to parse function call: exec",
            },
            new_items,
        )
        assert len(new_items) == 1
        assert "finishReason=MALFORMED_FUNCTION_CALL" in new_items[0]["content"]
        assert "Only the validly parsed tool calls above were executed" in new_items[0]["content"]

