"""Tests for unified_loop message conversion and reasoning replay."""

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
