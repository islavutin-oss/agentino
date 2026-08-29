"""Tests for Anthropic provider support in LLM client."""

from agentino.core.llm import LLMClient, _detect_provider, _is_setup_token
from agentino.core.message import Message, ToolCall
from agentino.core.tool import tool


class TestProviderDetection:
    def test_detects_anthropic(self):
        assert _detect_provider("https://api.anthropic.com") == "anthropic"

    def test_detects_openai_as_codex(self):
        assert _detect_provider("https://api.openai.com/v1") == "openai-codex"

    def test_detects_custom_as_codex(self):
        assert _detect_provider("http://localhost:8100/v1") == "openai-codex"


class TestSetupTokenDetection:
    def test_detects_setup_token(self):
        assert _is_setup_token("sk-ant-oat-abc123") is True

    def test_rejects_regular_key(self):
        assert _is_setup_token("sk-ant-api01-abc123") is False

    def test_rejects_openai_key(self):
        assert _is_setup_token("sk-proj-abc123") is False


class TestAutoDetectFromKey:
    def test_anthropic_key_auto_detects_provider(self):
        client = LLMClient(api_key="sk-ant-api01-test123")
        assert client.provider == "anthropic"
        assert "anthropic" in client.base_url

    def test_setup_token_auto_detects_provider(self):
        client = LLMClient(api_key="sk-ant-oat-test123")
        assert client.provider == "anthropic"
        assert client._is_oauth is True

    def test_openai_key_defaults_to_codex(self):
        client = LLMClient(api_key="sk-proj-test123", base_url="https://api.openai.com/v1")
        assert client.provider == "openai-codex"


class TestAnthropicHeaders:
    def test_api_key_headers(self):
        client = LLMClient(
            base_url="https://api.anthropic.com",
            api_key="sk-ant-test",
            provider="anthropic",
        )
        headers = client._headers()
        assert headers["x-api-key"] == "sk-ant-test"
        assert "anthropic-version" in headers
        assert "Authorization" not in headers
        assert "anthropic-beta" not in headers

    def test_setup_token_headers(self):
        client = LLMClient(
            api_key="sk-ant-oat-test123",
            provider="anthropic",
        )
        headers = client._headers()
        assert headers["Authorization"] == "Bearer sk-ant-oat-test123"
        assert "x-api-key" not in headers
        assert "oauth-2025-04-20" in headers["anthropic-beta"]
        assert "claude-code-20250219" in headers["anthropic-beta"]

    def test_openai_headers(self):
        client = LLMClient(
            base_url="https://api.openai.com/v1",
            api_key="sk-test",
            provider="openai",
        )
        headers = client._headers()
        assert headers["Authorization"] == "Bearer sk-test"
        assert "x-api-key" not in headers


class TestAnthropicBodyBuilder:
    def test_builds_basic_body(self):
        client = LLMClient(
            base_url="https://api.anthropic.com",
            api_key="sk-ant-test",
            provider="anthropic",
        )
        messages = [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="Hello"),
        ]
        body = client._build_anthropic_body(
            messages, tools=None, model="claude-sonnet-4-20250514", temperature=0.7
        )
        assert body["model"] == "claude-sonnet-4-20250514"
        assert body["system"] == "You are helpful."
        assert body["max_tokens"] == 4096
        # System message should NOT be in messages array
        assert len(body["messages"]) == 1
        assert body["messages"][0]["role"] == "user"

    def test_builds_body_with_tools(self):
        @tool
        def search(query: str) -> str:
            """Search for things."""
            return query

        client = LLMClient(
            base_url="https://api.anthropic.com",
            api_key="sk-ant-test",
            provider="anthropic",
        )
        messages = [Message(role="user", content="Search for cats")]
        body = client._build_anthropic_body(
            messages, tools=[search], model="claude-sonnet-4-20250514", temperature=0.5
        )
        assert "tools" in body
        assert body["tools"][0]["name"] == "search"
        assert "input_schema" in body["tools"][0]

    def test_tool_call_message_format(self):
        client = LLMClient(
            base_url="https://api.anthropic.com",
            api_key="sk-ant-test",
            provider="anthropic",
        )
        messages = [
            Message(role="user", content="Search"),
            Message(
                role="assistant",
                content="Let me search",
                tool_calls=[ToolCall(id="toolu_123", name="search", arguments={"query": "cats"})],
            ),
            Message(role="tool", content="Found 5 cats", tool_call_id="toolu_123", name="search"),
        ]
        body = client._build_anthropic_body(
            messages, tools=None, model="claude-sonnet-4-20250514", temperature=0.7
        )
        # Assistant message should have tool_use block
        assistant_msg = body["messages"][1]
        assert assistant_msg["role"] == "assistant"
        content = assistant_msg["content"]
        assert any(b["type"] == "tool_use" for b in content)
        # Tool result should be a user message with tool_result block
        tool_msg = body["messages"][2]
        assert tool_msg["role"] == "user"
        assert tool_msg["content"][0]["type"] == "tool_result"


class TestAnthropicResponseParser:
    def test_parses_text_response(self):
        client = LLMClient(
            base_url="https://api.anthropic.com",
            api_key="sk-ant-test",
            provider="anthropic",
        )
        data = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        response = client._parse_anthropic_response(data)
        assert response.message.content == "Hello!"
        assert response.message.role == "assistant"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 5
        assert response.finish_reason == "stop"

    def test_parses_tool_use_response(self):
        client = LLMClient(
            base_url="https://api.anthropic.com",
            api_key="sk-ant-test",
            provider="anthropic",
        )
        data = {
            "id": "msg_456",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me search."},
                {
                    "type": "tool_use",
                    "id": "toolu_789",
                    "name": "search",
                    "input": {"query": "cats"},
                },
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 20, "output_tokens": 15},
        }
        response = client._parse_anthropic_response(data)
        assert response.message.content == "Let me search."
        assert len(response.message.tool_calls) == 1
        assert response.message.tool_calls[0].name == "search"
        assert response.message.tool_calls[0].arguments == {"query": "cats"}
        assert response.finish_reason == "tool_use"
