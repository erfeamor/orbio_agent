from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import pytest
from pydantic import BaseModel

from agent.llm.base import (
    ChatMessage,
    LLMConfigurationError,
    LLMOutputError,
    LLMRefusalError,
    LLMUnavailableError,
)
from agent.llm.claude import ClaudeLLM


class Answer(BaseModel):
    value: int


class FakeMessages:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = outcomes
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def response(text: str, stop_reason: str = "end_turn") -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="thinking"), SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        model="test-model",
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            cache_read_input_tokens=None,
            cache_creation_input_tokens=None,
        ),
        _request_id="req_test",
    )


def make_llm(*outcomes: Any) -> tuple[ClaudeLLM, FakeMessages]:
    messages = FakeMessages(list(outcomes))
    client = SimpleNamespace(messages=messages)
    return ClaudeLLM(
        "test-model",
        effort="low",
        timeout=5.0,
        max_retries=0,
        client=client,  # type: ignore[arg-type]
    ), messages


def generate(llm: ClaudeLLM) -> Answer:
    return llm.generate(system="sys", messages=[ChatMessage("user", "hi")], output_type=Answer)


REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def status_error(cls: type[anthropic.APIStatusError], code: int) -> anthropic.APIStatusError:
    return cls("error", response=httpx2.Response(code, request=REQUEST), body=None)


def test_returns_validated_output_and_sends_structured_output_config() -> None:
    llm, messages = make_llm(response('{"value": 3}'))

    assert generate(llm) == Answer(value=3)

    call = messages.calls[0]
    assert call["model"] == "test-model"
    assert call["output_config"]["effort"] == "low"
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert call["extra_body"] == {"fallbacks": "default"}
    assert call["messages"] == [{"role": "user", "content": "hi"}]


def test_invalid_output_is_retried_with_the_validation_error() -> None:
    llm, messages = make_llm(response('{"value": "three"}'), response('{"value": 3}'))

    assert generate(llm) == Answer(value=3)

    retry_messages = messages.calls[1]["messages"]
    assert retry_messages[1] == {"role": "assistant", "content": '{"value": "three"}'}
    assert "did not match the required schema" in retry_messages[2]["content"]


def test_output_still_invalid_after_retry_raises() -> None:
    llm, _ = make_llm(response("{"), response("{"))
    with pytest.raises(LLMOutputError):
        generate(llm)


def test_refusal_raises() -> None:
    llm, _ = make_llm(response("", stop_reason="refusal"))
    with pytest.raises(LLMRefusalError):
        generate(llm)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (anthropic.APIConnectionError(request=REQUEST), LLMUnavailableError),
        (anthropic.APITimeoutError(request=REQUEST), LLMUnavailableError),
        (status_error(anthropic.RateLimitError, 429), LLMUnavailableError),
        (status_error(anthropic.InternalServerError, 500), LLMUnavailableError),
        (status_error(anthropic.AuthenticationError, 401), LLMConfigurationError),
        (status_error(anthropic.BadRequestError, 400), LLMConfigurationError),
        (status_error(anthropic.NotFoundError, 404), LLMConfigurationError),
        (TypeError("Could not resolve authentication method."), LLMConfigurationError),
    ],
)
def test_sdk_errors_are_mapped(error: Exception, expected: type[Exception]) -> None:
    llm, _ = make_llm(error)
    with pytest.raises(expected):
        generate(llm)


def test_unrelated_type_errors_are_not_hidden() -> None:
    llm, _ = make_llm(TypeError("something else"))
    with pytest.raises(TypeError, match="something else"):
        generate(llm)


def test_response_without_text_is_treated_as_invalid_output() -> None:
    empty = response("")
    empty.content = []
    llm, messages = make_llm(empty, response('{"value": 1}'))

    assert generate(llm) == Answer(value=1)
    assert messages.calls[1]["messages"][1] == {"role": "assistant", "content": "(empty)"}


def test_cache_marker_goes_on_the_message_before_the_last() -> None:
    llm, messages = make_llm(response('{"value": 1}'))
    history = [
        ChatMessage("user", "opener"),
        ChatMessage("assistant", "question"),
        ChatMessage("user", "<interview_state>...</interview_state> answer"),
    ]

    llm.generate(system="sys", messages=history, output_type=Answer)

    sent = messages.calls[0]["messages"]
    assert sent[0] == {"role": "user", "content": "opener"}
    assert sent[1] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "question", "cache_control": {"type": "ephemeral"}}],
    }
    assert sent[2]["content"] == "<interview_state>...</interview_state> answer"
