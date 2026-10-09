"""Scripted LLM for tests and offline runs."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from pydantic import BaseModel

from agent.llm.base import ChatMessage, LLMClient, T


@dataclass(frozen=True)
class FakeCall:
    system: str
    messages: tuple[ChatMessage, ...]
    output_type: type[BaseModel]


class FakeLLM(LLMClient):
    """Returns queued responses in order. A queued exception is raised instead of returned."""

    def __init__(self, responses: Iterable[BaseModel | Exception] = ()) -> None:
        self._responses = list(responses)
        self.calls: list[FakeCall] = []

    @property
    def model_name(self) -> str:
        return "fake"

    def queue(self, *responses: BaseModel | Exception) -> None:
        self._responses.extend(responses)

    @property
    def pending(self) -> int:
        return len(self._responses)

    def generate(self, *, system: str, messages: Sequence[ChatMessage], output_type: type[T]) -> T:
        self.calls.append(FakeCall(system, tuple(messages), output_type))
        if not self._responses:
            raise AssertionError(f"FakeLLM has no response queued for {output_type.__name__}")
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if not isinstance(response, output_type):
            raise AssertionError(
                f"FakeLLM expected to return {output_type.__name__}, "
                f"queued {type(response).__name__}"
            )
        return response
