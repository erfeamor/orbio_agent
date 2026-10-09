"""Provider-independent LLM interface."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class ChatMessage:
    role: Literal["user", "assistant"]
    content: str


class LLMError(Exception):
    """Base class for LLM failures."""


class LLMConfigurationError(LLMError):
    """Not recoverable by retrying: bad credentials, unknown model, malformed request."""


class RecoverableLLMError(LLMError):
    """The turn failed but the conversation can continue; the student can try again."""


class LLMUnavailableError(RecoverableLLMError):
    """Timeouts, rate limits, network errors or server errors that outlasted the retries."""


class LLMRefusalError(RecoverableLLMError):
    """The model declined to answer."""


class LLMOutputError(RecoverableLLMError):
    """The model's output did not match the expected schema, even after a retry."""


class LLMClient(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @abstractmethod
    def generate(self, *, system: str, messages: Sequence[ChatMessage], output_type: type[T]) -> T:
        """Return the model's answer validated as ``output_type``.

        Callers put anything that changes per call in the final message; everything before it
        is expected to repeat on the next call, so implementations may cache that prefix.
        Raises an ``LLMError`` subclass on failure.
        """
