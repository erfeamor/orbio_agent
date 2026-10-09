"""Claude implementation of the LLM interface, using structured outputs."""

import logging
import time
from collections.abc import Sequence
from typing import Any, Literal

import anthropic
from anthropic import transform_schema
from anthropic.types import MessageParam
from pydantic import ValidationError

from agent.llm.base import (
    ChatMessage,
    LLMClient,
    LLMConfigurationError,
    LLMOutputError,
    LLMRefusalError,
    LLMUnavailableError,
    T,
)

logger = logging.getLogger(__name__)

Effort = Literal["low", "medium", "high", "xhigh", "max"]

# Server-side fallback: if the model's safety classifiers decline a request, the API re-runs
# it on Anthropic's recommended fallback model instead of returning a refusal.
_FALLBACK_BETA = "server-side-fallback-2026-07-01"
_MAX_TOKENS = 16000
# One retry when the output fails Pydantic validation (e.g. truncated JSON).
_OUTPUT_ATTEMPTS = 2


class ClaudeLLM(LLMClient):
    def __init__(
        self,
        model: str,
        *,
        effort: Effort,
        timeout: float,
        max_retries: int,
        api_key: str | None = None,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self._model = model
        self._effort: Effort = effort
        # The SDK retries timeouts, connection errors, 408/409/429 and 5xx with backoff.
        self._client = client or anthropic.Anthropic(
            api_key=api_key, timeout=timeout, max_retries=max_retries
        )

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self, *, system: str, messages: Sequence[ChatMessage], output_type: type[T]) -> T:
        schema = transform_schema(output_type)
        conversation: list[MessageParam] = [
            {"role": m.role, "content": m.content} for m in messages
        ]
        last_error = ""
        for attempt in range(1, _OUTPUT_ATTEMPTS + 1):
            text = self._call(system, conversation, schema, output_type.__name__)
            try:
                return output_type.model_validate_json(text)
            except ValidationError as exc:
                last_error = str(exc)
                logger.warning(
                    "invalid %s output (attempt %d): %s", output_type.__name__, attempt, exc
                )
                conversation = [
                    *conversation,
                    {"role": "assistant", "content": text or "(empty)"},
                    {
                        "role": "user",
                        "content": "That output did not match the required schema:\n"
                        f"{exc}\nReturn the corrected JSON only.",
                    },
                ]
        raise LLMOutputError(f"{output_type.__name__} output invalid after retry: {last_error}")

    def _call(
        self,
        system: str,
        messages: list[MessageParam],
        schema: dict[str, Any],
        label: str,
    ) -> str:
        started = time.monotonic()
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=_MAX_TOKENS,
                system=system,
                messages=messages,
                output_config={
                    "effort": self._effort,
                    "format": {"type": "json_schema", "schema": schema},
                },
                extra_headers={"anthropic-beta": _FALLBACK_BETA},
                extra_body={"fallbacks": "default"},
            )
        except TypeError as exc:
            # Raised by the SDK before sending when no API key or profile is configured.
            if "authentication" not in str(exc):
                raise
            raise LLMConfigurationError("no Anthropic credentials configured") from exc
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
            raise LLMConfigurationError(f"Anthropic credentials rejected: {exc}") from exc
        except (anthropic.BadRequestError, anthropic.NotFoundError) as exc:
            raise LLMConfigurationError(f"Anthropic rejected the request: {exc}") from exc
        except anthropic.RateLimitError as exc:
            raise LLMUnavailableError("rate limited") from exc
        except anthropic.APIStatusError as exc:
            raise LLMUnavailableError(f"Anthropic API error {exc.status_code}") from exc
        except anthropic.APIConnectionError as exc:  # includes APITimeoutError
            raise LLMUnavailableError(f"cannot reach Anthropic API: {exc}") from exc

        logger.info(
            "llm call %s model=%s stop=%s in=%d out=%d latency_ms=%d request_id=%s",
            label,
            response.model,
            response.stop_reason,
            response.usage.input_tokens,
            response.usage.output_tokens,
            (time.monotonic() - started) * 1000,
            response._request_id,
        )
        if response.stop_reason == "refusal":
            raise LLMRefusalError("the model declined to answer")
        return next((block.text for block in response.content if block.type == "text"), "")
