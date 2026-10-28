"""LLM provider adapters for Solace.

Adapters:
- GeminiProvider   — google-genai SDK
- OpenAICompatProvider — openai SDK with custom base_url (Groq, OpenRouter, Mistral, Cerebras)
- MockProvider     — deterministic scripted outputs for tests
"""

from __future__ import annotations

import json
import logging
import math
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextlib import suppress
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any, TypedDict

from app.contracts import TaskKind

logger = logging.getLogger(__name__)


# ── Base class ────────────────────────────────────────────────────────────────


class LLMResult:
    """Output of a single LLM call."""

    __slots__ = ("text", "parsed", "provider", "model", "latency_ms", "usage", "attempts")

    def __init__(
        self,
        text: str,
        provider: str,
        model: str,
        latency_ms: float,
        parsed: Any = None,
        usage: dict[str, int] | None = None,
        attempts: list[ProviderAttempt] | None = None,
    ) -> None:
        self.text = text
        self.parsed = parsed
        self.provider = provider
        self.model = model
        self.latency_ms = latency_ms
        self.usage = usage
        self.attempts = attempts or []

    def __repr__(self) -> str:
        return (
            f"LLMResult(provider={self.provider!r}, model={self.model!r}, "
            f"latency_ms={self.latency_ms:.0f})"
        )


class ProviderError(Exception):
    """Base exception for known provider API failures."""

    def __init__(self, provider: str, status: int | None, message: str) -> None:
        super().__init__(f"[{provider}] HTTP {status}: {message}")
        self.provider = provider
        self.status = status
        self.message = message


class RateLimitError(ProviderError):
    """Raised on HTTP 429 — triggers cooldown in the gateway."""

    def __init__(
        self,
        provider: str,
        status: int | None,
        message: str,
        *,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(provider, status, message)
        self.retry_after = retry_after


class BillingOrQuota(ProviderError):
    """Raised when provider billing, payment, or account quota is exhausted (e.g. HTTP 402)."""


BillingOrQuotaError = BillingOrQuota


class EmptyResponseError(ProviderError):
    """Raised when a provider returns an empty response (e.g. reasoning token budget exhausted)."""


EmptyResponse = EmptyResponseError


class AuthenticationError(ProviderError):
    """Raised when provider credentials are missing, invalid, or unauthorized."""


class BadRequestError(ProviderError):
    """Raised when a provider rejects the request without a retryable failure."""


class ModelNotFoundError(BadRequestError):
    """Raised when the requested model is unavailable or decommissioned."""


class TransientProviderError(ProviderError):
    """Raised for timeouts, connection failures, and HTTP 5xx responses."""


class ProviderAttempt(TypedDict):
    """Diagnostic summary for a provider skipped or failed before service."""

    provider: str
    model: str
    outcome: str
    error_summary: str


def extract_retry_after(exc: Exception) -> float | None:
    """Extract a retry delay in seconds from headers or a structured error hint."""
    import re

    def parse_seconds(value: object, *, milliseconds: bool = False) -> float | None:
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(seconds) or seconds < 0:
            return None
        return seconds / 1000 if milliseconds else seconds

    response = getattr(exc, "response", None)
    if response is not None:
        headers = getattr(response, "headers", {}) or {}
        if isinstance(headers, Mapping):
            normalized_headers = {str(key).lower(): value for key, value in headers.items()}
            retry_after = parse_seconds(normalized_headers.get("retry-after"))
            if retry_after is not None:
                return retry_after
            retry_after_ms = parse_seconds(
                normalized_headers.get("retry-after-ms"),
                milliseconds=True,
            )
            if retry_after_ms is not None:
                return retry_after_ms
            raw_date = normalized_headers.get("retry-after")
            if isinstance(raw_date, str):
                try:
                    retry_date = parsedate_to_datetime(raw_date)
                    if retry_date.tzinfo is None:
                        retry_date = retry_date.replace(tzinfo=UTC)
                    return max(0.0, (retry_date - datetime.now(UTC)).total_seconds())
                except (TypeError, ValueError, OverflowError):
                    pass

    body = getattr(exc, "body", None)
    if isinstance(body, str):
        with suppress(json.JSONDecodeError):
            body = json.loads(body)

    def body_hint(value: object) -> float | None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                normalized_key = str(key).lower().replace("-", "_")
                if normalized_key in ("retry_after", "retry_after_seconds"):
                    parsed = parse_seconds(item)
                    if parsed is not None:
                        return parsed
                elif normalized_key == "retry_after_ms":
                    parsed = parse_seconds(item, milliseconds=True)
                    if parsed is not None:
                        return parsed
                nested = body_hint(item)
                if nested is not None:
                    return nested
        elif isinstance(value, list):
            for item in value:
                nested = body_hint(item)
                if nested is not None:
                    return nested
        return None

    parsed_body = body_hint(body)
    if parsed_body is not None:
        return parsed_body

    msg = f"{body!s} {exc}"
    millisecond_match = re.search(
        r"retry[-_ ]after[-_ ]ms\s*[:=]\s*(\d+(?:\.\d+)?)",
        msg,
        re.IGNORECASE,
    )
    if millisecond_match:
        return parse_seconds(millisecond_match.group(1), milliseconds=True)
    match = re.search(
        r"(?:try again in|retry[-_ ]after(?:[\"']?\s*[:=]\s*|\s+)|"
        r"retry after\s+|wait\s+)(\d+(?:\.\d+)?)\s*"
        r"(ms|milliseconds|s|sec|seconds)?",
        msg,
        re.IGNORECASE,
    )
    if match:
        value = float(match.group(1))
        unit = (match.group(2) or "").lower()
        return parse_seconds(value, milliseconds=unit in ("ms", "milliseconds"))
    return None


def _classified_exception(
    provider: str,
    exc: Exception,
    status: int | None,
) -> ProviderError | None:
    """Map recognized provider failures to gateway resilience categories."""
    message = str(exc)
    code = getattr(exc, "code", "")
    error_type = getattr(exc, "type", "")
    details = f"{code} {error_type} {message}".lower()

    if (
        status == 402
        or "payment_required" in details
        or "payment required" in details
        or "insufficient_quota" in details
        or "credit balance" in details
        or "out of credits" in details
        or (status in (400, 403) and ("quota" in details or "billing" in details))
        or ("billing" in details and any(w in details for w in ("disabled", "required", "visit", "tab")))
    ):
        return BillingOrQuota(provider, status or 402, message)
    if (
        status == 429
        or "resource_exhausted" in details
    ):
        retry_after = extract_retry_after(exc)
        return RateLimitError(provider, status or 429, message, retry_after=retry_after)
    if status in (401, 403):
        return AuthenticationError(provider, status, message)
    if status == 404 or any(
        marker in details
        for marker in ("model_not_found", "model_decommissioned", "model does not exist")
    ):
        return ModelNotFoundError(provider, status or 404, message)
    if status == 400:
        return BadRequestError(provider, status, message)
    if status is not None and status >= 500:
        return TransientProviderError(provider, status, message)
    exception_name = type(exc).__name__.lower()
    if isinstance(exc, (TimeoutError, ConnectionError)) or any(
        marker in exception_name for marker in ("timeout", "connection", "connect")
    ):
        return TransientProviderError(provider, status, message)
    return None


class BaseProvider(ABC):
    """Abstract base for all LLM provider adapters."""

    name: str = "base"

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 800,
        response_format: dict[str, Any] | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
    ) -> LLMResult:
        """Send messages to the provider and return an LLMResult.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in the response.
            response_format: Optional dict for requesting structured output
                             (e.g. {"type": "json_object"}).
            model: Optional per-call model override.
            reasoning_effort: Optional reasoning effort ('low' | 'medium' | 'high').

        Raises:
            RateLimitError: On HTTP 429.
            ProviderError: On other HTTP or transport errors.
        """


# ── Gemini ────────────────────────────────────────────────────────────────────


class GeminiProvider(BaseProvider):
    """Adapter for Google Gemini via the google-genai SDK."""

    name = "gemini"

    def __init__(self, api_key: str, model: str = "gemini-3.1-flash-lite") -> None:
        if not api_key:
            raise ValueError("GeminiProvider requires a non-empty api_key.")
        self._model = model
        self._api_key = api_key
        # Import here so missing package doesn't break non-Gemini paths.
        try:
            from google import genai  # type: ignore[import-untyped]
            self._client = genai.Client(api_key=api_key)
        except ImportError as exc:
            raise ImportError(
                "google-genai is not installed. Run: pip install google-genai"
            ) from exc

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 800,
        response_format: dict[str, Any] | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
    ) -> LLMResult:
        """Call Gemini generate_content and return LLMResult."""
        from google.genai import types  # type: ignore[import-untyped]

        # Convert OpenAI-style messages to a single prompt string for Gemini.
        # System messages are prepended; user/assistant alternate.
        prompt_parts: list[str] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                prompt_parts.insert(0, f"[System]: {content}")
            elif role == "assistant":
                prompt_parts.append(f"[Assistant]: {content}")
            else:
                prompt_parts.append(f"[User]: {content}")
        prompt = "\n".join(prompt_parts)

        if response_format and response_format.get("type") == "json_object":
            prompt += "\n\nRespond with valid JSON only. No markdown, no explanation."

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        t0 = time.perf_counter()
        effective_model = model or self._model
        try:
            response = self._client.models.generate_content(
                model=effective_model,
                contents=prompt,
                config=config,
            )
        except Exception as exc:
            status = self._extract_status(exc)
            mapped = _classified_exception(self.name, exc, status)
            if mapped is None:
                raise
            raise mapped from exc

        latency_ms = (time.perf_counter() - t0) * 1000
        text = response.text or ""
        return LLMResult(
            text=text,
            provider=self.name,
            model=effective_model,
            latency_ms=latency_ms,
        )

    @staticmethod
    def _extract_status(exc: Exception) -> int | None:
        """Try to parse an HTTP status code from an exception message."""
        import re
        for candidate in (
            getattr(exc, "status_code", None),
            getattr(exc, "code", None),
            getattr(getattr(exc, "response", None), "status_code", None),
        ):
            if isinstance(candidate, int) and 400 <= candidate <= 599:
                return candidate
        match = re.search(
            r"\b([45]\d{2})\s+"
            r"(?:INVALID_ARGUMENT|NOT_FOUND|UNAUTHENTICATED|PERMISSION_DENIED|"
            r"RESOURCE_EXHAUSTED|INTERNAL|UNAVAILABLE|DEADLINE_EXCEEDED|PAYMENT_REQUIRED)\b",
            str(exc),
            re.IGNORECASE,
        )
        return int(match.group(1)) if match else None


# ── OpenAI-compatible ─────────────────────────────────────────────────────────


class OpenAICompatProvider(BaseProvider):
    """Adapter for any OpenAI-compatible provider (Groq, OpenRouter, Mistral, Cerebras)."""

    def __init__(self, *, name: str, base_url: str, api_key: str, model: str) -> None:
        if not api_key:
            raise ValueError(f"{name} provider requires a non-empty api_key.")
        self.name = name
        self._model = model
        try:
            from openai import OpenAI  # type: ignore[import-untyped]
            self._client = OpenAI(api_key=api_key, base_url=base_url)
        except ImportError as exc:
            raise ImportError(
                "openai package is not installed. Run: pip install openai"
            ) from exc

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 800,
        response_format: dict[str, Any] | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
    ) -> LLMResult:
        """Call the OpenAI-compatible endpoint and return LLMResult."""
        import re

        import openai

        effective_model = model or self._model
        is_gpt_oss = effective_model.startswith("openai/gpt-oss")

        kwargs: dict[str, Any] = {
            "model": effective_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format
        if is_gpt_oss and reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort

        t0 = time.perf_counter()

        def _execute_create(params: dict[str, Any]) -> Any:
            try:
                return self._client.chat.completions.create(**params)
            except openai.RateLimitError as exc:
                retry_after = extract_retry_after(exc)
                raise RateLimitError(self.name, 429, str(exc), retry_after=retry_after) from exc
            except openai.APITimeoutError as exc:
                raise TransientProviderError(self.name, None, str(exc)) from exc
            except openai.APIConnectionError as exc:
                raise TransientProviderError(self.name, None, str(exc)) from exc
            except openai.APIStatusError as exc:
                mapped = _classified_exception(self.name, exc, exc.status_code)
                if mapped is None:
                    raise
                raise mapped from exc

        try:
            response = _execute_create(kwargs)
        except Exception as exc:
            # Retry a rejected reasoning parameter once without that optional argument.
            if is_gpt_oss and "reasoning_effort" in kwargs:
                err_str = str(exc).lower()
                status = getattr(
                    exc, "status", getattr(exc, "status_code", getattr(exc, "code", None))
                )
                rejected_parameter = "reasoning_effort" in err_str and any(
                    marker in err_str
                    for marker in (
                        "unrecognized",
                        "unsupported",
                        "not supported",
                        "unknown",
                        "unexpected",
                        "extra",
                        "invalid",
                    )
                )
                rejected_bad_request = status == 400 and (
                    "reasoning_effort" in err_str
                    and any(
                        marker in err_str
                        for marker in ("parameter", "argument", "field", "property")
                    )
                )
                if rejected_parameter or rejected_bad_request:
                    logger.warning(
                        "Provider '%s' rejected reasoning_effort parameter; retrying once without it.",
                        self.name,
                    )
                    kwargs.pop("reasoning_effort", None)
                    response = _execute_create(kwargs)
                else:
                    raise
            else:
                raise

        def _extract_content(res: Any) -> tuple[str, str | None]:
            if not getattr(res, "choices", None):
                return "", None
            choice_obj = res.choices[0]
            raw = choice_obj.message.content or ""
            stripped = re.sub(
                r"(?is)<(?:think|reasoning)\b[^>]*>.*?</(?:think|reasoning)>",
                "",
                raw,
            ).strip()
            return stripped, getattr(choice_obj, "finish_reason", None)

        text, finish_reason = _extract_content(response)

        # If the reply is empty with finish_reason "length", retry once with a larger max_tokens
        # and then raise a clear EmptyResponse error that falls through to the next provider.
        if is_gpt_oss and not text and finish_reason == "length":
            larger_max_tokens = max(max_tokens * 2, max_tokens + 1000)
            logger.warning(
                "Provider '%s' returned empty content with finish_reason='length' for model '%s'. "
                "Retrying once with larger max_tokens=%d.",
                self.name,
                effective_model,
                larger_max_tokens,
            )
            retry_kwargs = dict(kwargs)
            retry_kwargs["max_tokens"] = larger_max_tokens
            response2 = _execute_create(retry_kwargs)
            text2, _ = _extract_content(response2)
            if not text2:
                raise EmptyResponseError(
                    self.name,
                    None,
                    f"Empty response from {effective_model} with finish_reason='length' "
                    f"after retry with max_tokens={larger_max_tokens}.",
                )
            response = response2
            text = text2

        latency_ms = (time.perf_counter() - t0) * 1000
        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        return LLMResult(
            text=text,
            provider=self.name,
            model=effective_model,
            latency_ms=latency_ms,
            usage=usage,
        )


# ── Mock ──────────────────────────────────────────────────────────────────────

_MOCK_OUTPUTS: dict[str, str] = {
    TaskKind.converse.value: (
        '{"reply": "I hear you. That sounds really difficult. '
        'Can you tell me more about what has been weighing on you?"}'
    ),
    TaskKind.classify.value: (
        '{"label": "sadness", "score": 0.82}'
    ),
    TaskKind.judge.value: (
        '{"passed": true, "issues": [], "grounded": true, '
        '"role_fidelity": true, "no_diagnosis": true}'
    ),
    TaskKind.extract.value: (
        '{"entities": [], "themes": ["loneliness", "work stress"]}'
    ),
}


class MockProvider(BaseProvider):
    """Deterministic mock provider for tests — never calls a real API.

    Returns scripted outputs keyed by TaskKind.  Optionally raise a
    specific exception to test resilience paths.
    """

    name = "mock"

    def __init__(
        self,
        *,
        task_kind: str = TaskKind.converse.value,
        raise_on_call: Exception | None = None,
        fixed_output: str | None = None,
    ) -> None:
        """
        Args:
            task_kind: Which scripted output to return.
            raise_on_call: If set, raise this exception instead of returning.
            fixed_output: Override the scripted output with a custom string.
        """
        self._task_kind = task_kind
        self._raise = raise_on_call
        self._fixed = fixed_output

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 800,
        response_format: dict[str, Any] | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
    ) -> LLMResult:
        """Return scripted output or raise the configured exception."""
        if self._raise is not None:
            raise self._raise

        text = self._fixed or _MOCK_OUTPUTS.get(
            self._task_kind, _MOCK_OUTPUTS[TaskKind.converse.value]
        )
        return LLMResult(
            text=text,
            provider="mock",
            model=model or "mock-model",
            latency_ms=1.0,
        )
