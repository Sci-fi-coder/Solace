"""LLM provider adapters for Solace.

Adapters:
- GeminiProvider   — google-genai SDK
- OpenAICompatProvider — openai SDK with custom base_url (Groq, OpenRouter, Mistral, Cerebras)
- MockProvider     — deterministic scripted outputs for tests
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from app.contracts import TaskKind

logger = logging.getLogger(__name__)


# ── Base class ────────────────────────────────────────────────────────────────


class LLMResult:
    """Output of a single LLM call."""

    __slots__ = ("text", "parsed", "provider", "model", "latency_ms", "usage")

    def __init__(
        self,
        text: str,
        provider: str,
        model: str,
        latency_ms: float,
        parsed: Any = None,
        usage: dict[str, int] | None = None,
    ) -> None:
        self.text = text
        self.parsed = parsed
        self.provider = provider
        self.model = model
        self.latency_ms = latency_ms
        self.usage = usage

    def __repr__(self) -> str:
        return (
            f"LLMResult(provider={self.provider!r}, model={self.model!r}, "
            f"latency_ms={self.latency_ms:.0f})"
        )


class ProviderError(Exception):
    """Raised when a provider call fails after all retries."""

    def __init__(self, provider: str, status: int | None, message: str) -> None:
        super().__init__(f"[{provider}] HTTP {status}: {message}")
        self.provider = provider
        self.status = status


class RateLimitError(ProviderError):
    """Raised on HTTP 429 — triggers cooldown in the gateway."""


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
    ) -> LLMResult:
        """Send messages to the provider and return an LLMResult.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in the response.
            response_format: Optional dict for requesting structured output
                             (e.g. {"type": "json_object"}).

        Raises:
            RateLimitError: On HTTP 429.
            ProviderError: On other HTTP or transport errors.
        """


# ── Gemini ────────────────────────────────────────────────────────────────────


class GeminiProvider(BaseProvider):
    """Adapter for Google Gemini via the google-genai SDK."""

    name = "gemini"

    def __init__(self, api_key: str, model: str = "gemini-1.5-flash") -> None:
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
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=config,
            )
        except Exception as exc:
            # Inspect exception type/message for rate limit signals.
            exc_str = str(exc)
            status = self._extract_status(exc_str)
            if status == 429 or "quota" in exc_str.lower() or "rate" in exc_str.lower():
                raise RateLimitError(self.name, 429, exc_str) from exc
            raise ProviderError(self.name, status, exc_str) from exc

        latency_ms = (time.perf_counter() - t0) * 1000
        text = response.text or ""
        return LLMResult(
            text=text,
            provider=self.name,
            model=self._model,
            latency_ms=latency_ms,
        )

    @staticmethod
    def _extract_status(msg: str) -> int | None:
        """Try to parse an HTTP status code from an exception message."""
        import re
        m = re.search(r"\b([45]\d{2})\b", msg)
        return int(m.group(1)) if m else None


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
    ) -> LLMResult:
        """Call the OpenAI-compatible endpoint and return LLMResult."""
        import openai

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format

        t0 = time.perf_counter()
        try:
            response = self._client.chat.completions.create(**kwargs)
        except openai.RateLimitError as exc:
            raise RateLimitError(self.name, 429, str(exc)) from exc
        except openai.APIStatusError as exc:
            raise ProviderError(self.name, exc.status_code, str(exc)) from exc
        except Exception as exc:
            raise ProviderError(self.name, None, str(exc)) from exc

        latency_ms = (time.perf_counter() - t0) * 1000
        text = response.choices[0].message.content or ""
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
            model=self._model,
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
            model="mock-model",
            latency_ms=1.0,
        )
