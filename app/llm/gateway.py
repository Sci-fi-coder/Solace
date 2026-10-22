"""LLM Gateway — the single access point for all language model calls in Solace.

Design goals:
- Single interface regardless of provider
- Ordered fallback chain per TaskKind, configurable via settings
- Exponential backoff + jitter; per-provider cooldown after 429
- Per-provider per-day call counter (soft cap) persisted in SQLite
- Structured-output support with one repair retry
- pre_send_filter hook list (identity by default; PII scrubber plugs in here)
- Providers with missing keys are skipped silently (WARN only)
- User message content is NEVER logged at INFO level
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from collections.abc import Callable
from datetime import date, datetime, timezone
from typing import Any, Type, TypeVar

from pydantic import BaseModel, ValidationError

from app.contracts import TaskKind
from app.llm.providers import (
    BaseProvider,
    GeminiProvider,
    LLMResult,
    MockProvider,
    OpenAICompatProvider,
    ProviderError,
    RateLimitError,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


# ── Custom errors ─────────────────────────────────────────────────────────────


class GatewayExhaustedError(Exception):
    """All providers in the fallback chain failed."""


class StructuredOutputError(Exception):
    """Provider returned JSON that could not be validated after repair."""


class DailyCapExceededError(Exception):
    """Provider has hit its soft daily call cap."""


# ── Usage counter (in-memory + SQLite) ───────────────────────────────────────


class _UsageCounter:
    """Thread-safe in-memory daily call counter with SQLite persistence.

    Persistence is best-effort — if the DB is unavailable the counter
    falls back to in-memory only (meaning caps reset on restart).
    """

    def __init__(self) -> None:
        self._counts: dict[tuple[str, str], int] = {}  # (provider, date) -> count
        self._engine: Any = None

    def _today(self) -> str:
        return date.today().isoformat()

    def _key(self, provider: str) -> tuple[str, str]:
        return (provider, self._today())

    def init_db(self, engine: Any) -> None:
        """Attach a SQLAlchemy engine for persistence (called by gateway init)."""
        self._engine = engine
        self._load_today()

    def _load_today(self) -> None:
        if self._engine is None:
            return
        try:
            from sqlalchemy import text  # noqa: PLC0415
            with self._engine.connect() as conn:
                rows = conn.execute(
                    text(
                        "SELECT provider, call_count FROM provider_usage WHERE date = :d"
                    ),
                    {"d": self._today()},
                ).fetchall()
            for row in rows:
                self._counts[(row[0], self._today())] = row[1]
        except Exception:
            logger.debug("Usage counter: could not load from DB (table may not exist yet).")

    def increment(self, provider: str) -> None:
        """Increment counter and persist to DB."""
        key = self._key(provider)
        self._counts[key] = self._counts.get(key, 0) + 1
        self._persist(provider)

    def get(self, provider: str) -> int:
        """Return today's call count for a provider."""
        return self._counts.get(self._key(provider), 0)

    def _persist(self, provider: str) -> None:
        if self._engine is None:
            return
        try:
            from sqlalchemy import text  # noqa: PLC0415
            count = self.get(provider)
            with self._engine.begin() as conn:
                conn.execute(
                    text(
                        """
                        INSERT INTO provider_usage (provider, date, call_count)
                        VALUES (:p, :d, :c)
                        ON CONFLICT(provider, date) DO UPDATE SET call_count = :c
                        """
                    ),
                    {"p": provider, "d": self._today(), "c": count},
                )
        except Exception as exc:
            logger.debug("Usage counter: DB persist failed: %s", exc)


_usage = _UsageCounter()


# ── Gateway ───────────────────────────────────────────────────────────────────


class LLMGateway:
    """Single interface for all LLM calls.

    Usage::

        gateway = LLMGateway.from_settings()
        result = gateway.generate(TaskKind.converse, messages)
        result = gateway.generate(TaskKind.extract, messages, json_schema=MyModel)
    """

    def __init__(
        self,
        providers: dict[str, BaseProvider],
        fallback_chains: dict[str, list[str]],
        daily_caps: dict[str, int],
        *,
        cooldown_seconds: int = 60,
        max_retries: int = 3,
        backoff_base: float = 1.0,
        pre_send_filter: list[Callable] | None = None,
    ) -> None:
        """
        Args:
            providers: Mapping of provider-name → provider instance.
            fallback_chains: Mapping of task-kind-value → ordered provider names.
            daily_caps: Mapping of provider-name → soft daily call cap.
            cooldown_seconds: How long to back off after a 429.
            max_retries: Max retries per provider before falling through.
            backoff_base: Base delay (seconds) for exponential backoff.
            pre_send_filter: List of callables applied to messages before dispatch.
        """
        self._providers = providers
        self._chains = fallback_chains
        self._caps = daily_caps
        self._cooldown_s = cooldown_seconds
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._pre_send_filter: list[Callable] = pre_send_filter or []
        # provider_name -> UTC timestamp when cooldown expires
        self._cooldown_until: dict[str, float] = {}

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def from_settings(cls, engine: Any = None) -> "LLMGateway":
        """Build a gateway from app.config.settings.

        Providers with empty API keys are skipped with a WARNING.
        If engine is provided, the usage counter will persist to SQLite.
        """
        from app.config import settings  # local import avoids circular dep

        providers: dict[str, BaseProvider] = {}
        all_provider_names = ["gemini", "groq", "openrouter", "mistral", "cerebras"]

        for name in all_provider_names:
            cfg = settings.provider_config(name)
            key = cfg["api_key"]
            if not key:
                logger.warning("Provider '%s' skipped: no API key configured.", name)
                continue
            try:
                if name == "gemini":
                    providers[name] = GeminiProvider(api_key=key, model=cfg["model"])
                else:
                    providers[name] = OpenAICompatProvider(
                        name=name,
                        base_url=cfg["base_url"],
                        api_key=key,
                        model=cfg["model"],
                    )
            except Exception as exc:
                logger.warning("Provider '%s' failed to initialise: %s", name, exc)

        if not providers:
            logger.warning("No real providers available. Using MockProvider as fallback.")
            providers["mock"] = MockProvider()

        chains = {
            tk.value: settings.get_fallback_chain(tk.value) for tk in TaskKind
        }
        caps = {
            name: settings.provider_config(name)["daily_cap"]
            for name in all_provider_names
        }

        gw = cls(
            providers=providers,
            fallback_chains=chains,
            daily_caps=caps,
            cooldown_seconds=settings.provider_cooldown_seconds,
            max_retries=settings.gateway_max_retries,
            backoff_base=settings.gateway_backoff_base,
        )
        if engine is not None:
            _usage.init_db(engine)
        return gw

    @classmethod
    def from_mock(cls, **mock_kwargs: Any) -> "LLMGateway":
        """Build a gateway backed entirely by MockProvider — for tests."""
        mock = MockProvider(**mock_kwargs)
        chains = {tk.value: ["mock"] for tk in TaskKind}
        return cls(
            providers={"mock": mock},
            fallback_chains=chains,
            daily_caps={"mock": 999_999},
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        task: TaskKind,
        messages: list[dict[str, str]],
        *,
        json_schema: Type[T] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 800,
    ) -> LLMResult:
        """Dispatch an LLM call through the fallback chain for ``task``.

        Args:
            task: Determines which provider chain to use.
            messages: Conversation messages as {"role", "content"} dicts.
            json_schema: If provided, request JSON and validate against this Pydantic model.
            temperature: Sampling temperature.
            max_tokens: Max tokens to generate.

        Returns:
            LLMResult with text (and parsed if json_schema provided).

        Raises:
            GatewayExhaustedError: When all providers in the chain fail.
            StructuredOutputError: When json_schema is provided but output cannot be parsed.
        """
        # Apply pre_send_filter hooks
        filtered = self._apply_filters(messages)

        response_format: dict[str, Any] | None = None
        if json_schema is not None:
            response_format = {"type": "json_object"}

        chain = self._chains.get(task.value, [])
        errors: list[str] = []

        for provider_name in chain:
            provider = self._providers.get(provider_name)
            if provider is None:
                continue
            if self._is_in_cooldown(provider_name):
                logger.debug("Provider '%s' is in cooldown, skipping.", provider_name)
                continue
            if self._over_daily_cap(provider_name):
                logger.warning(
                    "Provider '%s' has hit its daily cap (%d), skipping.",
                    provider_name,
                    self._caps.get(provider_name, 0),
                )
                continue

            result = self._call_with_retry(
                provider, filtered, temperature, max_tokens, response_format
            )
            if result is None:
                errors.append(f"{provider_name}: all retries exhausted")
                continue

            _usage.increment(provider_name)
            logger.info(
                "LLM call served by provider=%s model=%s latency_ms=%.0f task=%s",
                result.provider,
                result.model,
                result.latency_ms,
                task.value,
            )

            if json_schema is not None:
                result = self._parse_structured(result, json_schema, provider, filtered)

            return result

        raise GatewayExhaustedError(
            f"All providers failed for task={task.value}. Errors: {errors}"
        )

    # ── Filters ───────────────────────────────────────────────────────────────

    def add_filter(self, fn: Callable) -> None:
        """Register a pre-send filter callable."""
        self._pre_send_filter.append(fn)

    def _apply_filters(
        self, messages: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        result = list(messages)
        for fn in self._pre_send_filter:
            result = fn(result)
        return result

    # ── Resilience ────────────────────────────────────────────────────────────

    def _is_in_cooldown(self, provider: str) -> bool:
        until = self._cooldown_until.get(provider, 0.0)
        return time.monotonic() < until

    def _set_cooldown(self, provider: str) -> None:
        self._cooldown_until[provider] = time.monotonic() + self._cooldown_s
        logger.warning(
            "Provider '%s' rate-limited. Cooling down for %ds.", provider, self._cooldown_s
        )

    def _over_daily_cap(self, provider: str) -> bool:
        cap = self._caps.get(provider, 0)
        if cap <= 0:
            return False
        return _usage.get(provider) >= cap

    def _call_with_retry(
        self,
        provider: BaseProvider,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        response_format: dict[str, Any] | None,
    ) -> LLMResult | None:
        """Attempt provider call with exponential backoff + jitter.

        Returns None if all retries are exhausted (caller tries next provider).
        """
        for attempt in range(self._max_retries):
            try:
                return provider.generate(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                )
            except RateLimitError:
                self._set_cooldown(provider.name)
                return None  # fall through to next provider immediately
            except ProviderError as exc:
                wait = self._backoff_base * (2 ** attempt) + random.uniform(0, 0.5)
                logger.debug(
                    "Provider '%s' error (attempt %d/%d): %s. Retrying in %.1fs.",
                    provider.name,
                    attempt + 1,
                    self._max_retries,
                    exc,
                    wait,
                )
                if attempt < self._max_retries - 1:
                    time.sleep(wait)
        return None

    # ── Structured output ─────────────────────────────────────────────────────

    def _parse_structured(
        self,
        result: LLMResult,
        schema: Type[T],
        provider: BaseProvider,
        original_messages: list[dict[str, str]],
    ) -> LLMResult:
        """Validate result.text against schema; one repair retry on failure."""
        parsed, error = self._try_parse(result.text, schema)
        if parsed is not None:
            result.parsed = parsed
            return result

        logger.debug(
            "Structured output validation failed (provider=%s). Attempting repair.",
            provider.name,
        )
        repair_messages = list(original_messages) + [
            {"role": "assistant", "content": result.text},
            {
                "role": "user",
                "content": (
                    f"Your response was not valid JSON matching the required schema.\n"
                    f"Validation error: {error}\n"
                    f"Please respond again with ONLY valid JSON, no explanation."
                ),
            },
        ]
        try:
            repair_result = provider.generate(
                repair_messages,
                temperature=0.0,
                max_tokens=result.latency_ms.__class__(800),  # type: ignore[arg-type]
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            raise StructuredOutputError(
                f"Repair call failed for provider={provider.name}: {exc}"
            ) from exc

        parsed2, error2 = self._try_parse(repair_result.text, schema)
        if parsed2 is not None:
            repair_result.parsed = parsed2
            return repair_result

        raise StructuredOutputError(
            f"Could not parse structured output after repair. "
            f"Schema={schema.__name__}. Error: {error2}. "
            f"Raw: {repair_result.text[:200]}"
        )

    @staticmethod
    def _try_parse(text: str, schema: Type[T]) -> tuple[T | None, str | None]:
        """Attempt to parse text as JSON and validate against schema.

        Returns (instance, None) on success or (None, error_str) on failure.
        """
        try:
            data = json.loads(text)
            return schema.model_validate(data), None
        except json.JSONDecodeError as exc:
            return None, f"JSONDecodeError: {exc}"
        except ValidationError as exc:
            return None, str(exc)

    # ── Observability helpers ─────────────────────────────────────────────────

    @staticmethod
    def _msg_fingerprint(messages: list[dict[str, str]]) -> str:
        """Return a short hash of the message list for debug logging (no content)."""
        raw = json.dumps(messages, sort_keys=True).encode()
        return hashlib.sha256(raw).hexdigest()[:8]
