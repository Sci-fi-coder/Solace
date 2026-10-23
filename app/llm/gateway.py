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
from datetime import UTC, datetime
from threading import Lock
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from app.config import redact_secrets
from app.contracts import TaskKind
from app.llm.providers import (
    AuthenticationError,
    BadRequestError,
    BaseProvider,
    GeminiProvider,
    LLMResult,
    MockProvider,
    ModelNotFoundError,
    OpenAICompatProvider,
    ProviderAttempt,
    ProviderError,
    RateLimitError,
    TransientProviderError,
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
    """Thread-safe daily call counter with optional SQLite persistence."""

    def __init__(self) -> None:
        self._counts: dict[tuple[str, str], int] = {}  # (provider, date) -> count
        self._engine: Any = None
        self._lock = Lock()

    def _today(self) -> str:
        return datetime.now(UTC).date().isoformat()

    def _key(self, provider: str) -> tuple[str, str]:
        return (provider, self._today())

    def init_db(self, engine: Any) -> None:
        """Attach a SQLAlchemy engine for persistence (called by gateway init)."""
        self._engine = engine
        self._load_today()

    def _load_today(self) -> None:
        if self._engine is None:
            return
        from sqlalchemy import text  # noqa: PLC0415

        today = self._today()
        with self._engine.connect() as conn:
            rows = conn.execute(
                text("SELECT provider, call_count FROM provider_usage WHERE date = :d"),
                {"d": today},
            ).fetchall()
        with self._lock:
            self._counts = {
                key: value for key, value in self._counts.items() if key[1] != today
            }
            for row in rows:
                self._counts[(row[0], today)] = row[1]

    def claim(self, provider: str, cap: int | None) -> bool:
        """Atomically reserve one provider call without exceeding its daily cap."""
        if cap is not None and cap <= 0:
            return False

        key = self._key(provider)
        if self._engine is None:
            with self._lock:
                count = self._counts.get(key, 0)
                if cap is not None and count >= cap:
                    return False
                self._counts[key] = count + 1
                return True

        from sqlalchemy import text  # noqa: PLC0415

        with self._lock, self._engine.begin() as conn:
            result = conn.execute(
                text(
                    """
                    INSERT INTO provider_usage (provider, date, call_count)
                    VALUES (:p, :d, 1)
                    ON CONFLICT(provider, date) DO UPDATE
                    SET call_count = provider_usage.call_count + 1
                    WHERE :cap IS NULL OR provider_usage.call_count < :cap
                    """
                ),
                {"p": provider, "d": key[1], "cap": cap},
            )
            if result.rowcount == 0:
                row = conn.execute(
                    text(
                        "SELECT call_count FROM provider_usage "
                        "WHERE provider = :p AND date = :d"
                    ),
                    {"p": provider, "d": key[1]},
                ).one()
                self._counts[key] = row[0]
                return False

            count = conn.execute(
                text(
                    "SELECT call_count FROM provider_usage "
                    "WHERE provider = :p AND date = :d"
                ),
                {"p": provider, "d": key[1]},
            ).scalar_one()
            self._counts[key] = count
            return True

    def get(self, provider: str) -> int:
        """Return today's call count for a provider."""
        with self._lock:
            return self._counts.get(self._key(provider), 0)


_usage = _UsageCounter()
_PROCESS_DISABLED_PROVIDERS: dict[str, str] = {}
_PROCESS_DISABLED_LOCK = Lock()


# ── Gateway ───────────────────────────────────────────────────────────────────


class LLMGateway:
    """Single interface for all LLM calls.

    Usage::

        gateway = LLMGateway.from_settings(engine=database_engine)
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
        usage_counter: _UsageCounter | None = None,
        provider_models: dict[str, str] | None = None,
        skipped_providers: dict[str, str] | None = None,
        secrets: list[str] | None = None,
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
            usage_counter: Optional daily usage counter; defaults to an isolated counter.
        """
        self._providers = providers
        self._chains = fallback_chains
        self._caps = daily_caps
        self._cooldown_s = cooldown_seconds
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._pre_send_filter: list[Callable] = pre_send_filter or []
        self._usage_counter = usage_counter or _UsageCounter()
        self._provider_models = provider_models or {}
        self._skipped_providers = skipped_providers or {}
        self._secrets = [secret for secret in (secrets or []) if secret]
        # provider_name -> UTC timestamp when cooldown expires
        self._cooldown_until: dict[str, float] = {}

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def from_settings(cls, engine: Any) -> LLMGateway:
        """Build a gateway from app.config.settings.

        Providers with empty API keys are skipped with a WARNING.
        The supplied engine persists provider usage counters in SQLite.
        """
        from app.config import settings  # local import avoids circular dep

        providers: dict[str, BaseProvider] = {}
        all_provider_names = ["gemini", "groq", "openrouter", "mistral", "cerebras"]
        provider_models: dict[str, str] = {}
        skipped_providers: dict[str, str] = {}
        secrets: list[str] = []

        for name in all_provider_names:
            cfg = settings.provider_config(name)
            key = cfg["api_key"]
            provider_models[name] = cfg["model"]
            if not key:
                skipped_providers[name] = "no API key configured"
                logger.warning("Provider '%s' skipped: no API key configured.", name)
                continue
            secrets.append(key)
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
                skipped_providers[name] = f"initialization failed ({type(exc).__name__})"
                logger.warning(
                    "Provider '%s' failed to initialise (%s).",
                    name,
                    type(exc).__name__,
                )

        has_real_provider = bool(providers)
        if not has_real_provider:
            logger.warning("No real providers available. Using MockProvider as fallback.")
            providers["mock"] = MockProvider()

        chains = {
            tk.value: settings.get_fallback_chain(tk.value) for tk in TaskKind
        }
        if not has_real_provider:
            for chain in chains.values():
                chain.append("mock")
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
            provider_models=provider_models,
            skipped_providers=skipped_providers,
            secrets=secrets,
        )
        _usage.init_db(engine)
        gw._usage_counter = _usage
        return gw

    @classmethod
    def from_mock(cls, **mock_kwargs: Any) -> LLMGateway:
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
        json_schema: type[T] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 800,
        provider: str | None = None,
    ) -> LLMResult:
        """Dispatch an LLM call through the fallback chain for ``task``.

        Args:
            task: Determines which provider chain to use.
            messages: Conversation messages as {"role", "content"} dicts.
            json_schema: If provided, request JSON and validate against this Pydantic model.
            temperature: Sampling temperature.
            max_tokens: Max tokens to generate.
            provider: Optional provider name to call instead of the configured chain.

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

        chain = [provider] if provider else self._chains.get(task.value, [])
        attempts: list[ProviderAttempt] = []

        for chain_item in chain:
            provider_name, separator, model_override = chain_item.partition("/")
            provider_name = provider_name.strip().lower()
            model = (
                model_override.strip()
                if separator and model_override.strip()
                else self._provider_models.get(provider_name, "")
            )
            provider = self._providers.get(provider_name)
            if provider is None:
                reason = self._skipped_providers.get(provider_name, "provider unavailable")
                attempts.append(self._skip_attempt(provider_name, model, reason))
                continue
            with _PROCESS_DISABLED_LOCK:
                disabled_reason = _PROCESS_DISABLED_PROVIDERS.get(provider_name)
            if disabled_reason is not None:
                attempts.append(
                    self._skip_attempt(provider_name, model, disabled_reason)
                )
                continue
            if self._is_in_cooldown(provider_name):
                attempts.append(self._skip_attempt(provider_name, model, "cooldown"))
                continue
            result, attempt = self._call_with_retry(
                provider_name,
                provider,
                filtered,
                temperature,
                max_tokens,
                response_format,
                model,
            )
            if result is None:
                attempts.append(attempt)
                continue

            result.provider = provider_name
            result.attempts = attempts
            if attempts:
                logger.debug(
                    "Provider fallback attempts before serving: %s -> served by %s.",
                    self._format_attempts(attempts),
                    provider_name,
                )
            logger.info(
                "LLM call served by provider=%s model=%s latency_ms=%.0f task=%s",
                result.provider,
                result.model,
                result.latency_ms,
                task.value,
            )

            if json_schema is not None:
                result = self._parse_structured(
                    result, json_schema, provider_name, provider, messages
                )

            return result

        errors = [
            f"{attempt['provider']}: {attempt['error_summary']}"
            for attempt in attempts
        ]
        raise GatewayExhaustedError(
            f"All providers failed for task={task.value}. "
            f"Attempts: {'; '.join(errors) if errors else 'no providers configured'}"
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

    def _call_with_retry(
        self,
        provider_name: str,
        provider: BaseProvider,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        response_format: dict[str, Any] | None,
        model: str,
    ) -> tuple[LLMResult | None, ProviderAttempt]:
        """Attempt provider call with exponential backoff + jitter.

        Return the result and diagnostic attempt, or a failed attempt for fallback.
        """
        max_retries = max(1, self._max_retries)
        last_error: ProviderError | None = None
        for attempt_index in range(max_retries):
            if not self._usage_counter.claim(provider_name, self._caps.get(provider_name)):
                summary = "skipped: daily cap reached"
                if last_error is not None:
                    summary += f" after {self._exception_summary(last_error)}"
                logger.warning(
                    "Provider '%s' has hit its daily cap (%d), skipping.",
                    provider_name,
                    self._caps.get(provider_name, 0),
                )
                return None, self._attempt(provider_name, model, "skipped", summary)
            try:
                kwargs: dict[str, Any] = {
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "response_format": response_format,
                }
                if model:
                    kwargs["model"] = model
                result = provider.generate(messages, **kwargs)
                return result, self._attempt(provider_name, model or result.model, "served", "")
            except RateLimitError as exc:
                self._set_cooldown(provider_name)
                summary = self._exception_summary(exc)
                return None, self._attempt(provider_name, model, "failed", summary)
            except AuthenticationError as exc:
                self._disable_provider(provider_name)
                summary = self._exception_summary(exc)
                logger.warning(
                    "Provider '%s' disabled after an authentication failure: %s",
                    provider_name,
                    summary,
                )
                return None, self._attempt(provider_name, model, "failed", summary)
            except (ModelNotFoundError, BadRequestError) as exc:
                summary = self._exception_summary(exc)
                return None, self._attempt(provider_name, model, "failed", summary)
            except ProviderError as exc:
                if exc.status in (401, 403):
                    self._disable_provider(provider_name)
                    summary = self._exception_summary(exc)
                    logger.warning(
                        "Provider '%s' disabled after an authentication failure: %s",
                        provider_name,
                        summary,
                    )
                    return None, self._attempt(provider_name, model, "failed", summary)
                if exc.status == 429:
                    self._set_cooldown(provider_name)
                    return None, self._attempt(
                        provider_name, model, "failed", self._exception_summary(exc)
                    )
                if exc.status in (400, 404):
                    return None, self._attempt(
                        provider_name, model, "failed", self._exception_summary(exc)
                    )
                if not isinstance(exc, TransientProviderError) and not (
                    exc.status is not None and exc.status >= 500
                ):
                    raise
                last_error = exc
                self._retry_sleep(provider_name, exc, attempt_index, max_retries)
                continue
            except (TimeoutError, ConnectionError) as exc:
                last_error = TransientProviderError(provider_name, None, str(exc))
                self._retry_sleep(provider_name, last_error, attempt_index, max_retries)
                continue

        if last_error is None:
            raise RuntimeError("Provider retry loop ended without a result or failure.")
        summary = self._exception_summary(last_error)
        summary = f"{summary} (after {max_retries} attempts)"
        return None, self._attempt(provider_name, model, "failed", summary)

    def _retry_sleep(
        self,
        provider_name: str,
        error: ProviderError,
        attempt_index: int,
        max_retries: int,
    ) -> None:
        """Log and wait before a transient provider retry."""
        if attempt_index >= max_retries - 1:
            return
        wait = self._backoff_base * (2 ** attempt_index) + random.uniform(0, 0.5)
        logger.debug(
            "Provider '%s' transient error (attempt %d/%d): %s. Retrying in %.1fs.",
            provider_name,
            attempt_index + 1,
            max_retries,
            self._exception_summary(error),
            wait,
        )
        time.sleep(wait)

    @staticmethod
    def _disable_provider(provider_name: str) -> None:
        """Disable an authentication-failing provider for this process."""
        with _PROCESS_DISABLED_LOCK:
            _PROCESS_DISABLED_PROVIDERS[provider_name] = (
                "authentication failure (provider disabled)"
            )

    def _exception_summary(self, error: Exception) -> str:
        """Format an exception type and message with configured keys redacted."""
        return redact_secrets(
            f"{type(error).__name__}: {error}",
            self._secrets,
        )

    @staticmethod
    def _attempt(
        provider: str,
        model: str,
        outcome: str,
        error_summary: str,
    ) -> ProviderAttempt:
        """Build a structured provider-attempt record."""
        return {
            "provider": provider,
            "model": model,
            "outcome": outcome,
            "error_summary": error_summary,
        }

    def _skip_attempt(self, provider: str, model: str, reason: str) -> ProviderAttempt:
        """Record a skipped provider with a safe, explicit reason."""
        summary = f"skipped: {reason}"
        logger.debug("Provider '%s' %s.", provider, summary)
        return self._attempt(provider, model, "skipped", summary)

    @staticmethod
    def _format_attempts(attempts: list[ProviderAttempt]) -> str:
        """Format attempts for key-free debug and smoke diagnostics."""
        return " -> ".join(
            f"{attempt['provider']}: {attempt['outcome']} "
            f"({attempt['error_summary']})"
            for attempt in attempts
        )

    # ── Structured output ─────────────────────────────────────────────────────

    def _parse_structured(
        self,
        result: LLMResult,
        schema: type[T],
        provider_name: str,
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
            provider_name,
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
        if not self._usage_counter.claim(provider_name, self._caps.get(provider_name)):
            raise StructuredOutputError(
                f"Provider '{provider_name}' reached its daily cap before JSON repair."
            )
        repair_messages = self._apply_filters(repair_messages)
        try:
            repair_result = provider.generate(
                repair_messages,
                temperature=0.0,
                max_tokens=800,
                response_format={"type": "json_object"},
            )
        except RateLimitError as exc:
            self._set_cooldown(provider_name)
            raise StructuredOutputError(
                f"Repair call was rate-limited for provider={provider_name}: {exc}"
            ) from exc
        except Exception as exc:
            raise StructuredOutputError(
                f"Repair call failed for provider={provider_name}: {exc}"
            ) from exc

        parsed2, error2 = self._try_parse(repair_result.text, schema)
        if parsed2 is not None:
            repair_result.parsed = parsed2
            repair_result.provider = provider_name
            repair_result.attempts = result.attempts
            return repair_result

        raise StructuredOutputError(
            f"Could not parse structured output after repair. "
            f"Schema={schema.__name__}. Error: {error2}. "
            f"Raw: {repair_result.text[:200]}"
        )

    @staticmethod
    def _try_parse(text: str, schema: type[T]) -> tuple[T | None, str | None]:
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
