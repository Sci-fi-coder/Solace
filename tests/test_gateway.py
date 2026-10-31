"""Tests for app/llm/gateway — resilience, filters, structured output, daily cap."""

from __future__ import annotations

import logging

import pytest
from pydantic import BaseModel

from app.contracts import TaskKind
from app.llm.gateway import (
    GatewayExhaustedError,
    LLMGateway,
    _UsageCounter,
)
from app.llm.providers import (
    AuthenticationError,
    BadRequestError,
    BillingOrQuota,
    LLMResult,
    MockProvider,
    ModelNotFoundError,
    ProviderError,
    RateLimitError,
    TransientProviderError,
)
from app.storage.repository import build_engine

# ── Helpers ───────────────────────────────────────────────────────────────────


def _simple_messages() -> list[dict[str, str]]:
    return [{"role": "user", "content": "Synthetic test message."}]


def _gateway_with(provider: MockProvider, *, caps: dict | None = None) -> LLMGateway:
    """Build a gateway backed by a single provider for testing."""
    chains = {tk.value: ["mock"] for tk in TaskKind}
    return LLMGateway(
        providers={"mock": provider},
        fallback_chains=chains,
        daily_caps=caps or {"mock": 999_999},
        cooldown_seconds=1,   # short for tests
        max_retries=2,
        backoff_base=0.01,    # minimal sleep in tests
    )


class _FailingProvider(MockProvider):
    """Count calls to a deterministic provider that raises a chosen exception."""

    def __init__(self, error: Exception, name: str) -> None:
        super().__init__()
        self.error = error
        self.name = name
        self.calls = 0

    def generate(self, messages, **kwargs):  # type: ignore[override]
        self.calls += 1
        raise self.error


# ── Basic success ─────────────────────────────────────────────────────────────


def test_gateway_returns_result_on_success() -> None:
    """Gateway returns LLMResult when the provider succeeds."""
    gw = _gateway_with(MockProvider(task_kind=TaskKind.converse.value))
    result = gw.generate(TaskKind.converse, _simple_messages())
    assert result.provider == "mock"
    assert isinstance(result.text, str)
    assert result.latency_ms >= 0


def test_provider_model_chain_override_is_used() -> None:
    """Fallback-chain entries may select a model as provider/model."""
    provider = MockProvider(fixed_output="model override")
    gateway = LLMGateway(
        providers={"mock": provider},
        fallback_chains={task.value: ["mock/custom-model"] for task in TaskKind},
        daily_caps={"mock": 100},
    )

    result = gateway.generate(TaskKind.converse, _simple_messages())

    assert result.model == "custom-model"


# ── 429 / fallback ────────────────────────────────────────────────────────────


def test_gateway_fallback_on_rate_limit() -> None:
    """When first provider raises 429, gateway falls through to next provider."""
    mock_429 = MockProvider(raise_on_call=RateLimitError("mock_429", 429, "too many requests"))
    mock_ok = MockProvider(fixed_output="Fallback reply")
    mock_ok.name = "mock_fallback"

    chains = {tk.value: ["mock_429", "mock_fallback"] for tk in TaskKind}
    gw = LLMGateway(
        providers={"mock_429": mock_429, "mock_fallback": mock_ok},
        fallback_chains=chains,
        daily_caps={"mock_429": 999_999, "mock_fallback": 999_999},
        cooldown_seconds=0,   # instant cooldown reset for test
        max_retries=1,
        backoff_base=0.0,
    )

    result = gw.generate(TaskKind.converse, _simple_messages())
    assert result.provider == "mock_fallback"
    assert result.text == "Fallback reply"
    assert result.attempts == [
        {
            "provider": "mock_429",
            "model": "",
            "outcome": "failed",
            "error_summary": "RateLimitError: [mock_429] HTTP 429: too many requests",
        }
    ]


def test_gateway_sets_cooldown_after_429() -> None:
    """After a 429, the first provider is in cooldown and the second is used."""
    mock_429 = MockProvider(raise_on_call=RateLimitError("m1", 429, "ratelimited"))
    mock_429.name = "m1"
    mock_ok = MockProvider(fixed_output="ok")
    mock_ok.name = "m2"

    gw = LLMGateway(
        providers={"m1": mock_429, "m2": mock_ok},
        fallback_chains={tk.value: ["m1", "m2"] for tk in TaskKind},
        daily_caps={"m1": 999_999, "m2": 999_999},
        cooldown_seconds=60,
        max_retries=1,
        backoff_base=0.0,
    )
    # First call — m1 429s, m2 serves
    r1 = gw.generate(TaskKind.converse, _simple_messages())
    assert r1.provider == "m2"
    assert r1.attempts[0]["error_summary"].startswith("RateLimitError:")

    # m1 should now be in cooldown; next call should still go to m2
    r2 = gw.generate(TaskKind.converse, _simple_messages())
    assert r2.provider == "m2"
    assert gw._is_in_cooldown("m1")


def test_gateway_exhausted_when_all_providers_fail() -> None:
    """GatewayExhaustedError is raised when every provider in chain fails."""
    mock_fail = MockProvider(raise_on_call=ProviderError("mock", 500, "server error"))
    gw = _gateway_with(mock_fail)
    with pytest.raises(GatewayExhaustedError) as raised:
        gw.generate(TaskKind.converse, _simple_messages())
    assert "ProviderError: [mock] HTTP 500: server error" in str(raised.value)


def test_authentication_failure_is_not_retried_and_disables_provider() -> None:
    """Authentication failures skip retries and disable that provider for later calls."""
    auth = _FailingProvider(AuthenticationError("auth", 401, "invalid credentials"), "auth")
    backup = MockProvider(fixed_output="backup")
    backup.name = "backup"
    gateway = LLMGateway(
        providers={"auth": auth, "backup": backup},
        fallback_chains={task.value: ["auth", "backup"] for task in TaskKind},
        daily_caps={"auth": 100, "backup": 100},
        max_retries=3,
        backoff_base=0.0,
    )

    first = gateway.generate(TaskKind.converse, _simple_messages())
    another_gateway = LLMGateway(
        providers={"auth": auth, "backup": backup},
        fallback_chains={task.value: ["auth", "backup"] for task in TaskKind},
        daily_caps={"auth": 100, "backup": 100},
        max_retries=3,
        backoff_base=0.0,
    )
    second = another_gateway.generate(TaskKind.converse, _simple_messages())

    assert first.provider == second.provider == "backup"
    assert auth.calls == 1
    assert "AuthenticationError:" in first.attempts[0]["error_summary"]
    assert second.attempts[0]["error_summary"] == (
        "skipped: authentication failure (provider disabled)"
    )


def test_model_not_found_is_not_retried_and_falls_through() -> None:
    """A missing model is recorded once and the gateway serves from its next provider."""
    invalid_model = _FailingProvider(
        ModelNotFoundError("invalid", 404, "model not found"),
        "invalid",
    )
    backup = MockProvider(fixed_output="backup")
    backup.name = "backup"
    gateway = LLMGateway(
        providers={"invalid": invalid_model, "backup": backup},
        fallback_chains={task.value: ["invalid", "backup"] for task in TaskKind},
        daily_caps={"invalid": 100, "backup": 100},
        max_retries=3,
        backoff_base=0.0,
    )

    result = gateway.generate(TaskKind.converse, _simple_messages())

    assert result.provider == "backup"
    assert invalid_model.calls == 1
    assert result.attempts[0]["error_summary"].startswith("ModelNotFoundError:")


def test_transient_failure_retries_then_falls_through() -> None:
    """Transient errors retry up to the configured limit before fallback."""
    transient = _FailingProvider(
        TransientProviderError("transient", 503, "service unavailable"),
        "transient",
    )
    backup = MockProvider(fixed_output="backup")
    backup.name = "backup"
    gateway = LLMGateway(
        providers={"transient": transient, "backup": backup},
        fallback_chains={task.value: ["transient", "backup"] for task in TaskKind},
        daily_caps={"transient": 100, "backup": 100},
        max_retries=2,
        backoff_base=0.0,
    )

    result = gateway.generate(TaskKind.converse, _simple_messages())

    assert result.provider == "backup"
    assert transient.calls == 2
    assert result.attempts[0]["error_summary"].endswith("(after 2 attempts)")


def test_timeout_retries_then_falls_through() -> None:
    """Timeout exceptions use the transient retry path."""
    timeout = _FailingProvider(TimeoutError("synthetic timeout"), "timeout")
    backup = MockProvider(fixed_output="backup")
    backup.name = "backup"
    gateway = LLMGateway(
        providers={"timeout": timeout, "backup": backup},
        fallback_chains={task.value: ["timeout", "backup"] for task in TaskKind},
        daily_caps={"timeout": 100, "backup": 100},
        max_retries=2,
        backoff_base=0.0,
    )

    result = gateway.generate(TaskKind.converse, _simple_messages())

    assert result.provider == "backup"
    assert timeout.calls == 2
    assert result.attempts[0]["error_summary"].startswith("TransientProviderError:")


def test_unknown_exception_falls_through_and_appears_in_attempts() -> None:
    """Unknown provider exceptions do not abort fallback and appear in attempts."""
    unknown = _FailingProvider(RuntimeError("synthetic unknown failure"), "unknown")
    backup = MockProvider(fixed_output="backup reply")
    backup.name = "backup"
    gateway = LLMGateway(
        providers={"unknown": unknown, "backup": backup},
        fallback_chains={task.value: ["unknown", "backup"] for task in TaskKind},
        daily_caps={"unknown": 100, "backup": 100},
        max_retries=2,
        backoff_base=0.0,
    )

    result = gateway.generate(TaskKind.converse, _simple_messages())

    assert result.provider == "backup"
    assert result.text == "backup reply"
    assert unknown.calls == 1
    assert len(result.attempts) == 1
    assert result.attempts[0]["provider"] == "unknown"
    assert result.attempts[0]["outcome"] == "failed"
    assert "RuntimeError: synthetic unknown failure" in result.attempts[0]["error_summary"]


def test_billing_or_quota_402_disables_provider() -> None:
    """402 Payment Required classifies as BillingOrQuota, disables provider process-wide, and falls through."""
    billing = _FailingProvider(
        BillingOrQuota("cerebras", 402, "Payment required to access this resource."),
        "cerebras",
    )
    backup = MockProvider(fixed_output="backup reply")
    backup.name = "backup"
    gateway = LLMGateway(
        providers={"cerebras": billing, "backup": backup},
        fallback_chains={task.value: ["cerebras", "backup"] for task in TaskKind},
        daily_caps={"cerebras": 100, "backup": 100},
        max_retries=3,
        backoff_base=0.0,
    )

    first = gateway.generate(TaskKind.converse, _simple_messages())
    assert first.provider == "backup"
    assert billing.calls == 1
    assert first.attempts[0]["provider"] == "cerebras"
    assert first.attempts[0]["outcome"] == "failed"
    assert "BillingOrQuota:" in first.attempts[0]["error_summary"]
    assert "Payment required" in first.attempts[0]["error_summary"]

    another_gateway = LLMGateway(
        providers={"cerebras": billing, "backup": backup},
        fallback_chains={task.value: ["cerebras", "backup"] for task in TaskKind},
        daily_caps={"cerebras": 100, "backup": 100},
        max_retries=3,
        backoff_base=0.0,
    )
    second = another_gateway.generate(TaskKind.converse, _simple_messages())
    assert second.provider == "backup"
    assert billing.calls == 1  # Not called again because it was disabled process-wide
    assert second.attempts[0]["provider"] == "cerebras"
    assert second.attempts[0]["outcome"] == "skipped"
    assert second.attempts[0]["error_summary"] == (
        "skipped: billing or quota exhausted (provider disabled)"
    )


def test_all_fail_error_contains_every_attempt() -> None:
    """When all providers fail, GatewayExhaustedError lists every attempt."""
    p1 = _FailingProvider(RuntimeError("custom error 1"), "p1")
    p2 = _FailingProvider(BillingOrQuota("p2", 402, "out of credits"), "p2")
    gateway = LLMGateway(
        providers={"p1": p1, "p2": p2},
        fallback_chains={task.value: ["p1", "p2"] for task in TaskKind},
        daily_caps={"p1": 100, "p2": 100},
        max_retries=1,
        backoff_base=0.0,
    )

    with pytest.raises(GatewayExhaustedError) as raised:
        gateway.generate(TaskKind.converse, _simple_messages())

    assert len(raised.value.attempts) == 2
    assert raised.value.attempts[0]["provider"] == "p1"
    assert "RuntimeError: custom error 1" in raised.value.attempts[0]["error_summary"]
    assert raised.value.attempts[1]["provider"] == "p2"
    assert "BillingOrQuota:" in raised.value.attempts[1]["error_summary"]
    error_str = str(raised.value)
    assert "p1: RuntimeError: custom error 1" in error_str
    assert "p2: BillingOrQuota: [p2] HTTP 402: out of credits" in error_str


def test_missing_provider_is_reported_as_skipped_before_fallback() -> None:
    """Providers without configured keys appear in fallback diagnostics."""
    backup = MockProvider(fixed_output="backup")
    backup.name = "backup"
    gateway = LLMGateway(
        providers={"backup": backup},
        fallback_chains={task.value: ["missing", "backup"] for task in TaskKind},
        daily_caps={"backup": 100},
        skipped_providers={"missing": "no API key configured"},
        provider_models={"missing": "some-model"},
    )

    result = gateway.generate(TaskKind.converse, _simple_messages())

    assert result.attempts[0] == {
        "provider": "missing",
        "model": "some-model",
        "outcome": "skipped",
        "error_summary": "skipped: no API key configured",
    }


# ── Daily cap ─────────────────────────────────────────────────────────────────


def test_gateway_respects_daily_cap() -> None:
    """Gateway skips provider when its daily cap is reached."""
    # Cap of 0 means always exceeded
    mock_ok = MockProvider(fixed_output="should not be called")
    mock_backup = MockProvider(fixed_output="backup")
    mock_backup.name = "backup"

    gw = LLMGateway(
        providers={"mock": mock_ok, "backup": mock_backup},
        fallback_chains={tk.value: ["mock", "backup"] for tk in TaskKind},
        daily_caps={"mock": 0, "backup": 999_999},  # mock capped at 0
        cooldown_seconds=60,
        max_retries=1,
        backoff_base=0.0,
    )
    result = gw.generate(TaskKind.converse, _simple_messages())
    assert result.provider == "backup"
    assert result.attempts[0]["error_summary"] == "skipped: daily cap reached"


def test_daily_cap_after_n_calls() -> None:
    """Provider is skipped once it reaches its cap within the same process."""
    mock_ok = MockProvider(fixed_output="primary")
    mock_backup = MockProvider(fixed_output="backup")
    mock_backup.name = "backup"

    gw = LLMGateway(
        providers={"mock": mock_ok, "backup": mock_backup},
        fallback_chains={tk.value: ["mock", "backup"] for tk in TaskKind},
        daily_caps={"mock": 2, "backup": 999_999},
        cooldown_seconds=0,
        max_retries=1,
        backoff_base=0.0,
    )

    # First two calls should hit mock
    r1 = gw.generate(TaskKind.converse, _simple_messages())
    r2 = gw.generate(TaskKind.converse, _simple_messages())
    assert r1.provider == "mock"
    assert r2.provider == "mock"

    # Third call: mock is capped, should fall to backup
    r3 = gw.generate(TaskKind.converse, _simple_messages())
    assert r3.provider == "backup"


def test_daily_cap_is_persisted_in_sqlite() -> None:
    """A new gateway reads the existing provider usage count from SQLite."""
    engine = build_engine("sqlite:///:memory:")
    counter = _UsageCounter()
    counter.init_db(engine)
    chains = {tk.value: ["mock"] for tk in TaskKind}

    first = LLMGateway(
        providers={"mock": MockProvider(fixed_output="first")},
        fallback_chains=chains,
        daily_caps={"mock": 1},
        usage_counter=counter,
    )
    assert first.generate(TaskKind.converse, _simple_messages()).text == "first"
    assert counter.get("mock") == 1

    persisted_counter = _UsageCounter()
    persisted_counter.init_db(engine)
    second = LLMGateway(
        providers={"mock": MockProvider(fixed_output="second")},
        fallback_chains=chains,
        daily_caps={"mock": 1},
        usage_counter=persisted_counter,
    )
    with pytest.raises(GatewayExhaustedError):
        second.generate(TaskKind.converse, _simple_messages())


def test_user_message_is_not_logged_at_info(caplog: pytest.LogCaptureFixture) -> None:
    """Gateway INFO logs do not contain user message content."""
    private_content = "synthetic-private-message-content"
    gateway = _gateway_with(MockProvider())
    caplog.set_level(logging.INFO, logger="app.llm.gateway")
    gateway.generate(
        TaskKind.converse,
        [{"role": "user", "content": private_content}],
    )
    assert private_content not in caplog.text


# ── Structured output ─────────────────────────────────────────────────────────


class _SampleSchema(BaseModel):
    """Minimal schema for structured output tests."""
    answer: str
    confidence: float


def test_structured_output_success() -> None:
    """Gateway parses valid JSON and returns a populated parsed field."""
    valid_json = '{"answer": "42", "confidence": 0.99}'
    mock = MockProvider(fixed_output=valid_json)
    gw = _gateway_with(mock)
    result = gw.generate(TaskKind.extract, _simple_messages(), json_schema=_SampleSchema)
    assert result.parsed is not None
    assert isinstance(result.parsed, _SampleSchema)
    assert result.parsed.answer == "42"


def test_structured_output_repair_retry() -> None:
    """Gateway attempts one repair call when first JSON is invalid, succeeds on repair."""
    call_count = {"n": 0}
    sent_messages: list[list[dict[str, str]]] = []
    valid_json = '{"answer": "repaired", "confidence": 0.5}'

    class _TwoCallMock(MockProvider):
        """Returns bad JSON first, valid JSON on second call."""
        def generate(self, messages, **kwargs):  # type: ignore[override]
            call_count["n"] += 1
            sent_messages.append(messages)
            if call_count["n"] == 1:
                return LLMResult(
                    text="not json at all",
                    provider="mock",
                    model="mock-model",
                    latency_ms=1.0,
                )
            return LLMResult(
                text=valid_json,
                provider="mock",
                model="mock-model",
                latency_ms=1.0,
            )

    gw = _gateway_with(_TwoCallMock())
    gw.add_filter(
        lambda messages: messages + [{"role": "system", "content": "filtered"}]
    )
    result = gw.generate(TaskKind.extract, _simple_messages(), json_schema=_SampleSchema)
    assert result.parsed is not None
    assert result.parsed.answer == "repaired"
    assert call_count["n"] == 2
    assert all(any(m["content"] == "filtered" for m in msgs) for msgs in sent_messages)


def test_structured_output_raises_after_failed_repair() -> None:
    """Failed JSON parsing is recorded before the gateway exhausts its chain."""
    mock = MockProvider(fixed_output="not json at all {{}")
    gw = _gateway_with(mock)
    with pytest.raises(GatewayExhaustedError) as raised:
        gw.generate(TaskKind.extract, _simple_messages(), json_schema=_SampleSchema)
    assert len(raised.value.attempts) == 1
    assert raised.value.attempts[0]["outcome"] == "failed"
    assert "structured output" in raised.value.attempts[0]["error_summary"].lower()


def test_invalid_structured_output_falls_through_to_next_provider() -> None:
    """A provider whose JSON repair fails does not block a later provider."""
    invalid = MockProvider(fixed_output="not valid JSON")
    invalid.name = "invalid"
    backup = MockProvider(fixed_output='{"answer":"recovered","confidence":0.8}')
    backup.name = "backup"
    gateway = LLMGateway(
        providers={"invalid": invalid, "backup": backup},
        fallback_chains={
            task.value: ["invalid/model-a", "backup/model-b"] for task in TaskKind
        },
        daily_caps={"invalid": 10, "backup": 10},
        max_retries=1,
    )

    result = gateway.generate(
        TaskKind.extract,
        _simple_messages(),
        json_schema=_SampleSchema,
    )

    assert result.provider == "backup"
    assert result.parsed.answer == "recovered"
    assert result.attempts[0]["provider"] == "invalid"
    assert result.attempts[0]["model"] == "model-a"
    assert result.attempts[0]["outcome"] == "failed"


def test_failed_structured_repair_falls_through_with_safe_error_details() -> None:
    """A provider API error during repair is recorded without exposing its body."""

    class _RepairFailureProvider(MockProvider):
        """Return invalid JSON, then simulate a provider schema-validation error."""

        calls = 0

        def generate(self, messages, **kwargs):  # type: ignore[override]
            self.calls += 1
            if self.calls == 1:
                return LLMResult(
                    text="not valid JSON",
                    provider="primary",
                    model="model-a",
                    latency_ms=1.0,
                )
            raise BadRequestError(
                "primary",
                400,
                "json_validate_failed failed_generation=synthetic-private-output",
            )

    primary = _RepairFailureProvider()
    primary.name = "primary"
    backup = MockProvider(fixed_output='{"answer":"recovered","confidence":0.8}')
    backup.name = "backup"
    gateway = LLMGateway(
        providers={"primary": primary, "backup": backup},
        fallback_chains={
            task.value: ["primary/model-a", "backup/model-b"] for task in TaskKind
        },
        daily_caps={"primary": 10, "backup": 10},
        max_retries=1,
    )

    result = gateway.generate(
        TaskKind.extract,
        _simple_messages(),
        json_schema=_SampleSchema,
    )

    assert result.provider == "backup"
    failure = result.attempts[0]["error_summary"]
    assert "HTTP 400 json_validate_failed" in failure
    assert "synthetic-private-output" not in failure


def test_gpt_oss_structured_extraction_is_skipped_without_affecting_other_tasks() -> None:
    """Only extraction skips the unsupported Groq model; other task kinds still use it."""
    groq = MockProvider(fixed_output="plain response")
    groq.name = "groq"
    backup = MockProvider(
        fixed_output='{"answer":"backup","confidence":0.9}'
    )
    backup.name = "backup"
    gateway = LLMGateway(
        providers={"groq": groq, "backup": backup},
        fallback_chains={
            task.value: ["groq/openai/gpt-oss-20b", "backup"] for task in TaskKind
        },
        daily_caps={"groq": 10, "backup": 10},
        max_retries=1,
    )

    extracted = gateway.generate(
        TaskKind.extract,
        _simple_messages(),
        json_schema=_SampleSchema,
    )
    classified = gateway.generate(TaskKind.classify, _simple_messages())

    assert extracted.provider == "backup"
    assert extracted.attempts[0]["outcome"] == "skipped"
    assert "structured extraction unsupported" in extracted.attempts[0]["error_summary"]
    assert classified.provider == "groq"


# ── pre_send_filter ───────────────────────────────────────────────────────────


def test_pre_send_filter_is_applied() -> None:
    """pre_send_filter callables mutate the message list before dispatch."""
    received: list[list[dict]] = []

    class _CapturingMock(MockProvider):
        def generate(self, messages, **kwargs):  # type: ignore[override]
            received.append(messages)
            return LLMResult(text="ok", provider="mock", model="m", latency_ms=1.0)

    def _add_tag(msgs: list[dict]) -> list[dict]:
        return msgs + [{"role": "system", "content": "FILTERED"}]

    gw = _gateway_with(_CapturingMock())
    gw.add_filter(_add_tag)
    gw.generate(TaskKind.converse, _simple_messages())

    assert len(received) == 1
    assert any(m["content"] == "FILTERED" for m in received[0])


def test_multiple_filters_applied_in_order() -> None:
    """Multiple pre_send_filters are applied left-to-right."""
    log: list[str] = []

    def f1(msgs: list[dict]) -> list[dict]:
        log.append("f1")
        return msgs

    def f2(msgs: list[dict]) -> list[dict]:
        log.append("f2")
        return msgs

    gw = _gateway_with(MockProvider())
    gw.add_filter(f1)
    gw.add_filter(f2)
    gw.generate(TaskKind.converse, _simple_messages())
    assert log == ["f1", "f2"]
