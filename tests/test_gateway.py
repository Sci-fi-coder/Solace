"""Tests for app/llm/gateway — resilience, filters, structured output, daily cap."""

from __future__ import annotations

import pytest
import time

from app.contracts import TaskKind
from app.llm.gateway import (
    DailyCapExceededError,
    GatewayExhaustedError,
    LLMGateway,
    StructuredOutputError,
    _UsageCounter,
)
from app.llm.providers import MockProvider, RateLimitError, ProviderError, LLMResult
from pydantic import BaseModel


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


# ── Basic success ─────────────────────────────────────────────────────────────


def test_gateway_returns_result_on_success() -> None:
    """Gateway returns LLMResult when the provider succeeds."""
    gw = _gateway_with(MockProvider(task_kind=TaskKind.converse.value))
    result = gw.generate(TaskKind.converse, _simple_messages())
    assert result.provider == "mock"
    assert isinstance(result.text, str)
    assert result.latency_ms >= 0


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

    # m1 should now be in cooldown; next call should still go to m2
    r2 = gw.generate(TaskKind.converse, _simple_messages())
    assert r2.provider == "m2"
    assert gw._is_in_cooldown("m1")


def test_gateway_exhausted_when_all_providers_fail() -> None:
    """GatewayExhaustedError is raised when every provider in chain fails."""
    mock_fail = MockProvider(raise_on_call=ProviderError("mock", 500, "server error"))
    gw = _gateway_with(mock_fail)
    with pytest.raises(GatewayExhaustedError):
        gw.generate(TaskKind.converse, _simple_messages())


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
    # Note: usage counter is module-level, so we patch it inline
    from app.llm import gateway as gw_module
    original_get = gw_module._usage.get
    try:
        gw_module._usage._counts[("mock", gw_module._usage._today())] = 2
        r3 = gw.generate(TaskKind.converse, _simple_messages())
        assert r3.provider == "backup"
    finally:
        # restore
        gw_module._usage._counts.pop(("mock", gw_module._usage._today()), None)


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
    valid_json = '{"answer": "repaired", "confidence": 0.5}'

    class _TwoCallMock(MockProvider):
        """Returns bad JSON first, valid JSON on second call."""
        def generate(self, messages, **kwargs):  # type: ignore[override]
            call_count["n"] += 1
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
    result = gw.generate(TaskKind.extract, _simple_messages(), json_schema=_SampleSchema)
    assert result.parsed is not None
    assert result.parsed.answer == "repaired"
    assert call_count["n"] == 2


def test_structured_output_raises_after_failed_repair() -> None:
    """StructuredOutputError is raised when both attempts return bad JSON."""
    mock = MockProvider(fixed_output="not json at all {{}")
    gw = _gateway_with(mock)
    with pytest.raises(StructuredOutputError):
        gw.generate(TaskKind.extract, _simple_messages(), json_schema=_SampleSchema)


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
