"""Tests for provider-specific reasoning and response handling."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.config import Settings
from app.contracts import TaskKind
from app.llm.gateway import LLMGateway
from app.llm.providers import LLMResult, MockProvider, OpenAICompatProvider


class _FakeCompletions:
    """Return queued completion responses and record outgoing request arguments."""

    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        """Return the next response or raise the next queued exception."""
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _response(
    content: str | None,
    *,
    finish_reason: str = "stop",
    reasoning: str = "",
    reasoning_content: str = "",
) -> SimpleNamespace:
    """Build the small response shape consumed by OpenAICompatProvider."""
    message = SimpleNamespace(
        content=content,
        reasoning=reasoning,
        reasoning_content=reasoning_content,
    )
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=None)


def _provider(*responses: object) -> tuple[OpenAICompatProvider, _FakeCompletions]:
    """Build a provider around a deterministic fake OpenAI client."""
    completions = _FakeCompletions(list(responses))
    provider = OpenAICompatProvider.__new__(OpenAICompatProvider)
    provider.name = "groq"
    provider._model = "openai/gpt-oss-20b"
    provider._client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    return provider, completions


@pytest.mark.parametrize(
    ("task", "expected_effort"),
    [
        (TaskKind.converse, "medium"),
        (TaskKind.classify, "low"),
        (TaskKind.judge, "low"),
        (TaskKind.extract, "low"),
    ],
)
def test_gpt_oss_receives_configured_reasoning_effort(
    task: TaskKind,
    expected_effort: str,
) -> None:
    """Gateway sends task-configured reasoning effort for GPT-OSS requests."""
    config = Settings(_env_file=None)
    provider, completions = _provider(_response("synthetic answer"))
    chains = {kind.value: ["groq"] for kind in TaskKind}
    gateway = LLMGateway(
        providers={"groq": provider},
        fallback_chains=chains,
        daily_caps={"groq": 10},
        task_reasoning_efforts={
            kind.value: config.get_reasoning_effort(kind.value) for kind in TaskKind
        },
    )

    gateway.generate(task, [{"role": "user", "content": "Synthetic test."}])

    assert completions.calls[0]["reasoning_effort"] == expected_effort


def test_non_gpt_oss_models_do_not_receive_reasoning_effort() -> None:
    """Other OpenAI-compatible models do not receive GPT-OSS-only parameters."""
    provider, completions = _provider(_response("synthetic answer"))

    provider.generate(
        [{"role": "user", "content": "Synthetic test."}],
        model="vendor/other-model",
        reasoning_effort="low",
    )

    assert "reasoning_effort" not in completions.calls[0]


def test_rejected_reasoning_effort_retries_once_without_parameter() -> None:
    """Retry a GPT-OSS request once when the endpoint rejects reasoning_effort."""
    provider, completions = _provider(
        RuntimeError("Unsupported parameter: reasoning_effort"),
        _response("synthetic answer"),
    )

    result = provider.generate(
        [{"role": "user", "content": "Synthetic test."}],
        reasoning_effort="low",
    )

    assert result.text == "synthetic answer"
    assert len(completions.calls) == 2
    assert completions.calls[0]["reasoning_effort"] == "low"
    assert "reasoning_effort" not in completions.calls[1]


def test_reasoning_content_is_excluded_from_llm_result_text() -> None:
    """Only final answer content is returned, not separate or tagged reasoning."""
    provider, _ = _provider(
        _response(
            "<think>hidden tagged reasoning</think> final answer",
            reasoning="hidden separate reasoning",
            reasoning_content="hidden reasoning content",
        )
    )

    result = provider.generate([{"role": "user", "content": "Synthetic test."}])

    assert result.text == "final answer"
    assert "hidden" not in result.text


def test_empty_length_response_retries_with_larger_token_limit() -> None:
    """An empty GPT-OSS length-limited response gets one larger-budget retry."""
    provider, completions = _provider(
        _response("", finish_reason="length"),
        _response("synthetic answer"),
    )

    result = provider.generate(
        [{"role": "user", "content": "Synthetic test."}],
        max_tokens=80,
        reasoning_effort="low",
    )

    assert result.text == "synthetic answer"
    assert [call["max_tokens"] for call in completions.calls] == [80, 1080]
    assert len(completions.calls) == 2


def test_empty_length_response_error_falls_through_to_next_provider(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A second empty length-limited response raises and falls through in the gateway."""
    provider, completions = _provider(
        _response("", finish_reason="length"),
        _response("", finish_reason="length"),
    )
    backup = MockProvider(fixed_output="backup answer")
    gateway = LLMGateway(
        providers={"groq": provider, "backup": backup},
        fallback_chains={kind.value: ["groq", "backup"] for kind in TaskKind},
        daily_caps={"groq": 10, "backup": 10},
    )

    result: LLMResult = gateway.generate(
        TaskKind.converse,
        [{"role": "user", "content": "Synthetic test."}],
    )

    assert result.provider == "backup"
    assert result.text == "backup answer"
    assert "EmptyResponseError" in result.attempts[0]["error_summary"]
    assert len(completions.calls) == 2
    assert "unexpected ProviderError" not in caplog.text
