"""Tests for the local Layer 1 smoke-test command-line interface."""

from __future__ import annotations

import json

import pytest

from scripts.try_layer1 import _parse_args, main


def test_try_layer1_defaults_to_rules_only_with_emotion_enabled() -> None:
    """The CLI does not opt into an LLM or disable emotion unless requested."""
    args = _parse_args([])

    assert args.with_llm is False
    assert args.no_emotion is False
    assert args.verbose is False


def test_try_layer1_accepts_all_optional_flags() -> None:
    """All three smoke-test flags are independently exposed by argparse."""
    args = _parse_args(["--with-llm", "--no-emotion", "--verbose"])

    assert args.with_llm is True
    assert args.no_emotion is True
    assert args.verbose is True


def test_no_emotion_prints_only_safety_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Skipping emotion avoids its output and keeps the default result concise."""
    messages = iter(["A synthetic calm test message.", "/quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(messages))

    main(["--no-emotion"])

    output = capsys.readouterr().out.strip()
    result = json.loads(output.removeprefix("You> "))
    assert set(result) == {
        "level",
        "reasons",
        "degraded",
        "classifier_provider",
        "classifier_model",
    }
    assert result["level"] == "none"


def test_verbose_prints_full_result_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verbose mode includes the full trace and optional emotion field."""
    messages = iter(["A synthetic calm test message.", "/quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(messages))

    main(["--no-emotion", "--verbose"])

    result = json.loads(capsys.readouterr().out.removeprefix("You> "))
    assert {"safety", "emotion", "fixed_template", "trace"} == set(result)
    assert result["emotion"] is None
    assert result["trace"]["safety"]["level"] == "none"
    assert [step["name"] for step in result["trace"]["steps"]] == ["safety"]
