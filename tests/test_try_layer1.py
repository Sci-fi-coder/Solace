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
    assert args.fresh is False
    assert args.session is None


def test_try_layer1_accepts_all_optional_flags() -> None:
    """The smoke-test and fresh-session flags are exposed by argparse."""
    args = _parse_args(["--with-llm", "--no-emotion", "--verbose", "--fresh"])

    assert args.with_llm is True
    assert args.no_emotion is True
    assert args.verbose is True
    assert args.fresh is True


def test_try_layer1_accepts_named_session() -> None:
    """A named session enables sticky escalation across interactive inputs."""
    args = _parse_args(["--session", "synthetic-session"])

    assert args.session == "synthetic-session"
    assert args.fresh is False


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
        "rule_level",
        "final_level",
        "level",
        "reasons",
        "degraded",
        "classifier_provider",
        "classifier_model",
    }
    assert result["level"] == "none"
    assert result["rule_level"] == "none"
    assert result["final_level"] == "none"


def test_verbose_prints_full_result_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verbose mode includes the full trace and optional emotion field."""
    messages = iter(["A synthetic calm test message.", "/quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(messages))

    main(["--no-emotion", "--verbose"])

    result = json.loads(capsys.readouterr().out.removeprefix("You> "))
    assert {
        "rule_level",
        "final_level",
        "safety",
        "emotion",
        "fixed_template",
        "trace",
    } == set(result)
    assert result["emotion"] is None
    assert result["rule_level"] == result["final_level"] == "none"
    assert result["trace"]["safety"]["level"] == "none"
    assert [step["name"] for step in result["trace"]["steps"]] == ["safety"]


@pytest.mark.parametrize(
    ("args", "expected_final_level"),
    [
        (["--no-emotion"], "none"),
        (["--no-emotion", "--fresh"], "none"),
        (["--no-emotion", "--session", "synthetic-session"], "elevated"),
    ],
)
def test_inputs_are_fresh_unless_a_session_is_named(
    args: list[str],
    expected_final_level: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Sticky escalation is isolated per input unless the caller reuses a session."""
    messages = iter(["I will kill myself tonight.", "A synthetic calm test message.", "/quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(messages))

    main(args)

    results = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert results[0]["rule_level"] == results[0]["final_level"] == "crisis"
    assert results[1]["rule_level"] == "none"
    assert results[1]["final_level"] == expected_final_level
