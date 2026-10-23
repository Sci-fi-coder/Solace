"""Tests for smoke-test CLI argument handling."""

from __future__ import annotations

import pytest

from scripts.smoke_llm import _parse_args


def test_smoke_defaults_to_one_converse_task() -> None:
    """Smoke-test defaults to one task and leaves provider selection unset."""
    args = _parse_args([])

    assert args.task == "converse"
    assert args.provider is None


def test_smoke_accepts_provider_and_task_options() -> None:
    """Smoke-test options select the requested provider and task."""
    args = _parse_args(["--provider", "gemini", "--task", "classify"])

    assert args.provider == "gemini"
    assert args.task == "classify"


def test_smoke_rejects_unknown_options() -> None:
    """Unknown options are parser errors instead of being silently ignored."""
    with pytest.raises(SystemExit):
        _parse_args(["--unknowable"])
