"""Tests for settings that affect provider configuration and credential handling."""

from __future__ import annotations

from app.config import Settings


def test_empty_and_placeholder_keys_are_not_configured() -> None:
    """Whitespace and example placeholder values do not activate providers."""
    config = Settings(
        _env_file=None,
        gemini_api_key="   ",
        groq_api_key="your_groq_api_key",
        cerebras_api_key="replace-with-api-key",
    )

    assert config.provider_config("gemini")["api_key"] == ""
    assert config.provider_config("groq")["api_key"] == ""
    assert config.provider_config("cerebras")["api_key"] == ""


def test_unrecognized_environment_variables_are_ignored() -> None:
    """Provider settings ignore unrelated variables such as TAVILY_API_KEY."""
    config = Settings(_env_file=None, tavily_api_key="synthetic-unused-value")

    assert not hasattr(config, "tavily_api_key")
