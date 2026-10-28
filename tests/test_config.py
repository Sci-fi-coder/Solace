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


def test_default_model_fallbacks_use_gemini_lite_before_groq() -> None:
    """Cheap-task defaults try verified Gemini Flash Lite before Groq GPT-OSS."""
    config = Settings(_env_file=None)

    assert config.groq_model == "openai/gpt-oss-20b"
    assert config.fallback_converse.split(",")[0] == "gemini"
    for task in ("classify", "judge", "extract"):
        chain = config.get_fallback_chain(task)
        assert chain[:2] == [
            "gemini/gemini-3.1-flash-lite",
            "groq/openai/gpt-oss-20b",
        ]
        assert all(not entry.startswith("cerebras") for entry in chain)
