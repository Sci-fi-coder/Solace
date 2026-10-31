"""Tests for settings that affect provider configuration and credential handling."""

from __future__ import annotations

from pathlib import Path

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


def test_extract_fallback_excludes_groq_gpt_oss() -> None:
    """Structured extraction avoids the unsupported Groq GPT-OSS model only."""
    config = Settings(_env_file=None)

    assert config.groq_model == "openai/gpt-oss-20b"
    assert config.fallback_converse.split(",")[0] == "gemini"
    for task in ("classify", "judge"):
        chain = config.get_fallback_chain(task)
        assert chain[:2] == [
            "gemini/gemini-3.1-flash-lite",
            "groq/openai/gpt-oss-20b",
        ]
        assert all(not entry.startswith("cerebras") for entry in chain)
    assert config.get_fallback_chain("extract") == [
        "gemini/gemini-3.1-flash-lite",
        "mistral",
        "openrouter",
    ]


def test_env_example_lists_all_settings_as_empty_values() -> None:
    """The example dotenv file exposes every setting without storing defaults."""
    expected_names = {name.upper() for name in Settings.model_fields}
    entries = {}
    for line in Path(".env.example").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#") and "=" in line:
            name, value = line.split("=", maxsplit=1)
            entries[name] = value

    assert set(entries) == expected_names
    assert all(value == "" for value in entries.values())


def test_empty_dotenv_values_leave_settings_defaults_intact(tmp_path: Path) -> None:
    """Blank values copied from the example do not invalidate settings."""
    env_file = tmp_path / "empty-settings.env"
    env_file.write_text(
        "GEMINI_MODEL=\nGROQ_ENABLED=\nGATEWAY_MAX_RETRIES=\n",
        encoding="utf-8",
    )

    config = Settings(_env_file=env_file)

    assert config.gemini_model == "gemini-3.1-flash-lite"
    assert config.groq_enabled is True
    assert config.gateway_max_retries == 3
