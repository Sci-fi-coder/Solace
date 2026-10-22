"""Solace application configuration.

All settings come from environment variables (or a .env file).
See .env.example for the full list.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ── Per-provider settings ─────────────────────────────────────────────────────


class Settings(BaseSettings):
    """Application-wide settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ──────────────────────────────────────────────────────────────────
    environment: str = "development"
    log_level: str = "INFO"

    # ── Database ─────────────────────────────────────────────────────────────
    database_url: str = "sqlite:///./data/solace.db"

    # ── Gemini ───────────────────────────────────────────────────────────────
    gemini_api_key: str = ""
    gemini_model: str = "gemini-1.5-flash"
    gemini_daily_cap: int = 1500

    # ── Groq ─────────────────────────────────────────────────────────────────
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "llama3-8b-8192"
    groq_daily_cap: int = 1500

    # ── OpenRouter ───────────────────────────────────────────────────────────
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "mistralai/mistral-7b-instruct"
    openrouter_daily_cap: int = 200

    # ── Mistral ──────────────────────────────────────────────────────────────
    mistral_api_key: str = ""
    mistral_base_url: str = "https://api.mistral.ai/v1"
    mistral_model: str = "mistral-small-latest"
    mistral_daily_cap: int = 500

    # ── Cerebras ─────────────────────────────────────────────────────────────
    cerebras_api_key: str = ""
    cerebras_base_url: str = "https://api.cerebras.ai/v1"
    cerebras_model: str = "llama3.1-8b"
    cerebras_daily_cap: int = 1000

    # ── Gateway resilience ────────────────────────────────────────────────────
    provider_cooldown_seconds: int = 60
    gateway_max_retries: int = 3
    gateway_backoff_base: float = 1.0

    # ── Fallback chains per TaskKind ─────────────────────────────────────────
    # Each entry is "<provider>/<model>" — resolved at runtime by the gateway.
    # Override via env vars: FALLBACK_CONVERSE="groq/llama3-8b-8192,gemini/gemini-1.5-flash"
    fallback_converse: str = Field(
        default="gemini,groq,openrouter,mistral,cerebras",
        description="Comma-separated provider order for 'converse' tasks.",
    )
    fallback_classify: str = Field(
        default="groq,gemini,mistral,cerebras,openrouter",
        description="Comma-separated provider order for 'classify' tasks.",
    )
    fallback_judge: str = Field(
        default="gemini,groq,openrouter,mistral,cerebras",
        description="Comma-separated provider order for 'judge' tasks.",
    )
    fallback_extract: str = Field(
        default="gemini,groq,mistral,cerebras,openrouter",
        description="Comma-separated provider order for 'extract' tasks.",
    )

    def get_fallback_chain(self, task: str) -> list[str]:
        """Return ordered list of provider names for a given task kind."""
        raw = getattr(self, f"fallback_{task}", self.fallback_converse)
        return [p.strip() for p in raw.split(",") if p.strip()]

    def provider_config(self, provider: str) -> dict[str, Any]:
        """Return the key/base_url/model/daily_cap dict for a named provider."""
        p = provider.lower()
        return {
            "api_key": getattr(self, f"{p}_api_key", ""),
            "base_url": getattr(self, f"{p}_base_url", ""),
            "model": getattr(self, f"{p}_model", ""),
            "daily_cap": getattr(self, f"{p}_daily_cap", 0),
        }

    @model_validator(mode="after")
    def _warn_no_keys(self) -> "Settings":
        """Log a warning (not an error) if no LLM keys are configured."""
        providers = ["gemini", "groq", "openrouter", "mistral", "cerebras"]
        has_key = any(getattr(self, f"{p}_api_key") for p in providers)
        if not has_key:
            import logging
            logging.getLogger(__name__).warning(
                "No LLM API keys found. Gateway will use MockProvider only. "
                "Set at least one key in .env."
            )
        return self


# Singleton — import this from anywhere in the app.
settings = Settings()
