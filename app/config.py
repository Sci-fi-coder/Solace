"""Solace application configuration.

All settings come from environment variables (or a .env file).
See .env.example for the full list.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


def redact_secrets(value: str, secrets: list[str] | None = None) -> str:
    """Remove known and recognizable API-key values from diagnostic text."""
    for secret in secrets or []:
        if secret:
            value = value.replace(secret, "[REDACTED]")
    return re.sub(
        r"(?i)(gsk_[A-Za-z0-9_-]+|sk-[A-Za-z0-9_-]+|AIza[0-9A-Za-z_-]+)",
        "[REDACTED]",
        value,
    )


def _configured_key(value: str) -> str:
    """Return a configured API key, treating whitespace and placeholders as absent."""
    key = value.strip()
    normalized = re.sub(r"[^a-z0-9]", "", key.lower())
    placeholders = {
        "changeme",
        "insertapikeyhere",
        "placeholder",
        "yourapikey",
        "yourcerebrasapikey",
        "yourgeminiapikey",
        "yourgroqapikey",
        "yourmistralapikey",
        "youropenrouterapikey",
        "yourapikeyhere",
        "replacewithapikey",
    }
    if normalized in placeholders or normalized.startswith("placeholder"):
        return ""
    return key


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
    gemini_model: str = "gemini-2.5-flash"
    # Local protective caps, not claims about vendor quotas; override via *_DAILY_CAP.
    gemini_daily_cap: int = 20

    # ── Groq ─────────────────────────────────────────────────────────────────
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "llama-3.1-8b-instant"
    groq_daily_cap: int = 20

    # ── OpenRouter ───────────────────────────────────────────────────────────
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "mistralai/mistral-small-2603"
    openrouter_daily_cap: int = 20

    # ── Mistral ──────────────────────────────────────────────────────────────
    mistral_api_key: str = ""
    mistral_base_url: str = "https://api.mistral.ai/v1"
    mistral_model: str = "mistral-small-latest"
    mistral_daily_cap: int = 20

    # ── Cerebras ─────────────────────────────────────────────────────────────
    cerebras_api_key: str = ""
    cerebras_base_url: str = "https://api.cerebras.ai/v1"
    cerebras_model: str = "qwen-3.8-27b"
    cerebras_daily_cap: int = 20

    # ── Gateway resilience ────────────────────────────────────────────────────
    provider_cooldown_seconds: int = 60
    gateway_max_retries: int = 3
    gateway_backoff_base: float = 1.0

    # ── Fallback chains per TaskKind ─────────────────────────────────────────
    # Each entry is "<provider>/<model>" — resolved at runtime by the gateway.
    # Override via env vars: FALLBACK_CONVERSE="groq/llama-3.1-8b-instant,gemini/gemini-2.5-flash"
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
        """Return ordered provider or provider/model entries for a task kind."""
        raw = getattr(self, f"fallback_{task}", self.fallback_converse)
        return [p.strip() for p in raw.split(",") if p.strip()]

    def provider_config(self, provider: str) -> dict[str, Any]:
        """Return the key/base_url/model/daily_cap dict for a named provider."""
        p = provider.lower()
        return {
            "api_key": _configured_key(getattr(self, f"{p}_api_key", "")),
            "base_url": getattr(self, f"{p}_base_url", ""),
            "model": getattr(self, f"{p}_model", ""),
            "daily_cap": getattr(self, f"{p}_daily_cap", 0),
        }

    @property
    def dotenv_path(self) -> Path:
        """Return the absolute path to the configured dotenv file."""
        configured_path = self.model_config.get("env_file", ".env")
        if isinstance(configured_path, (list, tuple)):
            configured_path = configured_path[0] if configured_path else ".env"
        return Path(str(configured_path)).resolve()

    def log_configuration(self) -> None:
        """Log dotenv provenance and configured provider names without values."""
        path = self.dotenv_path
        if path.is_file():
            logger.info("Loaded dotenv file: %s", path)
            dotenv = dotenv_values(path)
            real_environment = {name.upper(): value for name, value in os.environ.items()}
            recognized_settings = {
                name.upper() for name in type(self).model_fields
            }
            for name, file_value in dotenv.items():
                if name.upper() not in recognized_settings:
                    continue
                environment_value = real_environment.get(name.upper())
                if (
                    file_value is not None
                    and environment_value is not None
                    and file_value != environment_value
                ):
                    logger.warning(
                        "Environment variable %s differs between process environment "
                        "and %s; process environment takes precedence.",
                        name.upper(),
                        path,
                    )
        else:
            logger.info("Dotenv file not found: %s", path)

        provider_names = ("gemini", "groq", "openrouter", "mistral", "cerebras")
        configured = [
            name for name in provider_names if self.provider_config(name)["api_key"]
        ]
        logger.info(
            "LLM providers with configured keys: %s",
            ", ".join(configured) if configured else "none",
        )

    @model_validator(mode="after")
    def _warn_no_keys(self) -> Settings:
        """Log a warning (not an error) if no LLM keys are configured."""
        providers = ["gemini", "groq", "openrouter", "mistral", "cerebras"]
        has_key = any(self.provider_config(provider)["api_key"] for provider in providers)
        if not has_key:
            logger.warning(
                "No LLM API keys found. Gateway will use MockProvider only. "
                "Set at least one key in .env."
            )
        return self


# Singleton — import this from anywhere in the app.
settings = Settings()
