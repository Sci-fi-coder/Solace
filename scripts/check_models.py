"""List configured providers' models and compare them with the configured defaults."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable

from app.config import Settings, redact_secrets, settings

logger = logging.getLogger(__name__)

_PROVIDERS = ("gemini", "groq", "openrouter", "mistral", "cerebras")


def _list_provider_models(name: str, config: dict[str, object]) -> list[str]:
    """Fetch model identifiers from one provider using official or OpenAI-compatible client."""
    key = str(config["api_key"])
    if name == "gemini":
        from google import genai  # type: ignore[import-untyped]

        client = genai.Client(api_key=key)
        return [
            m.name.removeprefix("models/")
            for m in client.models.list()
            if getattr(m, "name", None)
        ]

    from openai import OpenAI  # type: ignore[import-untyped]

    base_url = str(config["base_url"]).rstrip("/")
    client = OpenAI(api_key=key, base_url=base_url)
    models_page = client.models.list()
    return [m.id for m in models_page if getattr(m, "id", None)]


def check_models(
    *,
    config: Settings = settings,
    provider_names: list[str] | None = None,
    output: Callable[[str], None] = print,
    verbose: bool = False,
) -> dict[str, list[str]]:
    """List available models for configured providers and warn on stale defaults.

    By default, prints only warnings (configured model missing, catalog unreachable).
    Pass verbose=True to print skipped providers and complete model lists.
    """
    selected = provider_names or list(_PROVIDERS)
    catalogs: dict[str, list[str]] = {}

    for name in selected:
        provider_config = config.provider_config(name)
        key = str(provider_config["api_key"])
        if not key:
            if verbose:
                output(f"{name}: skipped: no API key configured")
            continue

        try:
            models = _list_provider_models(name, provider_config)
        except Exception as exc:
            err_msg = redact_secrets(f"{type(exc).__name__}: {exc}", [key])
            output(f"WARNING {name}: model catalog request failed ({err_msg})")
            continue

        catalogs[name] = models
        if verbose:
            output(f"{name}: available models ({len(models)}):")
            output(", ".join(models) if models else "(none returned)")

        configured_model = str(provider_config["model"])
        field = Settings.model_fields.get(f"{name}_model")
        default_model = str(field.default) if field and field.default is not None else None

        if configured_model not in models:
            output(
                f"WARNING {name}: configured model {configured_model!r} "
                "is not in the provider catalog."
            )
        if default_model and default_model != configured_model and default_model not in models:
            output(
                f"WARNING {name}: default model {default_model!r} "
                "is not in the provider catalog."
            )
    return catalogs


def main(argv: list[str] | None = None) -> int:
    """Run model-catalog checks for configured provider keys."""
    parser = argparse.ArgumentParser(description="Check provider model catalogs.")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print full model lists and skipped providers.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    settings.log_configuration()
    check_models(verbose=args.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
