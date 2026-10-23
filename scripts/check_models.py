"""List configured providers' models and compare them with the configured defaults."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from app.config import Settings, settings

logger = logging.getLogger(__name__)

_PROVIDERS = ("gemini", "groq", "openrouter", "mistral", "cerebras")
_GEMINI_MODELS_URL = "https://generativelanguage.googleapis.com/v1beta/models"


def _list_provider_models(name: str, config: dict[str, object]) -> list[str]:
    """Fetch model identifiers from one provider's model catalog."""
    base_url = str(config["base_url"]).rstrip("/")
    url = _GEMINI_MODELS_URL if name == "gemini" else f"{base_url}/models"
    headers = {"Accept": "application/json"}
    key = str(config["api_key"])
    if name == "gemini":
        headers["x-goog-api-key"] = key
    else:
        headers["Authorization"] = f"Bearer {key}"

    model_ids: list[str] = []
    page_token = ""
    while True:
        page_url = url
        if name == "gemini" and page_token:
            page_url = f"{url}?{urlencode({'pageToken': page_token})}"
        request = Request(page_url, headers=headers)
        with urlopen(request, timeout=20) as response:
            body = json.loads(response.read())

        entries = body.get("models", body.get("data", []))
        for entry in entries:
            model_id = str(entry.get("name", entry.get("id", "")))
            if model_id.startswith("models/"):
                model_id = model_id.removeprefix("models/")
            if model_id:
                model_ids.append(model_id)

        page_token = str(body.get("nextPageToken", ""))
        if name != "gemini" or not page_token:
            break
    return model_ids


def check_models(
    *,
    config: Settings = settings,
    provider_names: list[str] | None = None,
    output: Callable[[str], None] = print,
) -> dict[str, list[str]]:
    """List available models for configured providers and warn on stale defaults."""
    selected = provider_names or list(_PROVIDERS)
    catalogs: dict[str, list[str]] = {}

    for name in selected:
        provider_config = config.provider_config(name)
        key = str(provider_config["api_key"])
        if not key:
            output(f"{name}: skipped: no API key configured")
            continue

        try:
            models = _list_provider_models(name, provider_config)
        except HTTPError as exc:
            output(f"WARNING {name}: model catalog request failed (HTTP {exc.code})")
            continue
        except (TimeoutError, URLError, json.JSONDecodeError) as exc:
            output(f"WARNING {name}: model catalog request failed ({type(exc).__name__})")
            continue

        catalogs[name] = models
        output(f"{name}: available models ({len(models)}):")
        output(", ".join(models) if models else "(none returned)")
        configured_model = str(provider_config["model"])
        if configured_model not in models:
            output(
                f"WARNING {name}: configured model {configured_model!r} "
                "is not in the provider catalog."
            )
    return catalogs


def main() -> int:
    """Run model-catalog checks for configured provider keys."""
    logging.basicConfig(level=logging.INFO)
    settings.log_configuration()
    check_models()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
