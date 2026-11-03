"""Record/replay wrapper around the LLM gateway: stable demos, fewer calls, honest labelling.

Modes: off (pass through), auto (use a recorded answer if present, else call live and record),
record (always call live and overwrite), replay (recorded answers only; a miss raises LookupError).
The key covers the task, schema, temperature and full messages, so any prompt change is a miss.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from pydantic import BaseModel

from app.contracts import TaskKind

logger = logging.getLogger(__name__)
MODES = ("off", "auto", "record", "replay")


def default_cache_path() -> Path:
    """data/cache/llm_cache.json relative to the repository root (git-ignored)."""
    return Path(__file__).resolve().parents[2] / "data" / "cache" / "llm_cache.json"


class CachingGateway:
    """Drop-in for LLMGateway.generate() used by the engine."""

    def __init__(self, inner: Any, path: Path | None = None, mode: str = "auto") -> None:
        if mode not in MODES:
            raise ValueError(f"cache mode must be one of {MODES}")
        self._inner = inner
        self._path = path or default_cache_path()
        self.mode = mode
        self._store: dict[str, dict[str, str]] = self._load()
        self.hits = 0
        self.misses = 0

    def add_filter(self, fn: Any) -> None:
        """Forward pre-send filters to the wrapped gateway."""
        add = getattr(self._inner, "add_filter", None)
        if add is not None:
            add(fn)

    def _load(self) -> dict[str, dict[str, str]]:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=self._path.parent, delete=False, encoding="utf-8") as fh:
            json.dump(self._store, fh, ensure_ascii=False)
            tmp = fh.name
        os.replace(tmp, self._path)

    @staticmethod
    def _key(task: TaskKind, messages: list[dict[str, str]], schema: type[BaseModel] | None, temperature: float) -> str:
        blob = json.dumps(
            {"t": task.value, "m": messages, "s": schema.__name__ if schema else None, "temp": round(temperature, 2)},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def generate(
        self,
        task: TaskKind,
        messages: list[dict[str, str]],
        *,
        json_schema: type[BaseModel] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 800,
        **extra: Any,
    ) -> Any:
        """Serve a recorded answer or call the real gateway, depending on the mode."""
        if self.mode == "off":
            return self._inner.generate(
                task, messages, json_schema=json_schema, temperature=temperature, max_tokens=max_tokens, **extra
            )
        key = self._key(task, messages, json_schema, temperature)
        entry = self._store.get(key)
        if entry is not None and self.mode in ("auto", "replay"):
            self.hits += 1
            parsed = json_schema.model_validate_json(entry["payload"]) if json_schema else None
            return SimpleNamespace(
                parsed=parsed,
                text=entry["payload"],
                provider=f"cache:{entry.get('provider', '?')}",
                model=entry.get("model", "?"),
                attempts=[],
            )
        if self.mode == "replay":
            self.misses += 1
            raise LookupError("no recorded answer for this prompt (replay mode)")
        result = self._inner.generate(
            task, messages, json_schema=json_schema, temperature=temperature, max_tokens=max_tokens, **extra
        )
        parsed = getattr(result, "parsed", None)
        payload = parsed.model_dump_json() if isinstance(parsed, BaseModel) else (getattr(result, "text", "") or "")
        self._store[key] = {
            "payload": payload,
            "provider": str(getattr(result, "provider", "?")),
            "model": str(getattr(result, "model", "?")),
        }
        try:
            self._save()
        except OSError as exc:
            logger.warning("Could not write the LLM cache (%s).", type(exc).__name__)
        self.misses += 1
        return result
