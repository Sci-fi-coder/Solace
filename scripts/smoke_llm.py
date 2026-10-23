"""Send a harmless synthetic prompt through the real LLM gateway.

Usage::

    python -m scripts.smoke_llm
    python -m scripts.smoke_llm --provider gemini --task converse
"""

from __future__ import annotations

import argparse
import logging
import sys


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse smoke-test options and reject unknown flags."""
    from app.contracts import TaskKind

    parser = argparse.ArgumentParser(description="Smoke-test the Solace LLM gateway.")
    parser.add_argument(
        "--provider",
        choices=("gemini", "groq", "openrouter", "mistral", "cerebras"),
        help="Call only this provider instead of its configured fallback chain.",
    )
    parser.add_argument(
        "--task",
        choices=tuple(task.value for task in TaskKind),
        default=TaskKind.converse.value,
        help="Task to run (defaults to one converse call).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run one synthetic gateway smoke test."""
    args = _parse_args(argv)
    try:
        from app.config import redact_secrets, settings
        from app.contracts import TaskKind
        from app.llm.gateway import GatewayExhaustedError, LLMGateway
        from app.storage.repository import build_engine
        from scripts.check_models import check_models
    except ImportError as exc:
        print(f"[smoke_llm] Import error: {exc}")
        print("  Did you run: pip install -e '.[dev]' ?")
        return 1

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )
    settings.log_configuration()
    print("Solace LLM Gateway - smoke test")
    print("=" * 50)

    check_models()
    gateway = LLMGateway.from_settings(engine=build_engine(settings.database_url))
    task = TaskKind(args.task)
    synthetic_messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant in a test environment. "
                "All data here is synthetic."
            ),
        },
        {
            "role": "user",
            "content": (
                "This is a synthetic test message. "
                "Please respond with exactly: 'Smoke test OK.'"
            ),
        },
    ]
    secrets = [
        str(settings.provider_config(name)["api_key"])
        for name in ("gemini", "groq", "openrouter", "mistral", "cerebras")
        if settings.provider_config(name)["api_key"]
    ]

    print(f"\n  Task: {task.value}")
    try:
        result = gateway.generate(
            task,
            synthetic_messages,
            max_tokens=50,
            provider=args.provider,
        )
        if result.attempts:
            entries = []
            for attempt in result.attempts:
                if attempt["outcome"] == "skipped":
                    entries.append(f"{attempt['provider']}: {attempt['error_summary']}")
                else:
                    entries.append(
                        f"{attempt['provider']}: failed ({attempt['error_summary']})"
                    )
            entries.append(f"served by {result.provider}")
            print(f"    attempts : {' -> '.join(entries)}")
        else:
            print(f"    attempts : served by {result.provider}")
        print(f"    provider : {result.provider}")
        print(f"    model    : {result.model}")
        print(f"    latency  : {result.latency_ms:.0f} ms")
        reply = redact_secrets(result.text[:80], secrets)
        print(f"    reply    : {reply!r}")
    except GatewayExhaustedError as exc:
        print(f"    FAILED   : {redact_secrets(str(exc), secrets)}")
        return 1
    except Exception as exc:
        print(
            f"    ERROR    : {type(exc).__name__}: "
            f"{redact_secrets(str(exc), secrets)}"
        )
        return 1

    print("\n" + "=" * 50)
    print("Smoke test PASSED - a provider responded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
