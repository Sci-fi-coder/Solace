"""smoke_llm.py — Send one harmless synthetic prompt through the real LLM gateway.

Usage::

    python -m scripts.smoke_llm

Prints the provider, model, and latency for each TaskKind.
Uses only synthetic data — never real user content.

Exit codes:
    0 — at least one provider succeeded
    1 — all providers failed (check .env for API keys)
"""

from __future__ import annotations

import sys


def main() -> int:
    """Run the smoke test."""
    # Import here so errors surface cleanly.
    try:
        from app.config import settings
        from app.contracts import TaskKind
        from app.llm.gateway import GatewayExhaustedError, LLMGateway
        from app.storage.repository import build_engine
    except ImportError as exc:
        print(f"[smoke_llm] Import error: {exc}")
        print("  Did you run: pip install -e '.[dev]' ?")
        return 1

    print("Solace LLM Gateway — smoke test")
    print("=" * 50)

    gateway = LLMGateway.from_settings(engine=build_engine(settings.database_url))

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

    any_success = False

    for task in TaskKind:
        print(f"\n  Task: {task.value}")
        try:
            result = gateway.generate(task, synthetic_messages, max_tokens=50)
            print(f"    ✓ provider : {result.provider}")
            print(f"      model    : {result.model}")
            print(f"      latency  : {result.latency_ms:.0f} ms")
            print(f"      reply    : {result.text[:80]!r}")
            any_success = True
        except GatewayExhaustedError as exc:
            print(f"    ✗ All providers failed: {exc}")
        except Exception as exc:
            print(f"    ✗ Unexpected error: {exc}")

    print("\n" + "=" * 50)
    if any_success:
        print("Smoke test PASSED — at least one provider responded.")
        return 0
    else:
        print("Smoke test FAILED — no providers succeeded.")
        print("Add at least one API key to .env and retry.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
