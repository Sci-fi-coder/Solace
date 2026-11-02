"""Talk to Solace in the terminal (real providers from .env). Use synthetic text only.

Commands: /state  what Solace understands so far   /why  why it replied that way
          /sources  cards used   /end  save the session summary and quit   /quit
"""

from __future__ import annotations

import argparse
import logging

from app.pipeline.factory import build_engine_from_settings


def main() -> None:
    """Interactive loop."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--user", default="cli-user")
    parser.add_argument("--name", default=None)
    parser.add_argument("--emotion", action="store_true", help="enable the GoEmotions hint (needs the [ml] extra)")
    parser.add_argument("--retriever", choices=["auto", "keyword", "chroma"], default="auto")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.INFO if args.debug else logging.ERROR)

    engine = build_engine_from_settings(use_emotion=args.emotion or None, retriever_mode=args.retriever)
    info = engine.start_session(args.user, args.name)
    print(f"\nSolace> {info.greeting}\n(synthetic text only; free-tier providers may log requests)\n")
    last = None
    while True:
        try:
            text = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text in ("/quit", "/end"):
            if text == "/end":
                print(engine.end_session(info.session_id))
            break
        if text == "/state":
            print(last.state_summary if last else "(nothing yet)")
            continue
        if text == "/why":
            print(last.explanation if last else "(nothing yet)")
            continue
        if text == "/sources":
            print([c.model_dump() for c in last.citations] if last else "(nothing yet)")
            continue
        last = engine.handle_turn(info.session_id, text)
        if last.handoff:
            print(f"  [{last.handoff}]")
        who = last.speaker_name or "Solace"
        flag = "  [degraded]" if last.degraded else ""
        print(f"\n{who}> {last.reply}\n  (phase: {last.phase.value}, move: {last.move}, "
              f"safety: {last.safety_level.value}, {last.latency_ms:.0f} ms){flag}\n")
        if last.degraded or args.debug:
            print("  [steps] " + " | ".join(last.debug_steps) + "\n")


if __name__ == "__main__":
    main()
