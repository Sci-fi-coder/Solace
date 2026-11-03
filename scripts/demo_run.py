"""Run a scripted, synthetic conversation through the engine and write a transcript.

  python -m scripts.demo_run --cache record     # live models, records every answer
  python -m scripts.demo_run --cache replay     # recorded answers only: no network, same transcript
Use a fresh profile and no name for both runs so the prompts (and cache keys) are identical.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

from app.pipeline.factory import build_engine_from_settings


def main() -> None:
    """Replay a script and save docs/demo_transcript*.md."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--script", default="docs/demo_script.txt")
    parser.add_argument("--out", default=None)
    parser.add_argument("--cache", choices=["off", "auto", "record", "replay"], default="auto")
    parser.add_argument("--retriever", choices=["auto", "keyword", "chroma"], default="auto")
    parser.add_argument("--pause", type=float, default=0.0, help="seconds between turns (rate limits)")
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.ERROR)
    os.environ["SOLACE_LLM_CACHE"] = args.cache

    lines = [
        ln.strip() for ln in Path(args.script).read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    engine = build_engine_from_settings(retriever_mode=args.retriever)
    user = f"demo-{int(time.time())}"
    info = engine.start_session(user)
    out_path = Path(args.out or f"docs/demo_transcript_{Path(args.script).stem}.md")
    md = [f"# Demo transcript ({Path(args.script).name}, cache={args.cache})", "",
          "Synthetic messages only.", "", f"**Solace:** {info.greeting}", ""]
    last = None
    for text in lines:
        out = engine.handle_turn(info.session_id, text)
        last = out
        who = out.speaker_name or "Solace (fixed safety message)"
        print(f"\nYou> {text}\n{who}> {out.reply}\n  [{out.phase.value} / {out.move} / safety {out.safety_level.value}"
              f" / {out.latency_ms:.0f} ms{' / DEGRADED' if out.degraded else ''}]")
        md += [f"**You:** {text}", ""]
        if out.handoff:
            md += [f"*{out.handoff}*", ""]
        md += [f"**{who}:** {out.reply}", "",
               f"<sub>phase: {out.phase.value} | move: {out.move} | safety: {out.safety_level.value} | "
               f"provider: {out.provider_used} | {out.latency_ms:.0f} ms</sub>", ""]
        if out.citations:
            md += ["Grounded in: " + "; ".join(f"{c.title} ({'; '.join(c.sources)})" for c in out.citations), ""]
        if args.pause:
            time.sleep(args.pause)
    if last is not None:
        md += ["## What Solace understood at the end", "", last.state_summary, "", "## Why the last reply", "",
               last.explanation, ""]
    out_path.write_text("\n".join(md), encoding="utf-8")
    engine.delete_user_data(user)
    print(f"\nTranscript written to {out_path} (demo profile deleted).")


if __name__ == "__main__":
    main()
