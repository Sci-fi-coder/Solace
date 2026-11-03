# Demo reliability: record once, replay on the day

Free-tier models can be slow or overloaded. Solace can record real model answers for a scripted demo and
replay them later with no network. The UI/CLI label such turns `cache:<provider>` in the raw trace, so the
demo is honest about it. State this in the report.

1. `Add-Content .gitignore "data/cache/"` (once), then in `.env` leave `SOLACE_LLM_CACHE` unset (default `auto`).
2. With working API keys: `python -m scripts.demo_run --cache record` (and `--script docs/demo_safety_script.txt`).
   Read `docs/demo_transcript_demo_script.md`. If a turn is degraded or odd, fix the cause and record again.
3. Check: `python -m scripts.demo_run --cache replay` must give the same transcript with zero network.
4. On demo day use a FRESH profile and no name (Delete my data first): the cache key contains the whole prompt,
   so a stored profile or a different name changes the prompt and causes a miss.
5. Run the UI with `SOLACE_LLM_CACHE=replay` for a guaranteed-stable demo, or `auto` to allow new live answers.
