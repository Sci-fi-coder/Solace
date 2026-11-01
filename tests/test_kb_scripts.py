"""Tests for local PDF extraction, card citation checks, and cache behavior."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import fitz
import pytest
import yaml

from app.contracts import TaskKind
from app.rag.cards import Card
from scripts import kb_draft_cards, kb_validate
from scripts.kb_draft_cards import (
    DenySection,
    DraftBatch,
    PageText,
    QuotaExhausted,
    SourceRecord,
    cache_key,
    citation_review_notes,
    draft_source,
    parse_pages,
    select_pages,
    write_cards,
)
from scripts.kb_extract import extract_pdf
from tests.test_rag_cards import valid_card_payload


class CountingGateway:
    """Return a deterministic structured response without making an API call."""

    def __init__(self, batch: DraftBatch) -> None:
        self.batch = batch
        self.calls = 0

    def generate(
        self,
        task: TaskKind,
        messages: list[dict[str, str]],
        *,
        json_schema: type,
        temperature: float,
        max_tokens: int,
    ) -> SimpleNamespace:
        """Return the injected card batch and count the logical gateway request."""
        assert task == TaskKind.extract
        assert messages[0]["role"] == "system"
        assert json_schema is DraftBatch
        assert temperature == 0.2
        assert max_tokens == 5000
        self.calls += 1
        return SimpleNamespace(
            parsed=self.batch,
            provider="test-double",
            model="synthetic",
            usage={"prompt_tokens": 10, "completion_tokens": 20},
        )


def test_pdf_extraction_preserves_pages_and_reports_blank_pages(tmp_path: Path) -> None:
    """A synthetic PDF is extracted with one-based page numbers and blank-page reporting."""
    pdf_path = tmp_path / "synthetic.pdf"
    document = fitz.open()
    first_page = document.new_page()
    first_page.insert_text((72, 72), "Synthetic grounding text.")
    document.new_page()
    document.save(pdf_path)
    document.close()

    empty_pages = extract_pdf("synthetic", pdf_path, tmp_path / "work")
    extracted = json.loads((tmp_path / "work" / "synthetic.json").read_text(encoding="utf-8"))

    assert empty_pages == [2]
    assert extracted == [
        {"page_number": 1, "text": "Synthetic grounding text.\n"},
        {"page_number": 2, "text": ""},
    ]


def test_page_selection_parses_one_based_ranges_and_rejects_bad_ranges() -> None:
    """Page selectors accept inclusive ranges and reject indexes outside the PDF."""
    assert parse_pages("2-4,7", 8) == [2, 3, 4, 7]

    with pytest.raises(ValueError, match="outside"):
        parse_pages("0-2", 8)


def test_page_selection_strips_contact_lines_but_keeps_safe_text() -> None:
    """URL-bearing footer lines are removed while sufficiently long safe prose remains."""
    safe_text = (
        "Grounding can help with stress by bringing attention back to the present. "
        "Notice a sound, a color, or the feeling of your feet on the floor. "
        "Then choose one small action that feels manageable and repeat the exercise "
        "whenever attention drifts. "
    ) * 2
    page = PageText(
        1,
        safe_text
        + "\nFind more at https://example.org/stress\n"
        + "Try noticing one useful detail.",
    )

    selected, _ = select_pages(
        [page],
        source_id="nimh_stress",
        selection=None,
    )

    assert len(selected) == 1
    assert "Grounding can help with stress by" in selected[0].text
    assert "https://" not in selected[0].text
    assert "Try noticing one useful detail." in selected[0].text


def test_page_sanitisation_keeps_other_banned_validator_matches() -> None:
    """Page sanitisation removes only requested line categories, not whole pages."""
    prose = (
        "People can practise a small grounding exercise during ordinary stress. "
        "Notice a few details in the room and return attention to the present moment. "
    ) * 2
    page = PageText(
        1,
        prose
        + "\nCall 112 for immediate help."
        + "\nVisit https://example.org."
        + "\nAsk about sertraline 10 mg."
        + "\nA helpline can provide support."
        + "\nSelf-harm is discussed in the restricted clinical section.",
    )

    selected, excluded = select_pages(
        [page],
        source_id="synthetic",
        selection=None,
    )

    assert [item.page_number for item in selected] == [1]
    assert "Self-harm is discussed" in selected[0].text
    assert not any(term in selected[0].text for term in ("112", "https:", "sertraline", "helpline"))
    assert excluded == {}


def test_page_selection_reports_deny_list_and_too_short_reasons() -> None:
    """Deny-listed sections and short sanitized pages have separate explanations."""
    long_text = (
        "Communication skills help people share concerns with respect. "
        "Listen carefully and ask a clear, open question. "
    ) * 3
    pages = [
        PageText(1, long_text),
        PageText(2, "A short page with a useful idea."),
        PageText(3, long_text),
    ]

    selected, excluded = select_pages(
        pages,
        source_id="synthetic",
        selection=None,
        include_pages="1",
        deny_sections=(DenySection("restricted section", "3"),),
    )

    assert [page.page_number for page in selected] == [1]
    assert "too short after sanitisation" in excluded[2]
    assert excluded[3] == "deny list: restricted section"


def test_dry_run_prints_selected_pages_and_each_exclusion_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI dry run explains selected, short, and deny-listed pages."""
    source = SourceRecord(
        "synthetic",
        "Synthetic source",
        "synthetic.pdf",
        include_pages="1",
        deny_sections=(DenySection("restricted section", "3"),),
    )
    long_text = (
        "Communication skills help people share concerns with respect. "
        "Listen carefully and ask a clear, open question. "
    ) * 3
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "synthetic.json").write_text(
        json.dumps(
            [
                {"page_number": 1, "text": long_text},
                {"page_number": 2, "text": "A short page."},
                {"page_number": 3, "text": long_text},
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(kb_draft_cards, "load_sources", lambda: [source])
    monkeypatch.setattr(kb_draft_cards, "WORK_DIR", work_dir)
    monkeypatch.setattr(kb_draft_cards, "CARDS_DIR", tmp_path / "cards")
    monkeypatch.setattr(kb_draft_cards, "_existing_cards", lambda _path: ({}, {}))
    monkeypatch.setattr("sys.argv", ["kb_draft_cards", "--dry-run"])

    assert kb_draft_cards.main() == 0
    output = capsys.readouterr().out

    assert "synthetic: 1 selected PDF pages (1)" in output
    assert "2: too short after sanitisation" in output
    assert "3: deny list: restricted section" in output


def test_failure_output_lists_redacted_attempts_without_default_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Default failures are one-line and include safe per-provider diagnostics."""
    monkeypatch.setattr(kb_draft_cards, "_FAILURE_SECRETS", ["synthetic-secret"])

    class GatewayFailure(Exception):
        attempts = [
            {
                "provider": "groq",
                "model": "openai/gpt-oss-20b",
                "outcome": "failed",
                "error_summary": "BadRequestError: synthetic-secret json_validate_failed",
            }
        ]

    kb_draft_cards._print_failure(GatewayFailure("failed\nwith a secret synthetic-secret"), debug=False)
    output = capsys.readouterr().out

    assert "ERROR: GatewayFailure: failed with a secret [REDACTED]" in output
    assert "provider=groq model=openai/gpt-oss-20b outcome=failed" in output
    assert "json_validate_failed" in output
    assert "synthetic-secret" not in output
    assert "Traceback" not in output


def test_failure_output_prints_traceback_only_in_debug_mode(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The debug flag is the only mode that prints a traceback."""
    try:
        raise RuntimeError("synthetic failure")
    except RuntimeError as exc:
        kb_draft_cards._print_failure(exc, debug=True)

    assert "Traceback (most recent call last)" in capsys.readouterr().out


def test_gateway_configuration_logs_provider_names_only(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Gateway setup reads the app settings object without printing credential values."""
    from app.llm.gateway import LLMGateway

    class SyntheticSettings:
        """Minimal settings substitute with one configured synthetic provider."""

        def provider_config(self, name: str) -> dict[str, object]:
            return {
                "api_key": "synthetic-secret" if name == "gemini" else "",
                "enabled": True,
            }

    monkeypatch.setattr(kb_draft_cards, "settings", SyntheticSettings())
    monkeypatch.setattr(kb_draft_cards, "_FAILURE_SECRETS", [])
    monkeypatch.setattr(
        LLMGateway,
        "from_settings",
        classmethod(lambda cls, engine: object()),
    )

    gateway, engine, _, has_provider = kb_draft_cards._create_gateway()
    engine.dispose()
    output = capsys.readouterr().out

    assert gateway is not None
    assert has_provider
    assert "gemini" in output
    assert "synthetic-secret" not in output


def test_citation_checks_flag_missing_pages_and_title_mismatch() -> None:
    """Bad citations remain visible as review notes rather than being discarded."""
    card = Card.model_validate(
        {
            **valid_card_payload(),
            "sources": [{"source_id": "dwm_stress", "pages": [7]}],
        }
    )
    issues = citation_review_notes(
        card,
        "dwm_stress",
        [PageText(1, "Grounding can help with stress.")],
        {1},
    )

    assert "cited PDF page 7 does not exist" in issues
    assert "fewer than two distinctive title words appear on cited pages" in issues


def test_draft_cache_prevents_repeated_gateway_calls(tmp_path: Path) -> None:
    """Identical source, pages, and prompt reuse the successful cached batch."""
    payload = valid_card_payload()
    card = Card.model_validate(payload)
    batch = DraftBatch(cards=[card])
    gateway = CountingGateway(batch)
    source = SourceRecord(
        id="dwm_stress",
        title="Doing What Matters in Times of Stress",
        local_filename="whodoingwhatmattersntimesstress.pdf",
    )
    source_page = PageText(
        4,
        "Grounding and stress cues help people notice surroundings. "
        "This skill can support a useful next choice.",
    )
    existing: dict[str, Card] = {}
    cache_dir = tmp_path / "cache"
    cards_dir = tmp_path / "cards"

    first = draft_source(
        gateway=gateway,
        gateway_error=RuntimeError,
        source=source,
        pages=[source_page],
        all_source_pages=[source_page],
        max_cards=3,
        existing=existing,
        cache_dir=cache_dir,
        cards_dir=cards_dir,
        include_seek_help=False,
        total_limit=30,
    )
    second = draft_source(
        gateway=gateway,
        gateway_error=RuntimeError,
        source=source,
        pages=[source_page],
        all_source_pages=[source_page],
        max_cards=3,
        existing=existing,
        cache_dir=cache_dir,
        cards_dir=cards_dir,
        include_seek_help=False,
        total_limit=30,
    )

    assert len(first[0]) == 1
    assert second[0] == []
    assert first[3] is False
    assert second[3] is True
    assert gateway.calls == 1
    assert len(list(cards_dir.glob("*.yaml"))) == 1


def test_invalid_citation_is_saved_with_needs_review_note(tmp_path: Path) -> None:
    """A missing source page is retained as a draft and explicitly marked for review."""
    payload = valid_card_payload()
    payload["sources"] = [{"source_id": "dwm_stress", "pages": [99]}]
    card = Card.model_validate(payload)
    existing: dict[str, Card] = {}

    saved, rejected = write_cards(
        DraftBatch(cards=[card]),
        source=SourceRecord(
            id="dwm_stress",
            title="Doing What Matters in Times of Stress",
            local_filename="whodoingwhatmattersntimesstress.pdf",
        ),
        pages=[PageText(4, "Grounding and stress cues.")],
        cards_dir=tmp_path / "cards",
        existing=existing,
        max_cards=1,
        existing_help_cards=0,
        max_total_cards=30,
        allowed_page_numbers={4},
    )

    assert len(saved) == 1
    assert rejected == []
    assert saved[0].reviewed is False
    assert "needs_review:" in saved[0].notes
    assert "cited PDF page 99 does not exist" in saved[0].notes


def test_copy_check_scans_all_source_pages_not_only_card_citations(tmp_path: Path) -> None:
    """A copied nine-word passage is rejected even when its page was not cited."""
    card = Card.model_validate(valid_card_payload())
    source_pages = [
        PageText(4, "Grounding under stress can be a useful skill."),
        PageText(5, card.summary),
    ]

    saved, rejected = write_cards(
        DraftBatch(cards=[card]),
        source=SourceRecord(
            id="dwm_stress",
            title="Doing What Matters in Times of Stress",
            local_filename="whodoingwhatmattersntimesstress.pdf",
        ),
        pages=source_pages,
        cards_dir=tmp_path / "cards",
        existing={},
        max_cards=1,
        existing_help_cards=0,
        max_total_cards=30,
        allowed_page_numbers={4},
    )

    assert saved == []
    assert len(rejected) == 1
    assert "nine or more consecutive source words" in rejected[0]


def test_quota_gateway_error_is_a_clean_resume_signal() -> None:
    """Gateway quota and rate-limit attempts are identified for cache-preserving stops."""

    class ExhaustedGatewayError(Exception):
        attempts = [{"error_summary": "RateLimitError: HTTP 429 resource_exhausted"}]

    class ExhaustedGateway:
        def generate(self, *_args: object, **_kwargs: object) -> SimpleNamespace:
            raise ExhaustedGatewayError("provider quota exhausted")

    source = SourceRecord(
        id="dwm_stress",
        title="Doing What Matters in Times of Stress",
        local_filename="whodoingwhatmattersntimesstress.pdf",
    )
    page = PageText(4, "Grounding and stress cues can support a useful next choice.")

    with pytest.raises(QuotaExhausted, match="quota"):
        draft_source(
            gateway=ExhaustedGateway(),
            gateway_error=ExhaustedGatewayError,
            source=source,
            pages=[page],
            all_source_pages=[page],
            max_cards=1,
            existing={},
            cache_dir=Path("unused-cache"),
            cards_dir=Path("unused-cards"),
            include_seek_help=False,
            total_limit=30,
        )


def test_cache_key_changes_with_source_pages_or_prompt() -> None:
    """Source identity, PDF pages, and exact prompt all participate in the cache hash."""
    baseline = cache_key("dwm_stress", [4, 5], "prompt")

    assert cache_key("mhgap_ig_v2", [4, 5], "prompt") != baseline
    assert cache_key("dwm_stress", [4, 6], "prompt") != baseline
    assert cache_key("dwm_stress", [4, 5], "changed prompt") != baseline


def test_validator_reports_duplicate_ids_and_banned_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The validator reports duplicate identifiers and rejects phone-number content."""
    cards_dir = tmp_path / "cards"
    cards_dir.mkdir()
    page_index = tmp_path / "work"
    page_index.mkdir()
    (page_index / "dwm_stress.json").write_text(
        json.dumps([{"page_number": 4, "text": "Grounding and stress cues."}]),
        encoding="utf-8",
    )
    monkeypatch.setattr(kb_validate, "WORK_DIR", page_index)
    payload = valid_card_payload()
    payload["steps"] = ["Call +1 800 555 1234 for help."]
    first = cards_dir / "grounding-under-stress.yaml"
    first.write_text(yaml.safe_dump(payload), encoding="utf-8")
    duplicate = cards_dir / "duplicate-copy.yaml"
    duplicate.write_text(yaml.safe_dump(payload), encoding="utf-8")

    result = kb_validate.validate_cards(cards_dir)
    output = capsys.readouterr().out

    assert result == 1
    assert "duplicate id grounding-under-stress" in output
    assert "banned content pattern (phone number)" in output
    assert "Counts by category:" in output
    assert "Emotion coverage" in output
