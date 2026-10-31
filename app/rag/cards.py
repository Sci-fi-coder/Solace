"""Pydantic schema and content checks for human-reviewed knowledge cards."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.emotion.recognizer import DEFAULT_LABEL_MAP

CardCategory = Literal[
    "technique",
    "psychoeducation",
    "communication_skill",
    "when_to_seek_help",
]
RoleAffinity = Literal["counsellor", "mentor", "teacher", "peer"]
COARSE_EMOTION_LABELS = frozenset(DEFAULT_LABEL_MAP.values())

_PHONE_RE = re.compile(
    r"(?<!\w)(?:\+\s?\d{1,3}[\s().-]*)?"
    r"(?:\(?\d{2,4}\)?[\s().-]*){1,3}\d{3,4}(?!\w)"
)
_EMERGENCY_NUMBER_RE = re.compile(r"(?<!\d)(?:112|911|988|999)(?!\d)")
_URL_RE = re.compile(
    r"(?i)\b(?:https?://|www\.)\S+|"
    r"\b[\w.-]+\.(?:com|org|net|gov|edu|int|io)\b"
)
_DOSAGE_RE = re.compile(
    r"(?i)\b\d+(?:\.\d+)?\s*(?:mg|mcg|μg|µg|g|ml|mL|"
    r"milligrams?|micrograms?|millilit(?:er|re)s?)\b"
)
_MEDICATION_RE = re.compile(
    r"(?i)\b(?:medications?|medicines?|pharmacological|pharmacotherapy|"
    r"antidepressants?|antipsychotics?|benzodiazepines?|"
    r"fluoxetine|sertraline|escitalopram|citalopram|paroxetine|venlafaxine|"
    r"duloxetine|amitriptyline|mirtazapine|bupropion|diazepam|alprazolam|"
    r"lorazepam|clonazepam|quetiapine|olanzapine|risperidone|aripiprazole|"
    r"lithium|valproate|carbamazepine|phenytoin|haloperidol|chlorpromazine|"
    r"ketamine|propranolol|hydroxyzine|melatonin|"
    r"[a-z]{4,}(?:oxetine|pram|zepam|zolam|prazole|sartan|olol|apine|idone|azepam))\b"
)
_HELPLINE_RE = re.compile(
    r"(?i)\b(?:helplines?|hotlines?|crisis lines?|lifelines?|samaritans|"
    r"befrienders?|crisis text lines?|suicide prevention lines?|"
    r"find ?a ?helpline)\b"
)
_DIAGNOSTIC_CRITERIA_RE = re.compile(
    r"(?i)\b(?:DSM(?:-?5(?:-TR)?)?|ICD(?:-?11)?|diagnostic criteria|"
    r"criteria for diagnosis|meets? (?:the )?criteria|diagnosed with)\b"
)
_METHOD_DETAIL_RE = re.compile(
    r"(?i)\b(?:how to|method(?:s)? for|ways to)\b.{0,60}"
    r"\b(?:suicid\w*|self[- ]harm|kill myself|end my life)\b|"
    r"\b(?:suicid\w*|self[- ]harm|kill myself|end my life)\b.{0,60}"
    r"\b(?:overdose|hanging|poison(?:ing)?|weapon|cutting|jumping)\b"
)
_BANNED_CONTENT_PATTERNS = (
    ("phone number", _PHONE_RE),
    ("emergency or helpline number", _EMERGENCY_NUMBER_RE),
    ("URL", _URL_RE),
    ("dosage", _DOSAGE_RE),
    ("medication name or reference", _MEDICATION_RE),
    ("helpline name or reference", _HELPLINE_RE),
    ("diagnostic criteria", _DIAGNOSTIC_CRITERIA_RE),
    ("self-harm or suicide method detail", _METHOD_DETAIL_RE),
)


class CardSource(BaseModel):
    """A source document and 1-based PDF page references for a card."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    source_id: str = Field(pattern=r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
    pages: list[int] = Field(min_length=1)

    @field_validator("pages")
    @classmethod
    def pages_are_positive(cls, value: list[int]) -> list[int]:
        """Require one-based page indexes and remove duplicate references."""
        if any(page < 1 for page in value):
            raise ValueError("source page numbers must be positive 1-based PDF indexes")
        return list(dict.fromkeys(value))


class Card(BaseModel):
    """A draft self-help or communication card requiring human review."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    title: str = Field(min_length=3, max_length=120)
    category: CardCategory
    summary: str = Field(min_length=1)
    when_to_use: list[str] = Field(min_length=1)
    when_not_to_use: list[str] = Field(min_length=1)
    steps: list[str] = Field(min_length=1, max_length=6)
    emotions: list[str] = Field(min_length=1)
    role_affinity: list[RoleAffinity] = Field(min_length=1)
    sources: list[CardSource] = Field(min_length=1)
    reviewed: bool = False
    notes: str = ""

    @field_validator("summary")
    @classmethod
    def summary_has_two_or_three_sentences(cls, value: str) -> str:
        """Require the card summary to contain two or three sentences."""
        sentence_ends = re.findall(r"[.!?](?:\s|$)", value)
        if len(sentence_ends) not in (2, 3):
            raise ValueError("summary must contain two or three sentences")
        return value

    @field_validator("emotions")
    @classmethod
    def emotions_are_recognizer_labels(cls, value: list[str]) -> list[str]:
        """Restrict emotion tags to the recognizer's mapped coarse labels."""
        unknown = set(value) - COARSE_EMOTION_LABELS
        if unknown:
            raise ValueError(f"unsupported coarse emotion labels: {sorted(unknown)}")
        return list(dict.fromkeys(value))


def banned_content_reasons(text: str) -> list[str]:
    """Return banned-content categories detected in card text."""
    return [label for label, pattern in _BANNED_CONTENT_PATTERNS if pattern.search(text)]


__all__ = [
    "COARSE_EMOTION_LABELS",
    "Card",
    "CardCategory",
    "CardSource",
    "RoleAffinity",
    "banned_content_reasons",
]
