"""Validated models for separately sourced historical role-model cards."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Literal

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, field_validator, model_validator


class Person(StrEnum):
    """Supported historical figures; `kalam` names A.P.J. Abdul Kalam."""

    gandhi = "gandhi"
    vivekananda = "vivekananda"
    marcus_aurelius = "marcus_aurelius"
    kalam = "kalam"


VerificationStatus = Literal["needs_verification", "verified"]

SUPPORTED_TAGS = frozenset(
    {
        "anger",
        "calm",
        "failure",
        "fear",
        "grief",
        "hardship",
        "humiliation",
        "low_mood",
        "pressure",
        "purpose",
        "responsibility",
        "routine",
        "self_doubt",
        "uncertainty",
    }
)
SUPPORTED_EMOTIONS = frozenset(
    {"anger", "anxiety_fear", "neutral", "positive", "sadness", "shame_guilt"}
)
_URL_RE = re.compile(
    r"(?i)\b(?:[a-z][a-z0-9+.-]*://|www\.)\S+|"
    r"\b[\w.-]+\.(?:com|org|net|gov|edu|int|io)\b"
)


class RoleModelSource(BaseModel):
    """Bibliographic provenance and the source-specific locator for a card."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: str = Field(min_length=2, max_length=240)
    author: str | None = Field(default=None, max_length=240)
    year: str | None = Field(default=None, max_length=120)
    locator: str = Field(min_length=5, max_length=500)
    url: AnyHttpUrl | None = None


class Verification(BaseModel):
    """Human verification record; unverified drafts are never active by default."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    status: VerificationStatus = "needs_verification"
    checked_by: str | None = None
    notes: str = ""


class RoleModelCard(BaseModel):
    """A paraphrased historical analogy with explicit source provenance."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    person: Person
    title: str = Field(min_length=3, max_length=160)
    tags: list[str] = Field(min_length=1)
    emotions: list[str] = Field(min_length=1)
    summary: str = Field(min_length=1, max_length=2000)
    lesson: str = Field(min_length=1, max_length=1000)
    framing: str = Field(min_length=1, max_length=1000)
    avoid_when: str = Field(min_length=1, max_length=1000)
    sources: list[RoleModelSource] = Field(min_length=1)
    verification: Verification
    reviewed: bool = False

    @field_validator("tags")
    @classmethod
    def tags_are_supported(cls, value: list[str]) -> list[str]:
        """Reject vocabulary drift while preserving supplied tag wording."""
        normalized = list(dict.fromkeys(tag.lower() for tag in value))
        unknown = set(normalized) - SUPPORTED_TAGS
        if unknown:
            raise ValueError(f"unsupported role-model tags: {sorted(unknown)}")
        return normalized

    @field_validator("emotions")
    @classmethod
    def emotions_are_supported(cls, value: list[str]) -> list[str]:
        """Limit emotional matches to the existing coarse emotion vocabulary."""
        normalized = list(dict.fromkeys(emotion.lower() for emotion in value))
        unknown = set(normalized) - SUPPORTED_EMOTIONS
        if unknown:
            raise ValueError(f"unsupported role-model emotions: {sorted(unknown)}")
        return normalized

    @field_validator("summary")
    @classmethod
    def summary_has_no_urls(cls, value: str) -> str:
        """Keep web links in provenance rather than historical summaries."""
        if _URL_RE.search(value):
            raise ValueError("summary must not contain URLs")
        return value

    @model_validator(mode="after")
    def reviewed_cards_must_be_verified(self) -> RoleModelCard:
        """Require affirmative source verification before a card can be reviewed."""
        if self.reviewed and self.verification.status != "verified":
            raise ValueError("reviewed cards require verification.status='verified'")
        return self


class RoleModelProvenance(BaseModel):
    """Source information attached only when a response actually used a card."""

    model_config = ConfigDict(frozen=True)

    card_id: str
    person: Person
    title: str
    source_locators: list[str] = Field(min_length=1)
