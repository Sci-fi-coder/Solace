"""Fixed, non-generative safety messages and reviewed helpline metadata."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TypedDict

import yaml

from app.contracts import SafetyLevel

logger = logging.getLogger(__name__)

PROCEED = "proceed"
SOFT_CHECK = "soft_check"
CRISIS_OVERRIDE = "crisis_override"

CRISIS_TEMPLATE = (
    "I'm really sorry you're carrying this right now. Please reach out to someone you trust and "
    "stay with them if you can. If you may act soon or are in immediate danger, contact your local "
    "emergency services now. To find a current crisis helpline in your area, visit "
    "https://findahelpline.com/."
)
ELEVATED_CHECK_IN = (
    "You don't have to handle this alone. If it feels helpful, consider reaching out to someone "
    "you trust or a local support line. You can find local options at "
    "https://findahelpline.com/. We can take this one step at a time."
)


class Helpline(TypedDict):
    """Reviewed helpline entry; unverified entries intentionally have no usable number."""

    name: str
    number: str
    hours: str
    region: str
    source_url: str
    verified: bool


def action_for(level: SafetyLevel) -> str:
    """Return the pipeline action for a safety level."""
    if level == SafetyLevel.crisis:
        return CRISIS_OVERRIDE
    if level == SafetyLevel.elevated:
        return SOFT_CHECK
    return PROCEED


def response_template(level: SafetyLevel) -> str | None:
    """Return a fixed reply only for crisis; elevated concern remains in the pipeline."""
    if level == SafetyLevel.crisis:
        return CRISIS_TEMPLATE
    if level == SafetyLevel.elevated:
        return ELEVATED_CHECK_IN
    return None


def load_helplines() -> list[Helpline]:
    """Load helpline metadata and warn once per unverified entry."""
    path = Path(__file__).resolve().parents[2] / "data" / "safety" / "helplines.yaml"
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("helplines"), list):
        raise ValueError("Helpline data must contain a 'helplines' list.")
    entries: list[Helpline] = []
    for item in payload["helplines"]:
        if not isinstance(item, dict):
            raise ValueError("Each helpline entry must be a mapping.")
        name = item.get("name")
        number = item.get("number")
        hours = item.get("hours")
        region = item.get("region")
        source_url = item.get("source_url")
        if (
            not isinstance(name, str)
            or not isinstance(number, str)
            or not isinstance(hours, str)
            or not isinstance(region, str)
            or not isinstance(source_url, str)
        ):
            raise ValueError("Helpline name, number, hours, region, and source_url must be strings.")
        verified = item.get("verified", False)
        if not isinstance(verified, bool):
            raise ValueError("Helpline 'verified' must be a boolean.")
        entry = Helpline(
            name=name,
            number=number,
            hours=hours,
            region=region,
            source_url=source_url,
            verified=verified,
        )
        if not entry["verified"]:
            logger.warning("Helpline entry is unverified: %s (%s)", entry["name"], entry["region"])
        entries.append(entry)
    return entries
