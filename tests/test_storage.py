"""Tests for app/storage — CRUD operations and delete_user_data."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy.orm import sessionmaker

from app.contracts import Message, Role, SafetyLevel, SafetyResult, TurnTrace, UserProfile
from app.storage.models import create_all, Base
from app.storage.repository import (
    append_message,
    build_engine,
    create_session,
    create_user,
    delete_user_data,
    get_profile,
    get_recent_messages,
    get_session,
    get_trace,
    get_user,
    make_session_factory,
    save_profile,
    save_trace,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def factory() -> sessionmaker:
    """In-memory SQLite session factory — isolated per test."""
    engine = build_engine("sqlite:///:memory:")
    return make_session_factory(engine)


def _uid() -> str:
    return f"u-{uuid.uuid4().hex[:8]}"


def _sid() -> str:
    return f"s-{uuid.uuid4().hex[:8]}"


def _tid() -> str:
    return f"t-{uuid.uuid4().hex[:8]}"


def _safety() -> SafetyResult:
    return SafetyResult(level=SafetyLevel.none, source="rule", action="continue")


# ── User CRUD ─────────────────────────────────────────────────────────────────


def test_create_and_get_user(factory: sessionmaker) -> None:
    """create_user persists a user; get_user retrieves it."""
    uid = _uid()
    row = create_user(uid, display_name="Alice", factory=factory)
    assert row.id == uid
    assert row.display_name == "Alice"

    fetched = get_user(uid, factory=factory)
    assert fetched is not None
    assert fetched.display_name == "Alice"


def test_get_user_missing_returns_none(factory: sessionmaker) -> None:
    """get_user returns None for unknown user IDs."""
    result = get_user("no-such-user", factory=factory)
    assert result is None


# ── Session CRUD ──────────────────────────────────────────────────────────────


def test_create_and_get_session(factory: sessionmaker) -> None:
    """create_session persists a session; get_session retrieves it."""
    uid = _uid()
    create_user(uid, factory=factory)
    sid = _sid()
    row = create_session(sid, uid, factory=factory)
    assert row.id == sid
    assert row.user_id == uid

    fetched = get_session(sid, factory=factory)
    assert fetched is not None
    assert fetched.phase == "checkin"


# ── Messages ──────────────────────────────────────────────────────────────────


def test_append_and_get_messages(factory: sessionmaker) -> None:
    """Messages are appended and returned in chronological order."""
    uid = _uid()
    create_user(uid, factory=factory)
    sid = _sid()
    create_session(sid, uid, factory=factory)

    msgs = [
        Message(role="user", content="Hello there"),
        Message(role="assistant", content="How can I help?", speaker_role=Role.counsellor),
        Message(role="user", content="I feel anxious"),
    ]
    for m in msgs:
        append_message(sid, m, factory=factory)

    retrieved = get_recent_messages(sid, limit=10, factory=factory)
    assert len(retrieved) == 3
    assert retrieved[0].role == "user"
    assert retrieved[0].content == "Hello there"
    assert retrieved[1].speaker_role == Role.counsellor


def test_get_recent_messages_limit(factory: sessionmaker) -> None:
    """get_recent_messages respects the limit parameter."""
    uid = _uid()
    create_user(uid, factory=factory)
    sid = _sid()
    create_session(sid, uid, factory=factory)

    for i in range(10):
        append_message(sid, Message(role="user", content=f"msg {i}"), factory=factory)

    retrieved = get_recent_messages(sid, limit=3, factory=factory)
    assert len(retrieved) == 3
    # Should be the 3 most recent
    assert retrieved[-1].content == "msg 9"


# ── Turn Traces ───────────────────────────────────────────────────────────────


def test_save_and_get_trace(factory: sessionmaker) -> None:
    """TurnTraces are saved and retrieved correctly."""
    uid = _uid()
    create_user(uid, factory=factory)
    sid = _sid()
    create_session(sid, uid, factory=factory)
    tid = _tid()

    trace = TurnTrace(
        turn_id=tid,
        session_id=sid,
        user_message="synthetic test",
        safety=_safety(),
        final_reply="I hear you.",
        explanation="test explanation",
        provider_used="mock/mock-model",
    )
    save_trace(trace, factory=factory)

    fetched = get_trace(tid, factory=factory)
    assert fetched is not None
    assert fetched.turn_id == tid
    assert fetched.final_reply == "I hear you."


def test_get_trace_missing_returns_none(factory: sessionmaker) -> None:
    """get_trace returns None for unknown turn IDs."""
    result = get_trace("no-such-turn", factory=factory)
    assert result is None


# ── Profiles ──────────────────────────────────────────────────────────────────


def test_save_and_get_profile(factory: sessionmaker) -> None:
    """UserProfile is upserted and retrieved correctly."""
    uid = _uid()
    create_user(uid, factory=factory)

    profile = UserProfile(user_id=uid, display_name="Bob", preferred_role=Role.mentor)
    save_profile(profile, factory=factory)

    fetched = get_profile(uid, factory=factory)
    assert fetched is not None
    assert fetched.display_name == "Bob"
    assert fetched.preferred_role == Role.mentor


def test_upsert_profile(factory: sessionmaker) -> None:
    """save_profile updates an existing profile."""
    uid = _uid()
    create_user(uid, factory=factory)

    p1 = UserProfile(user_id=uid, display_name="Carol")
    save_profile(p1, factory=factory)

    p2 = UserProfile(user_id=uid, display_name="Carol Updated")
    save_profile(p2, factory=factory)

    fetched = get_profile(uid, factory=factory)
    assert fetched is not None
    assert fetched.display_name == "Carol Updated"


def test_get_profile_missing_returns_none(factory: sessionmaker) -> None:
    """get_profile returns None when no profile exists."""
    uid = _uid()
    create_user(uid, factory=factory)
    result = get_profile(uid, factory=factory)
    assert result is None


# ── delete_user_data ──────────────────────────────────────────────────────────


def test_delete_user_data_removes_all_rows(factory: sessionmaker) -> None:
    """delete_user_data erases all rows for a user across all tables."""
    uid = _uid()
    create_user(uid, display_name="To Delete", factory=factory)
    sid = _sid()
    create_session(sid, uid, factory=factory)
    append_message(sid, Message(role="user", content="private message"), factory=factory)

    tid = _tid()
    trace = TurnTrace(
        turn_id=tid,
        session_id=sid,
        user_message="private",
        safety=_safety(),
        final_reply="ok",
        explanation="test",
        provider_used="mock/mock",
    )
    save_trace(trace, factory=factory)
    save_profile(UserProfile(user_id=uid, display_name="To Delete"), factory=factory)

    counts = delete_user_data(uid, factory=factory)
    assert counts["users"] == 1
    assert counts["sessions"] == 1
    assert counts["messages"] >= 1
    assert counts["turn_traces"] >= 1
    assert counts["profiles"] == 1

    # Verify nothing remains
    assert get_user(uid, factory=factory) is None
    assert get_session(sid, factory=factory) is None
    assert get_trace(tid, factory=factory) is None
    assert get_profile(uid, factory=factory) is None
    assert get_recent_messages(sid, factory=factory) == []


def test_delete_user_data_unknown_user(factory: sessionmaker) -> None:
    """delete_user_data on a non-existent user returns zeros without error."""
    counts = delete_user_data("no-such-user", factory=factory)
    assert counts["users"] == 0
    assert counts["sessions"] == 0


def test_delete_user_data_does_not_affect_other_users(factory: sessionmaker) -> None:
    """delete_user_data removes only the target user's data."""
    uid_a = _uid()
    uid_b = _uid()
    create_user(uid_a, factory=factory)
    create_user(uid_b, display_name="Keep me", factory=factory)

    delete_user_data(uid_a, factory=factory)

    assert get_user(uid_b, factory=factory) is not None
