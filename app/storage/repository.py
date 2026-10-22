"""Storage repository — all database operations for Solace.

All mutations go through these functions.  Direct ORM model usage is
restricted to this module.

Important: delete_user_data removes ALL rows for a given user across
every table — this is the primary privacy mechanism.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Generator

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.contracts import Message, TurnTrace, UserProfile
from app.storage.models import (
    Base,
    MessageRow,
    ProfileRow,
    SessionRow,
    TurnTraceRow,
    UserRow,
    create_all,
)

logger = logging.getLogger(__name__)


# ── Engine / session factory ──────────────────────────────────────────────────


def build_engine(database_url: str) -> Any:
    """Create and return a SQLAlchemy engine; creates tables if needed.

    Args:
        database_url: SQLAlchemy connection string (e.g. 'sqlite:///./data/solace.db').
    """
    connect_args = {}
    if database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False

    engine = create_engine(database_url, connect_args=connect_args, echo=False)
    create_all(engine)
    return engine


def make_session_factory(engine: Any) -> sessionmaker:
    """Return a session factory bound to *engine*."""
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)


@contextmanager
def db_session(factory: sessionmaker) -> Generator[Session, None, None]:
    """Context manager that yields a Session and commits / rolls back."""
    session: Session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ── Default engine (lazy) ─────────────────────────────────────────────────────

_default_engine: Any = None
_default_factory: sessionmaker | None = None


def _get_default_factory() -> sessionmaker:
    global _default_engine, _default_factory
    if _default_factory is None:
        from app.config import settings  # local import avoids circular dep
        _default_engine = build_engine(settings.database_url)
        _default_factory = make_session_factory(_default_engine)
    return _default_factory


# ── User ──────────────────────────────────────────────────────────────────────


def create_user(
    user_id: str,
    display_name: str | None = None,
    *,
    factory: sessionmaker | None = None,
) -> UserRow:
    """Insert a new user row.  Returns the created row.

    Args:
        user_id: Unique identifier for the user.
        display_name: Optional human-readable name.
        factory: Session factory to use (defaults to the app-wide factory).
    """
    factory = factory or _get_default_factory()
    row = UserRow(id=user_id, display_name=display_name)
    with db_session(factory) as session:
        session.add(row)
        session.flush()
        session.refresh(row)
    return row


def get_user(user_id: str, *, factory: sessionmaker | None = None) -> UserRow | None:
    """Fetch a user by ID or return None."""
    factory = factory or _get_default_factory()
    with db_session(factory) as session:
        return session.get(UserRow, user_id)


# ── Session ───────────────────────────────────────────────────────────────────


def create_session(
    session_id: str,
    user_id: str,
    *,
    factory: sessionmaker | None = None,
) -> SessionRow:
    """Create a new counselling session row.

    Args:
        session_id: Unique session identifier.
        user_id: Must refer to an existing user.
        factory: Session factory.
    """
    factory = factory or _get_default_factory()
    row = SessionRow(id=session_id, user_id=user_id)
    with db_session(factory) as session:
        session.add(row)
        session.flush()
        session.refresh(row)
    return row


def get_session(session_id: str, *, factory: sessionmaker | None = None) -> SessionRow | None:
    """Fetch a session row by ID or return None."""
    factory = factory or _get_default_factory()
    with db_session(factory) as session:
        return session.get(SessionRow, session_id)


# ── Messages ──────────────────────────────────────────────────────────────────


def append_message(
    session_id: str,
    message: Message,
    *,
    factory: sessionmaker | None = None,
) -> MessageRow:
    """Append a message to a session.

    Args:
        session_id: The session to append to.
        message: A contracts.Message instance.
        factory: Session factory.
    """
    factory = factory or _get_default_factory()
    row = MessageRow(
        session_id=session_id,
        role=message.role,
        content=message.content,
        speaker_role=message.speaker_role.value if message.speaker_role else None,
        timestamp=message.timestamp,
    )
    with db_session(factory) as session:
        session.add(row)
        session.flush()
        session.refresh(row)
    return row


def get_recent_messages(
    session_id: str,
    limit: int = 20,
    *,
    factory: sessionmaker | None = None,
) -> list[Message]:
    """Return the *limit* most recent messages for a session as contract objects.

    Args:
        session_id: Target session.
        limit: Maximum number of messages to return.
        factory: Session factory.
    """
    factory = factory or _get_default_factory()
    with db_session(factory) as session:
        rows = session.scalars(
            select(MessageRow)
            .where(MessageRow.session_id == session_id)
            .order_by(MessageRow.timestamp.desc())
            .limit(limit)
        ).all()
    rows = list(reversed(rows))  # return chronological order
    result: list[Message] = []
    for r in rows:
        from app.contracts import Role  # local import to avoid top-level cycle
        result.append(
            Message(
                role=r.role,
                content=r.content,
                speaker_role=Role(r.speaker_role) if r.speaker_role else None,
                timestamp=r.timestamp,
            )
        )
    return result


# ── Turn Traces ───────────────────────────────────────────────────────────────


def save_trace(
    trace: TurnTrace,
    *,
    factory: sessionmaker | None = None,
) -> TurnTraceRow:
    """Persist a TurnTrace to the database.

    Args:
        trace: A contracts.TurnTrace instance.
        factory: Session factory.
    """
    factory = factory or _get_default_factory()
    row = TurnTraceRow(
        session_id=trace.session_id,
        turn_id=trace.turn_id,
        data=trace.model_dump(mode="json"),
    )
    with db_session(factory) as session:
        session.add(row)
        session.flush()
        session.refresh(row)
    return row


def get_trace(turn_id: str, *, factory: sessionmaker | None = None) -> TurnTrace | None:
    """Retrieve a TurnTrace by turn_id or return None."""
    factory = factory or _get_default_factory()
    with db_session(factory) as session:
        row = session.scalars(
            select(TurnTraceRow).where(TurnTraceRow.turn_id == turn_id)
        ).first()
        if row is None:
            return None
        return TurnTrace.model_validate(row.data)


# ── Profiles ──────────────────────────────────────────────────────────────────


def get_profile(
    user_id: str, *, factory: sessionmaker | None = None
) -> UserProfile | None:
    """Load the UserProfile for a user or return None if none exists."""
    factory = factory or _get_default_factory()
    with db_session(factory) as session:
        row = session.scalars(
            select(ProfileRow).where(ProfileRow.user_id == user_id)
        ).first()
        if row is None:
            return None
        return UserProfile.model_validate(row.data)


def save_profile(
    profile: UserProfile, *, factory: sessionmaker | None = None
) -> ProfileRow:
    """Upsert a UserProfile for the given user.

    Args:
        profile: The UserProfile contract object.
        factory: Session factory.
    """
    factory = factory or _get_default_factory()
    with db_session(factory) as session:
        row = session.scalars(
            select(ProfileRow).where(ProfileRow.user_id == profile.user_id)
        ).first()
        if row is None:
            row = ProfileRow(
                user_id=profile.user_id,
                data=profile.model_dump(mode="json"),
            )
            session.add(row)
        else:
            row.data = profile.model_dump(mode="json")
        session.flush()
        session.refresh(row)
    return row


# ── Privacy: delete all user data ─────────────────────────────────────────────


def delete_user_data(
    user_id: str, *, factory: sessionmaker | None = None
) -> dict[str, int]:
    """Remove ALL data for a user from every table.

    Deletion order respects FK constraints:
    MessageRow → TurnTraceRow → SessionRow → ProfileRow → UserRow

    Returns a dict of table-name → rows deleted.

    Args:
        user_id: The user whose data should be erased.
        factory: Session factory.
    """
    factory = factory or _get_default_factory()
    counts: dict[str, int] = {}

    with db_session(factory) as session:
        # Collect session IDs for this user
        session_ids = list(
            session.scalars(
                select(SessionRow.id).where(SessionRow.user_id == user_id)
            ).all()
        )

        # Delete messages
        msg_deleted = 0
        for sid in session_ids:
            rows = session.scalars(
                select(MessageRow).where(MessageRow.session_id == sid)
            ).all()
            for r in rows:
                session.delete(r)
                msg_deleted += 1
        counts["messages"] = msg_deleted

        # Delete turn traces
        trace_deleted = 0
        for sid in session_ids:
            rows = session.scalars(
                select(TurnTraceRow).where(TurnTraceRow.session_id == sid)
            ).all()
            for r in rows:
                session.delete(r)
                trace_deleted += 1
        counts["turn_traces"] = trace_deleted

        # Delete sessions
        sess_deleted = 0
        for sid in session_ids:
            row = session.get(SessionRow, sid)
            if row:
                session.delete(row)
                sess_deleted += 1
        counts["sessions"] = sess_deleted

        # Delete profile
        profile_row = session.scalars(
            select(ProfileRow).where(ProfileRow.user_id == user_id)
        ).first()
        if profile_row:
            session.delete(profile_row)
            counts["profiles"] = 1
        else:
            counts["profiles"] = 0

        # Delete user
        user_row = session.get(UserRow, user_id)
        if user_row:
            session.delete(user_row)
            counts["users"] = 1
        else:
            counts["users"] = 0

    logger.info("Deleted all data for user (rows: %s)", counts)
    return counts
