"""SQLAlchemy 2.0 ORM models for Solace.

Tables:
    users           — registered users
    sessions        — counselling sessions
    messages        — individual messages within a session
    turn_traces     — JSON TurnTrace blobs, one per turn
    profiles        — user profile JSON blobs (1:1 with users)
    provider_usage  — per-provider per-day call counters
"""

from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""


class UserRow(Base):
    """Persistent user record."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    display_name: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    sessions: Mapped[list["SessionRow"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    profile: Mapped["ProfileRow | None"] = relationship(
        back_populates="user", cascade="all, delete-orphan", uselist=False
    )


class SessionRow(Base):
    """A single counselling session."""

    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    started_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    phase: Mapped[str] = mapped_column(String, default="checkin", nullable=False)

    user: Mapped["UserRow"] = relationship(back_populates="sessions")
    messages: Mapped[list["MessageRow"]] = relationship(
        back_populates="session", cascade="all, delete-orphan", order_by="MessageRow.timestamp"
    )
    turn_traces: Mapped[list["TurnTraceRow"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


class MessageRow(Base):
    """A single message in a session."""

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id", ondelete="CASCADE"))
    role: Mapped[str] = mapped_column(String, nullable=False)  # user|assistant|system
    content: Mapped[str] = mapped_column(Text, nullable=False)
    speaker_role: Mapped[str | None] = mapped_column(String, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    session: Mapped["SessionRow"] = relationship(back_populates="messages")


class TurnTraceRow(Base):
    """Serialised TurnTrace JSON blob for a single turn."""

    __tablename__ = "turn_traces"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id", ondelete="CASCADE"))
    turn_id: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    data: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    session: Mapped["SessionRow"] = relationship(back_populates="turn_traces")


class ProfileRow(Base):
    """User profile JSON blob."""

    __tablename__ = "profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), unique=True
    )
    data: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    user: Mapped["UserRow"] = relationship(back_populates="profile")


class ProviderUsageRow(Base):
    """Per-provider per-day call counter for soft daily cap enforcement."""

    __tablename__ = "provider_usage"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    provider: Mapped[str] = mapped_column(String, nullable=False)
    date: Mapped[str] = mapped_column(String, nullable=False)  # ISO date "YYYY-MM-DD"
    call_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    __table_args__ = (UniqueConstraint("provider", "date", name="uq_provider_date"),)


def create_all(engine: object) -> None:
    """Create all tables if they don't already exist."""
    Base.metadata.create_all(engine)  # type: ignore[arg-type]
