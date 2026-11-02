"""Streamlit interface for Solace sessions."""

from __future__ import annotations

import html
import os
from typing import Protocol

import streamlit as st

from app.pipeline.types import SessionInfo, TurnOutput

_PHASES = ("checkin", "exploration", "understanding", "guidance", "closing")
_DELETE_NOTICE = "Your data was deleted and the session has been reset."
_TURN_ERROR = "I couldn't process that message right now. Please try again."


class SessionEngine(Protocol):
    """The subset of the Solace engine used by the UI."""

    def start_session(self, user_id: str, display_name: str | None = None) -> SessionInfo:
        """Start a session for a user."""

    def handle_turn(self, session_id: str, text: str) -> TurnOutput:
        """Process one user message."""

    def end_session(self, session_id: str) -> dict[str, object]:
        """End a session and return its summary."""

    def delete_user_data(self, user_id: str) -> dict[str, int]:
        """Delete all stored data belonging to a user."""


@st.cache_resource
def get_engine() -> SessionEngine:
    """Return one cached demo or application engine."""
    if os.environ.get("SOLACE_UI_DEMO") == "1":
        from ui.demo_engine import DemoEngine

        return DemoEngine()

    from app.pipeline.factory import build_engine_from_settings

    return build_engine_from_settings()


def _value(value: object) -> str:
    """Return the plain value of an enum or string."""
    return str(getattr(value, "value", value)).lower()


def _initialize_state() -> None:
    """Initialize UI state without overwriting values from an earlier rerun."""
    st.session_state.setdefault("session_id", None)
    st.session_state.setdefault("session_user_id", None)
    st.session_state.setdefault("chat_messages", [])
    st.session_state.setdefault("latest_turn", None)
    st.session_state.setdefault("session_ended", False)
    st.session_state.setdefault("session_summary", None)
    st.session_state.setdefault("notice", None)


def _render_phase_tracker(phase: str) -> None:
    """Render the five session phases with the current phase highlighted."""
    phase = phase.lower()
    items = []
    for item in _PHASES:
        active = " active" if item == phase else ""
        label = html.escape(item.title())
        items.append(f'<span class="phase{active}">{label}</span>')
    st.markdown(
        "<style>"
        ".phase-tracker{display:flex;gap:.45rem;flex-wrap:wrap;margin:.4rem 0 1rem}"
        ".phase{border:1px solid #87909c;border-radius:1rem;padding:.25rem .7rem;"
        "color:#68717d;font-size:.85rem}.phase.active{background:#dcecff;"
        "border-color:#3978bd;color:#164b82;font-weight:600}"
        "</style>"
        f'<div class="phase-tracker">{"".join(items)}</div>',
        unsafe_allow_html=True,
    )


def _render_sidebar(engine: SessionEngine) -> None:
    """Render session controls and the fixed safety and privacy notice."""
    with st.sidebar:
        st.header("Session")
        st.info(
            "Solace is an AI companion, not a therapist, and cannot help in emergencies. "
            "Use synthetic text only while testing; free-tier AI providers may log requests."
        )
        if st.session_state.session_id is None:
            with st.form("start_session_form"):
                st.text_input("Name (optional)", key="start_name")
                st.text_input("Fake user ID", value="demo-user", key="start_user_id")
                submitted = st.form_submit_button("Start session", key="start_session")
            if submitted:
                user_id = st.session_state.start_user_id.strip()
                if not user_id:
                    st.error("Enter a user ID to start a session.")
                else:
                    try:
                        info = engine.start_session(
                            user_id, st.session_state.start_name.strip() or None
                        )
                    except Exception:
                        st.error("I couldn't start a session right now. Please try again.")
                    else:
                        st.session_state.session_id = info.session_id
                        st.session_state.session_user_id = user_id
                        st.session_state.chat_messages = [
                            {
                                "role": "assistant",
                                "content": info.greeting,
                                "speaker_name": "Solace",
                                "speaker_role": "counsellor",
                                "handoff": None,
                                "safety_level": "none",
                                "degraded": False,
                            }
                        ]
                        st.session_state.latest_turn = None
                        st.session_state.session_ended = False
                        st.session_state.session_summary = None
                        st.session_state.notice = None
                        st.rerun()
        else:
            st.caption(f"User ID: {st.session_state.session_user_id}")
            if (
                not st.session_state.session_ended
                and st.button("End session and save summary", key="end_session")
            ):
                try:
                    summary = engine.end_session(st.session_state.session_id)
                except Exception:
                    st.error("I couldn't save the session summary. Please try again.")
                else:
                    st.session_state.session_summary = summary
                    st.session_state.session_ended = True
                    st.rerun()

            confirmation_key = f"confirm_delete_{st.session_state.session_id}"
            confirmed = st.checkbox(
                "I understand this permanently deletes my data",
                key=confirmation_key,
            )
            if st.button("Delete all my data", key="delete_user_data"):
                if not confirmed:
                    st.warning("Confirm deletion before continuing.")
                else:
                    try:
                        engine.delete_user_data(st.session_state.session_user_id)
                    except Exception:
                        st.error("I couldn't delete the data right now. Please try again.")
                    else:
                        for key in (
                            "session_id",
                            "session_user_id",
                            "chat_messages",
                            "latest_turn",
                            "session_ended",
                            "session_summary",
                        ):
                            st.session_state.pop(key, None)
                        _initialize_state()
                        st.session_state.notice = _DELETE_NOTICE
                        st.rerun()


def _render_chat_message(message: dict[str, object]) -> None:
    """Render a user, assistant, or recoverable UI error message."""
    role = str(message["role"])
    if role == "error":
        st.error(str(message["content"]))
        return

    speaker_type = "user" if role == "user" else "assistant"
    with st.chat_message(speaker_type):
        if role == "assistant":
            handoff = message.get("handoff")
            if handoff:
                st.info(str(handoff))
            speaker_name = message.get("speaker_name") or "Solace"
            speaker_role = message.get("speaker_role")
            if speaker_role:
                st.caption(f"{speaker_name} · {str(speaker_role).title()}")

            if message.get("safety_level") == "crisis":
                content = html.escape(str(message["content"]))
                st.markdown(
                    '<div data-testid="crisis-reply" '
                    'style="border:2px solid #c62828;border-radius:.5rem;padding:1rem;'
                    'background:#fff5f5;white-space:pre-wrap">'
                    f"{content}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(str(message["content"]))
            if message.get("degraded"):
                st.caption("Limited mode")
        else:
            st.markdown(str(message["content"]))


def _render_panels(turn: TurnOutput) -> None:
    """Render emotional-state and explanation panels for a non-crisis turn."""
    if _value(turn.safety_level) == "crisis":
        return

    state_column, explanation_column = st.columns(2)
    with state_column:
        st.subheader("What I understand so far")
        st.write(turn.state_summary or "We are just getting started.")
        if turn.state is not None:
            hypotheses = sorted(
                turn.state.hypotheses, key=lambda item: item.probability, reverse=True
            )[:5]
            if hypotheses:
                chart_data = {
                    "Probability": {
                        item.label: item.probability for item in hypotheses
                    }
                }
                st.bar_chart(chart_data)
            st.caption(f"Confidence: {turn.state.confidence:.0%}")
            st.progress(turn.state.confidence)

    with explanation_column:
        st.subheader("Why I said this")
        st.write(turn.explanation or "A response was chosen for this point in the conversation.")
        st.caption(f"Phase: {_value(turn.phase).title()}")
        if turn.move:
            st.caption(f"Move: {turn.move.replace('_', ' ').title()}")
        if turn.citations:
            st.markdown("**Sources**")
            for citation in turn.citations:
                st.markdown(f"- **{citation.title}**")
                for source in citation.sources:
                    st.caption(source)


def _handle_submitted_message(engine: SessionEngine, text: str) -> None:
    """Record a user message and display either a turn or a friendly error."""
    cleaned = text.strip()
    if not cleaned:
        return
    st.session_state.chat_messages.append({"role": "user", "content": cleaned})
    try:
        with st.spinner("Solace is preparing a response..."):
            turn = engine.handle_turn(st.session_state.session_id, cleaned)
    except Exception:
        st.session_state.chat_messages.append(
            {"role": "error", "content": _TURN_ERROR}
        )
        return

    st.session_state.latest_turn = turn
    st.session_state.chat_messages.append(
        {
            "role": "assistant",
            "content": turn.reply,
            "speaker_name": turn.speaker_name,
            "speaker_role": _value(turn.speaker) if turn.speaker is not None else None,
            "handoff": turn.handoff,
            "safety_level": _value(turn.safety_level),
            "degraded": turn.degraded,
        }
    )


def main() -> None:
    """Run the Streamlit session UI."""
    st.set_page_config(page_title="Solace", page_icon="🌿", layout="wide")
    _initialize_state()
    engine = get_engine()

    st.title("Solace")
    st.caption("A structured, supportive conversation")
    if st.session_state.notice:
        st.success(st.session_state.notice)
        st.session_state.notice = None

    _render_sidebar(engine)

    latest_turn = st.session_state.latest_turn
    phase = _value(latest_turn.phase) if latest_turn is not None else "checkin"
    _render_phase_tracker(phase)

    session_id = st.session_state.session_id
    if session_id is not None and not st.session_state.session_ended:
        submitted = st.chat_input("Share what's on your mind", max_chars=2000)
        if submitted:
            _handle_submitted_message(engine, submitted)
    elif st.session_state.session_ended:
        st.info("This session has ended. Start a new session to continue.")

    for message in st.session_state.chat_messages:
        _render_chat_message(message)

    latest_turn = st.session_state.latest_turn
    if latest_turn is not None:
        level = _value(latest_turn.safety_level)
        if level == "elevated":
            st.info("This sounds like a particularly difficult moment. We can take it slowly.")
        _render_panels(latest_turn)
        with st.expander("Show raw trace"):
            st.write(f"Provider: {latest_turn.provider_used or 'not recorded'}")
            st.write(f"Latency: {latest_turn.latency_ms:.1f} ms")
            st.write(f"Turn ID: {latest_turn.turn_id or 'not recorded'}")
            st.write("Debug steps")
            st.write(latest_turn.debug_steps)

    if st.session_state.session_ended and st.session_state.session_summary is not None:
        st.subheader("Session summary")
        st.json(st.session_state.session_summary)


if __name__ == "__main__":
    main()
