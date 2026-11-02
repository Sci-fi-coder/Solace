"""Offline tests for the Streamlit UI using its scripted demo engine."""

from __future__ import annotations

import os
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).parents[1] / "ui" / "app.py"


def _run_demo_app() -> AppTest:
    """Start the app in demo mode with no provider calls."""
    os.environ["SOLACE_UI_DEMO"] = "1"
    return AppTest.from_file(str(APP_PATH), default_timeout=10).run()


def _start_session(app: AppTest) -> AppTest:
    """Start a session with the default synthetic user."""
    app.button(key="start_session").click()
    return app.run()


def _send(app: AppTest, message: str) -> AppTest:
    """Send one synthetic chat message."""
    app.chat_input[0].set_value(message)
    return app.run()


def test_app_starts_with_fixed_notice() -> None:
    """The app loads in demo mode and shows its fixed safety and privacy notice."""
    app = _run_demo_app()

    assert not app.exception
    assert any("not a therapist" in item.value for item in app.info)
    assert app.button(key="start_session")


def test_message_produces_an_assistant_reply() -> None:
    """A synthetic user message produces the scripted assistant response."""
    app = _start_session(_run_demo_app())
    app = _send(app, "A synthetic situation feels stressful.")

    assert not app.exception
    assert any("What part feels most present" in item.value for item in app.markdown)


def test_handoff_banner_and_citations_appear() -> None:
    """A scripted second turn displays a handoff and source citation."""
    app = _start_session(_run_demo_app())
    app = _send(app, "Synthetic message one.")
    app = _send(app, "Synthetic message two.")

    assert any("handoff to Mentor" in item.value for item in app.info)
    assert any("A small grounding pause" in item.value for item in app.markdown)


def test_crisis_turn_uses_red_box_and_hides_explanation_panels() -> None:
    """The scripted crisis turn is visually distinct and omits explanation panels."""
    app = _start_session(_run_demo_app())
    for index in range(6):
        app = _send(app, f"Synthetic message {index + 1}.")

    assert any("crisis-reply" in item.value for item in app.markdown)
    assert not any("Why I said this" in item.value for item in app.markdown)
    assert not any("What I understand so far" in item.value for item in app.markdown)


def test_delete_all_resets_the_session_and_chat() -> None:
    """Confirmed deletion clears the current session and its visible messages."""
    app = _start_session(_run_demo_app())
    app = _send(app, "A synthetic message to clear.")
    confirmation_key = f"confirm_delete_{app.session_state['session_id']}"
    app.checkbox(key=confirmation_key).check()
    app.button(key="delete_user_data").click()
    app.run()

    assert app.session_state["session_id"] is None
    assert app.session_state["chat_messages"] == []
    assert any("data was deleted" in item.value for item in app.success)
    assert app.button(key="start_session")
