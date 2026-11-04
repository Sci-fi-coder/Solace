"""Offline tests for the Streamlit UI using its scripted demo engine."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from streamlit.testing.v1 import AppTest

from app.rolemodels.models import Person, RoleModelProvenance
from ui.app import _safe_debug_steps
from ui.demo_engine import DemoEngine

APP_PATH = Path(__file__).parents[1] / "ui" / "app.py"


def test_debug_diagnostics_redact_recognizable_credentials() -> None:
    """Trace diagnostics do not display provider credentials or bearer tokens."""
    steps = _safe_debug_steps(
        ["provider key=sk-test-secret", "request Bearer token-secret"]
    )

    assert steps == ["provider key=[REDACTED]", "request [REDACTED]"]


def _run_demo_app() -> AppTest:
    """Start the app in demo mode with no provider calls."""
    st.cache_resource.clear()
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
    assert any("Safety state: crisis" in item.value for item in app.error)
    assert not any("Why I said this" in item.value for item in app.markdown)
    assert not any("What I understand so far" in item.value for item in app.markdown)


def test_ending_session_allows_starting_a_fresh_session() -> None:
    """Ending a session enables a fresh session and clears the old conversation."""
    app = _start_session(_run_demo_app())
    old_session_id = app.session_state["session_id"]
    app = _send(app, "A synthetic message from the first session.")
    app.button(key="end_session").click()
    app.run()

    assert app.button(key="new_session")
    app.button(key="new_session").click()
    app.run()

    assert app.session_state["session_id"] != old_session_id
    assert app.session_state["session_ended"] is False
    assert len(app.session_state["chat_messages"]) == 1
    assert "Hi, I'm Solace" in app.session_state["chat_messages"][0]["content"]


def test_demo_mode_does_not_initialize_the_provider_engine(monkeypatch) -> None:
    """Offline demo startup does not invoke the normal provider-backed factory."""
    from app.pipeline import factory

    def fail_if_called():
        raise AssertionError("normal engine factory must not run in demo mode")

    monkeypatch.setattr(factory, "build_engine_from_settings", fail_if_called)
    app = _start_session(_run_demo_app())

    assert not app.exception
    assert app.session_state["session_id"].startswith("demo-session-")


def test_normal_mode_routes_messages_through_the_engine_factory(monkeypatch) -> None:
    """Normal mode constructs the configured engine and sends it the user's message."""
    from app.pipeline import factory

    os.environ.pop("SOLACE_UI_DEMO", None)
    st.cache_resource.clear()
    engine = DemoEngine()
    factory_calls = []
    sent_messages = []
    original_handle_turn = engine.handle_turn

    def build_engine():
        factory_calls.append(True)
        return engine

    def handle_turn(
        session_id: str,
        text: str,
        *,
        enable_role_models: bool = False,
    ):
        sent_messages.append(text)
        return original_handle_turn(
            session_id, text, enable_role_models=enable_role_models
        )

    monkeypatch.setattr(factory, "build_engine_from_settings", build_engine)
    monkeypatch.setattr(engine, "handle_turn", handle_turn)
    app = AppTest.from_file(str(APP_PATH), default_timeout=10).run()
    app = _start_session(app)
    app = _send(app, "A synthetic message sent to the configured engine.")

    assert not app.exception
    assert factory_calls
    assert sent_messages == ["A synthetic message sent to the configured engine."]


def test_engine_initialization_failure_is_shown_as_a_generic_error(monkeypatch) -> None:
    """A factory failure does not expose its exception or crash the UI."""
    from app.pipeline import factory

    os.environ.pop("SOLACE_UI_DEMO", None)
    st.cache_resource.clear()

    def fail_initialization():
        raise RuntimeError("sensitive provider diagnostic")

    monkeypatch.setattr(factory, "build_engine_from_settings", fail_initialization)
    app = AppTest.from_file(str(APP_PATH), default_timeout=10).run()

    assert not app.exception
    assert any("couldn't initialize" in item.value for item in app.error)
    assert not any("sensitive provider diagnostic" in item.value for item in app.error)


def test_engine_turn_errors_are_presented_without_exception_details(monkeypatch) -> None:
    """An engine failure appears as a recoverable generic UI message."""
    def fail_turn(
        self: DemoEngine,
        session_id: str,
        text: str,
        *,
        enable_role_models: bool = False,
    ):
        del enable_role_models
        raise RuntimeError("sensitive provider diagnostic")

    monkeypatch.setattr(DemoEngine, "handle_turn", fail_turn)
    app = _start_session(_run_demo_app())
    app = _send(app, "Synthetic message that triggers a local failure.")

    assert not app.exception
    assert any("couldn't process that message" in item.value for item in app.error)
    assert not any("sensitive provider diagnostic" in item.value for item in app.error)


def test_historical_toggle_passes_setting_and_shows_only_used_provenance(monkeypatch) -> None:
    """The opt-in reaches the engine and only returned provenance renders a panel."""
    original_handle_turn = DemoEngine.handle_turn
    requested: list[bool] = []

    def handle_turn(
        self: DemoEngine,
        session_id: str,
        text: str,
        *,
        enable_role_models: bool = False,
    ):
        requested.append(enable_role_models)
        result = original_handle_turn(
            self, session_id, text, enable_role_models=enable_role_models
        )
        return result.model_copy(
            update={
                "historical_inspiration": RoleModelProvenance(
                    card_id="gandhi-pietermaritzburg",
                    person=Person.gandhi,
                    title="Responding to humiliation with purpose",
                    source_locators=["Chapter 4, printed pp. 14–15"],
                )
            }
        )

    monkeypatch.setattr(DemoEngine, "handle_turn", handle_turn)
    app = _start_session(_run_demo_app())
    assert not any("Historical inspiration" in item.value for item in app.subheader)
    app.checkbox(key="enable_role_models").check()
    app.run()
    app = _send(app, "A synthetic situation feels stressful.")

    assert requested == [True]
    assert any("Historical inspiration" in item.value for item in app.subheader)
    assert any("Mahatma Gandhi" in item.value for item in app.markdown)
    assert any("Chapter 4, printed pp. 14–15" in item.value for item in app.caption)
    assert any(
        "Shared as an analogy, not as the same situation." in item.value for item in app.caption
    )


def test_historical_panel_is_hidden_for_elevated_and_crisis_turns(monkeypatch) -> None:
    """Safety states suppress the provenance panel even if an engine returns one."""
    original_handle_turn = DemoEngine.handle_turn
    provenance = RoleModelProvenance(
        card_id="gandhi-pietermaritzburg",
        person=Person.gandhi,
        title="Responding to humiliation with purpose",
        source_locators=["Chapter 4, printed pp. 14–15"],
    )

    def handle_turn(
        self: DemoEngine,
        session_id: str,
        text: str,
        *,
        enable_role_models: bool = False,
    ):
        output = original_handle_turn(
            self, session_id, text, enable_role_models=enable_role_models
        )
        return output.model_copy(update={"historical_inspiration": provenance})

    monkeypatch.setattr(DemoEngine, "handle_turn", handle_turn)
    app = _start_session(_run_demo_app())
    for index in range(5):
        app = _send(app, f"Synthetic safety presentation turn {index + 1}.")

    assert any("Elevated concern" in item.value for item in app.caption) or any(
        "Elevated concern" in item.value for item in app.info
    )
    assert not any("Historical inspiration" in item.value for item in app.subheader)
    app = _send(app, "Synthetic safety presentation turn 6.")

    assert any("Safety state: crisis" in item.value for item in app.error)
    assert not any("Historical inspiration" in item.value for item in app.subheader)


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
