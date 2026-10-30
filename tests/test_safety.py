"""Tests for rule screening, classifier escalation, degradation, and crisis responses."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from app.contracts import EmotionReading, Message, SafetyLevel
from app.emotion.recognizer import DEFAULT_LABEL_MAP, EmotionRecognizer
from app.llm.gateway import GatewayExhaustedError, LLMGateway
from app.llm.providers import LLMResult, MockProvider
from app.safety.classifier import (
    ClassifierResult,
    GatewaySafetyClassifier,
)
from app.safety.responses import (
    CRISIS_OVERRIDE,
    CRISIS_TEMPLATE,
    SOFT_CHECK,
    action_for,
    load_helplines,
    response_template,
)
from app.safety.rules import classify_rules, normalize_text
from app.safety.sentinel import SafetyGate
from app.tracing.recorder import TraceRecorder


class StaticSafetyClassifier:
    """Return one deterministic structured risk classification."""

    def __init__(
        self,
        level: SafetyLevel,
        reasons: list[str] | None = None,
    ) -> None:
        self.level = level
        self.reasons = reasons or []
        self.calls = 0

    def classify(self, message: str, history: Sequence[Message]) -> ClassifierResult:
        """Return the configured level without inspecting synthetic message content."""
        self.calls += 1
        return ClassifierResult(
            level=self.level,
            reasons=self.reasons,
            provider="mock",
            model="mock-classifier",
        )


class SequenceSafetyClassifier:
    """Return configured levels in order to exercise sticky session state."""

    def __init__(self, levels: list[SafetyLevel]) -> None:
        self._levels = iter(levels)

    def classify(self, message: str, history: Sequence[Message]) -> ClassifierResult:
        """Return the next level in the test sequence."""
        return ClassifierResult(
            level=next(self._levels),
            provider="mock",
            model="mock-classifier",
        )


class StaticEmotionClassifier:
    """Return deterministic fine-grained emotion probabilities."""

    def predict(self, message: str) -> Sequence[tuple[str, float]]:
        """Return a sadness-heavy synthetic score distribution."""
        return [("sadness", 0.4), ("grief", 0.3), ("joy", 0.3)]


class CapturingMockProvider(MockProvider):
    """Capture the outgoing synthetic prompt while preserving MockProvider behavior."""

    def __init__(self) -> None:
        super().__init__(
            task_kind="classify",
            fixed_output='{"level":"none","reasons":[]}',
        )
        self.messages: list[dict[str, str]] = []

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 800,
        response_format: dict[str, Any] | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
    ) -> LLMResult:
        """Record the prompt and return the normal deterministic mock output."""
        self.messages = messages
        return super().generate(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            model=model,
            reasoning_effort=reasoning_effort,
        )


def test_rule_levels_cover_crisis_elevated_and_none() -> None:
    """Rules distinguish explicit intent, self-harm urges, and ordinary distress."""
    assert classify_rules("I have a plan to kill myself tonight.")[0] == SafetyLevel.crisis
    assert classify_rules("I feel like hurting myself when stressed.")[0] == SafetyLevel.elevated
    assert classify_rules("I feel sad about a difficult week.")[0] == SafetyLevel.none


@pytest.mark.parametrize(
    "message",
    [
        "This exam is killing me.",
        "I could die of embarrassment.",
        "I'm not going to hurt myself.",
        "I am not going to kill myself.",
        "I won't hurt anyone.",
        "The deadline is murdering my weekend.",
    ],
)
def test_idioms_and_negated_intent_do_not_trigger_crisis(message: str) -> None:
    """Idioms and clear negative intent remain below the crisis level."""
    assert classify_rules(message)[0] != SafetyLevel.crisis


def test_normalization_handles_repeated_letters_typos_and_leetspeak() -> None:
    """Text normalization handles common obfuscations before matching patterns."""
    assert normalize_text("I waaaant to k1ll myself!!!") == "i want to kill myself"
    assert classify_rules("I want to k1ll myself.")[0] == SafetyLevel.crisis
    assert normalize_text("k1ll mys3lf") == "kill myself"
    assert classify_rules("k1ll mys3lf")[0] == SafetyLevel.crisis


@pytest.mark.parametrize(
    "message",
    [
        "ab aur nahi jee sakta",
        "I just want to sleep and not wake up",
        "I wish I could disappear for good",
        "There is no reason to live anymore",
        "no reason to live",
        "I can't go on living anymore",
        "I can't take it anymore",
        "main jeena nahi chahti",
        "Main jeena nahi chahta",
        "ab aur nahi jee sakti",
        "bas sab khatam",
        "main sab khatam kar du",
        "everyone would be better off without me",
    ],
)
def test_passive_ideation_family_is_elevated(message: str) -> None:
    """English and Hinglish passive-ideation paraphrases receive elevated screening."""
    assert classify_rules(message)[0] == SafetyLevel.elevated


@pytest.mark.parametrize(
    "message",
    [
        "I wish I could just disappear",
        "I wish I could vanish from everyone's life",
        "I want to fade away and never come back",
        "I feel like fading away from everyone's life",
    ],
)
def test_disappearance_wishes_are_elevated(message: str) -> None:
    """First-person disappearance wishes receive elevated screening."""
    assert classify_rules(message)[0] == SafetyLevel.elevated


@pytest.mark.parametrize(
    "message",
    [
        "no one would even notice if I was gone",
        "Nobody would miss me if I were gone",
        "Everyone would be better off if I weren't here",
        "The world would be better off without me",
    ],
)
def test_better_off_without_me_wishes_are_elevated(message: str) -> None:
    """First-person absence and better-off wishes receive elevated screening."""
    assert classify_rules(message)[0] == SafetyLevel.elevated


@pytest.mark.parametrize(
    "message",
    [
        "thinking of ending it all tonight",
        "I'm planning to end it all today",
        "I am going to end it soon",
        "I'm ending it all tonight",
    ],
)
def test_ending_it_with_intent_or_time_is_crisis(message: str) -> None:
    """Intent or near-term timing with ending-it language receives crisis screening."""
    assert classify_rules(message)[0] == SafetyLevel.crisis


@pytest.mark.parametrize(
    "message",
    [
        "I want to end this meeting",
        "I'm ending my diet tonight",
        "The show ended it all with a twist",
        "The magician disappeared behind the curtain",
        "The file disappeared after the reboot",
        "The sun disappeared behind the clouds",
        "The team would be better off without the broken printer",
        "Everyone would be better off if the server were replaced",
        "The project would be better off without the outdated dependency",
        "I'm dying of laughter",
        "I could kill for a pizza right now",
    ],
)
def test_new_safety_family_hard_negatives_remain_none(message: str) -> None:
    """Benign disappearance, better-off, and ending-it contexts remain unflagged."""
    assert classify_rules(message)[0] == SafetyLevel.none


@pytest.mark.parametrize(
    "message",
    [
        "kill myself",
        "end my life",
        "hurt myself",
        "k1ll mys3lf",
        "h4rm mys3lf",
        "3nd my l1fe",
        "Main khud ko maarna chahta hu",
        "Main apni jaan le lunga",
    ],
)
def test_explicit_self_harm_family_is_crisis(message: str) -> None:
    """Bare, leetspeak, and Hinglish self-harm expressions receive crisis screening."""
    assert classify_rules(message)[0] == SafetyLevel.crisis


@pytest.mark.parametrize(
    "message",
    [
        "I want to sleep in and not wake up before noon on Sunday",
        "my phone died, I can't live without charging it lol",
        "jee bhar ke khaya",
        "main thak gaya hoon, bas chai chahiye",
    ],
)
def test_passive_ideation_hard_negatives_remain_none(message: str) -> None:
    """Benign sleep, phone, and Hinglish phrases do not trigger passive-ideation rules."""
    assert classify_rules(message)[0] == SafetyLevel.none


@pytest.mark.parametrize(
    "message",
    [
        "this heat is killing me",
        "exam dekh ke main mar gaya yaar",
        "this deadline is killing me",
    ],
)
def test_self_harm_idioms_remain_none(message: str) -> None:
    """Everyday killing/death idioms do not trigger explicit self-harm rules."""
    assert classify_rules(message)[0] == SafetyLevel.none


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("I feel like hurting myself when stressed.", SafetyLevel.elevated),
        ("I want to hurt myself when overwhelmed.", SafetyLevel.elevated),
        ("mujhe khud ko hurt karne ka urge hai", SafetyLevel.elevated),
    ],
)
def test_non_crisis_self_harm_urges_remain_elevated(
    message: str,
    expected: SafetyLevel,
) -> None:
    """Explicit self-harm expression matching does not upgrade urge framing to crisis."""
    assert classify_rules(message)[0] == expected


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("I don't want to be alive anymore.", SafetyLevel.elevated),
        ("I don’t want to be alive anymore.", SafetyLevel.elevated),
        ("I don‘t want to be alive anymore.", SafetyLevel.elevated),
        ("I donʼt want to be alive anymore.", SafetyLevel.elevated),
        ("I don`t want to be alive anymore.", SafetyLevel.elevated),
        ("I don´t want to be alive anymore.", SafetyLevel.elevated),
        ("I’m not going to kill myself", SafetyLevel.none),
        ("I won’t hurt anyone", SafetyLevel.none),
        ("Ｉ ｄｏｎ＇ｔ ｗａｎｔ ｔｏ ｂｅ ａｌｉｖｅ ａｎｙｍｏｒｅ.", SafetyLevel.elevated),
        ("I don't want to be alive\u200b anymore.", SafetyLevel.elevated),
        ("I\u00a0don't\u00a0want\u00a0to\u00a0be\u00a0alive\u00a0anymore.", SafetyLevel.elevated),
        ("Ｉ ｗｉｌｌ ｋｉｌｌ ｍｙｓｅｌｆ ｔｏｎｉｇｈｔ.", SafetyLevel.crisis),
        ("I\u200bwill\u200bkill\u200bmyself\u200btonight.", SafetyLevel.crisis),
        ("I will k\u2060ill myself tonight.", SafetyLevel.crisis),
        ("I\u00a0will\u00a0kill\u00a0myself\u00a0tonight.", SafetyLevel.crisis),
    ],
)
def test_unicode_safety_variants(message: str, expected: SafetyLevel) -> None:
    """Unicode punctuation and spacing preserve the intended rule classification."""
    assert classify_rules(message)[0] == expected


def test_dataset_messages_match_their_generated_curly_apostrophe_variants() -> None:
    """Every evaluation phrase keeps its exact rule result when apostrophes turn curly."""
    dataset_path = Path(__file__).parent / "data" / "safety_eval.jsonl"
    messages = [
        json.loads(line)["text"]
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
    ]

    for message in messages:
        curly_message = message.replace("'", "’")
        assert classify_rules(curly_message) == classify_rules(message), message


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Main khud ko maar dunga aaj raat.", SafetyLevel.crisis),
        ("Main apni jaan lene ka plan bana raha hu.", SafetyLevel.crisis),
        ("Jeene ka mann nahi karta.", SafetyLevel.elevated),
        ("Ab kuch behtar nahi hoga.", SafetyLevel.elevated),
        ("Aaj ka din stressful tha, ab chai pi raha hu.", SafetyLevel.none),
    ],
)
def test_hinglish_patterns(message: str, expected: SafetyLevel) -> None:
    """Romanised Hindi expressions map to the expected safety tier."""
    assert classify_rules(message)[0] == expected


def test_classifier_cannot_downgrade_a_rule_hit() -> None:
    """The maximum of the rule and classifier levels determines the outcome."""
    gate = SafetyGate(StaticSafetyClassifier(SafetyLevel.none))

    result = gate.check("I plan to kill myself tonight.", [], {})

    assert result.level == SafetyLevel.crisis
    assert result.action == CRISIS_OVERRIDE


def test_unavailable_classifier_degrades_to_rules_only(caplog: pytest.LogCaptureFixture) -> None:
    """Exhausted providers preserve rule results, mark degradation, and warn."""

    class FailedClassifier:
        def classify(self, message: str, history: Sequence[Message]) -> ClassifierResult:
            raise GatewayExhaustedError("synthetic provider exhaustion")

    result = SafetyGate(FailedClassifier()).check("I can't go on.", [], {})

    assert result.level == SafetyLevel.elevated
    assert result.degraded is True
    assert "rules only" in caplog.text


def test_crisis_escalation_is_sticky_within_the_session() -> None:
    """Once a session has reached crisis, later turns remain at least elevated."""
    classifier = SequenceSafetyClassifier([SafetyLevel.crisis, SafetyLevel.none])
    state: dict[str, object] = {}
    gate = SafetyGate(classifier)

    first = gate.check("A synthetic first turn.", [], state)
    later = gate.check("I am discussing a calm day.", [], state)

    assert first.level == SafetyLevel.crisis
    assert later.level == SafetyLevel.elevated
    assert "sticky_crisis_escalation" in later.reasons
    assert later.action == SOFT_CHECK


def test_crisis_response_is_fixed_and_elevated_does_not_override_pipeline() -> None:
    """Only crisis gets the immutable crisis reply; elevated returns soft-check guidance."""
    crisis = SafetyGate(StaticSafetyClassifier(SafetyLevel.crisis)).check(
        "Synthetic crisis-classifier signal.",
        [],
        {},
    )
    assert crisis.action == CRISIS_OVERRIDE
    assert response_template(crisis.level) == CRISIS_TEMPLATE
    assert "someone you trust" in CRISIS_TEMPLATE
    assert "emergency services" in CRISIS_TEMPLATE
    assert response_template(SafetyLevel.elevated) is not None
    assert "findahelpline.com" in response_template(SafetyLevel.elevated)
    assert action_for(SafetyLevel.elevated) == SOFT_CHECK
    assert response_template(SafetyLevel.none) is None


def test_gateway_primary_failure_falls_back_without_changing_safety_level() -> None:
    """A failed primary provider is recorded while the backup's safety result is retained."""
    output = '{"level":"crisis","reasons":["explicit intent"]}'
    failed_primary = MockProvider(
        task_kind="classify",
        raise_on_call=RuntimeError("synthetic primary failure"),
    )
    backup = MockProvider(task_kind="classify", fixed_output=output)
    gateway = LLMGateway(
        providers={"primary": failed_primary, "backup": backup},
        fallback_chains={"classify": ["primary", "backup"]},
        daily_caps={"primary": 10, "backup": 10},
        max_retries=1,
        provider_models={"primary": "model-primary", "backup": "model-backup"},
    )
    classifier = GatewaySafetyClassifier(gateway)

    fallback_result = SafetyGate(classifier).check("Synthetic classifier-only signal.", [], {})
    backup_only = GatewaySafetyClassifier(
        LLMGateway(
            providers={"backup": MockProvider(task_kind="classify", fixed_output=output)},
            fallback_chains={"classify": ["backup"]},
            daily_caps={"backup": 10},
            provider_models={"backup": "model-backup"},
        )
    )
    baseline_result = SafetyGate(backup_only).check("Synthetic classifier-only signal.", [], {})

    assert fallback_result.level == baseline_result.level == SafetyLevel.crisis
    assert fallback_result.classifier_provider == "backup"
    assert fallback_result.classifier_model == "model-backup"
    assert fallback_result.degraded is False
    assert fallback_result.classifier_attempts[0]["provider"] == "primary"


def test_classifier_receives_current_turn_and_only_six_prior_user_turns() -> None:
    """The classifier prompt keeps six recent user turns and omits assistant turns."""
    provider = CapturingMockProvider()
    gateway = LLMGateway(
        providers={"mock": provider},
        fallback_chains={"classify": ["mock"]},
        daily_caps={"mock": 10},
    )
    history = [
        Message(role="user", content=f"turn-{index}")
        for index in range(8)
    ] + [Message(role="assistant", content="assistant-only-content")]

    GatewaySafetyClassifier(gateway).classify("current synthetic turn", history)

    prompt = provider.messages[-1]["content"]
    assert "turn-0" not in prompt
    assert all(f"turn-{index}" in prompt for index in range(2, 8))
    assert "current synthetic turn" in prompt
    assert "assistant-only-content" not in prompt


def test_helpline_loader_warns_for_each_unverified_entry(caplog: pytest.LogCaptureFixture) -> None:
    """The loader reports every unverified help resource without inventing numbers."""
    entries = load_helplines()

    assert entries
    assert all(entry["number"] == "TODO_VERIFY" for entry in entries)
    assert "unverified" in caplog.text


def test_safety_and_emotion_steps_are_recorded_without_message_content() -> None:
    """Both Layer 1 components append summary-only StepRecords to a shared recorder."""
    recorder = TraceRecorder(
        turn_id="synthetic-turn",
        session_id="synthetic-session",
        user_message="synthetic content must not enter the step summary",
    )
    safety = SafetyGate(StaticSafetyClassifier(SafetyLevel.none), recorder).check(
        "Synthetic input.",
        [],
        {},
    )
    emotion = EmotionRecognizer(
        StaticEmotionClassifier(),
        trace_recorder=recorder,
    ).read("Synthetic input.")
    trace = recorder.build(
        safety=safety,
        emotion=emotion,
        final_reply="",
        explanation="Synthetic test.",
        provider_used="mock",
    )

    assert [step.name for step in trace.steps] == ["safety", "emotion"]
    assert all("Synthetic input" not in step.summary for step in trace.steps)


def test_all_goemotions_labels_have_a_coarse_mapping() -> None:
    """The configured default map covers the complete 28-label GoEmotions set."""
    assert len(DEFAULT_LABEL_MAP) == 28
    assert {"sadness", "anxiety_fear", "anger", "shame_guilt", "positive", "neutral"} <= set(
        DEFAULT_LABEL_MAP.values()
    )


def test_emotion_recognizer_returns_fine_and_aggregated_coarse_labels() -> None:
    """Emotion output includes the best fine label and summed coarse probabilities."""
    reading = EmotionRecognizer(StaticEmotionClassifier()).read("Synthetic emotion input.")

    assert isinstance(reading, EmotionReading)
    assert reading.fine_label == "sadness"
    assert reading.coarse_label == reading.label == "sadness"
    assert reading.score == pytest.approx(0.7)
    assert reading.top_k[0] == ("sadness", pytest.approx(0.7))


def test_custom_coarse_label_map_is_used() -> None:
    """Callers can configure the fine-to-coarse label mapping."""
    reading = EmotionRecognizer(
        StaticEmotionClassifier(),
        label_map={"sadness": "low", "grief": "low", "joy": "high"},
    ).read("Synthetic emotion input.")

    assert reading.coarse_label == "low"


@pytest.mark.slow
def test_default_emotion_model_can_infer_on_synthetic_text() -> None:
    """The optional real model integration works when explicitly running slow tests."""
    reading = EmotionRecognizer().read("I feel nervous about a synthetic test.")

    assert reading.fine_label
    assert reading.coarse_label
