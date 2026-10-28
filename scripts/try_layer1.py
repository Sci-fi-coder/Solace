"""Interactive, local Layer 1 smoke test; LLM classification is explicit opt-in."""

from __future__ import annotations

import argparse
import json
from uuid import uuid4

from app.contracts import Message, SafetyLevel
from app.emotion import EmotionRecognizer, HuggingFaceEmotionClassifier
from app.safety.classifier import GatewaySafetyClassifier
from app.safety.responses import response_template
from app.safety.sentinel import SafetyGate
from app.tracing.recorder import TraceRecorder


def main() -> None:
    """Run an interactive safety/emotion loop until the user enters /quit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="Send messages to configured providers; only use synthetic test text.",
    )
    args = parser.parse_args()

    classifier = None
    if args.with_llm:
        from sqlalchemy import create_engine

        from app.llm.gateway import LLMGateway

        engine = create_engine("sqlite:///:memory:")
        classifier = GatewaySafetyClassifier(LLMGateway.from_settings(engine=engine))
    print("Layer 1 local test. Enter /quit to exit.")
    print("Use synthetic text only; free-tier providers may log requests.")
    history: list[Message] = []
    session_state: dict[str, object] = {}
    session_id = str(uuid4())
    emotion_classifier = HuggingFaceEmotionClassifier()
    try:
        while True:
            try:
                message = input("You> ").strip()
            except EOFError:
                break
            if message.lower() in {"/quit", "/exit"}:
                break
            if not message:
                continue
            recorder = TraceRecorder(
                turn_id=str(uuid4()),
                session_id=session_id,
                user_message=message,
            )
            safety = SafetyGate(classifier, recorder).check(
                message,
                history,
                session_state,
            )
            emotion = EmotionRecognizer(
                emotion_classifier,
                trace_recorder=recorder,
            ).read(message)
            trace = recorder.build(
                safety=safety,
                emotion=emotion,
                final_reply=response_template(safety.level) or "",
                explanation="Layer 1 safety/emotion smoke test only.",
                provider_used=safety.classifier_provider or emotion.source_model,
            )
            print("SafetyResult:")
            print(safety.model_dump_json(indent=2))
            print("EmotionReading:")
            print(emotion.model_dump_json(indent=2))
            print("Fixed template:")
            print(response_template(safety.level) or "(none)")
            if safety.level == SafetyLevel.crisis:
                print("Crisis override: do not send this turn to a reply-generating agent.")
            print("StepRecords:")
            print(json.dumps([step.model_dump(mode="json") for step in trace.steps], indent=2))
            history.append(Message(role="user", content=message))
    finally:
        if args.with_llm:
            engine.dispose()


if __name__ == "__main__":
    main()
