"""Interactive, local Layer 1 smoke test; LLM classification is explicit opt-in."""

from __future__ import annotations

import argparse
import json
from uuid import uuid4

from app.contracts import Message
from app.emotion import EmotionRecognizer, HuggingFaceEmotionClassifier
from app.safety.classifier import GatewaySafetyClassifier
from app.safety.responses import response_template
from app.safety.sentinel import SafetyGate
from app.tracing.recorder import TraceRecorder


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse local smoke-test options without changing provider or model defaults."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="Enable the safety LLM classifier; only use synthetic test text.",
    )
    parser.add_argument(
        "--no-emotion",
        action="store_true",
        help="Skip emotion classification and model loading.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print complete safety, emotion, and trace JSON.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run an interactive safety/emotion loop until the user enters /quit."""
    args = _parse_args(argv)

    classifier = None
    engine = None
    if args.with_llm:
        from sqlalchemy import create_engine

        from app.llm.gateway import LLMGateway

        engine = create_engine("sqlite:///:memory:")
        classifier = GatewaySafetyClassifier(LLMGateway.from_settings(engine=engine))
    history: list[Message] = []
    session_state: dict[str, object] = {}
    session_id = str(uuid4())
    emotion_classifier = None if args.no_emotion else HuggingFaceEmotionClassifier()
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
            emotion = (
                EmotionRecognizer(
                    emotion_classifier,
                    trace_recorder=recorder,
                ).read(message)
                if emotion_classifier is not None
                else None
            )
            trace = recorder.build(
                safety=safety,
                emotion=emotion,
                final_reply=response_template(safety.level) or "",
                explanation="Layer 1 safety/emotion smoke test only.",
                provider_used=(
                    f"{safety.classifier_provider}/{safety.classifier_model}"
                    if safety.classifier_provider and safety.classifier_model
                    else emotion.source_model
                    if emotion is not None
                    else "rules-only"
                ),
            )
            if args.verbose:
                print(
                    json.dumps(
                        {
                            "safety": safety.model_dump(mode="json"),
                            "emotion": emotion.model_dump(mode="json") if emotion else None,
                            "fixed_template": response_template(safety.level),
                            "trace": trace.model_dump(mode="json"),
                        },
                        indent=2,
                    )
                )
            else:
                print(
                    json.dumps(
                        {
                            "level": safety.level.value,
                            "reasons": safety.reasons,
                            "degraded": safety.degraded,
                            "classifier_provider": safety.classifier_provider,
                            "classifier_model": safety.classifier_model,
                        }
                    )
                )
            history.append(Message(role="user", content=message))
    finally:
        if engine is not None:
            engine.dispose()


if __name__ == "__main__":
    main()
