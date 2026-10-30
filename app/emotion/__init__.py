"""Lazy Hugging Face emotion recognition with injectable test classifiers."""

from app.emotion.recognizer import EmotionRecognizer, HuggingFaceEmotionClassifier

__all__ = ["EmotionRecognizer", "HuggingFaceEmotionClassifier"]
