from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .types import ClassificationResult


@dataclass(slots=True)
class ModelBundle:
    pipeline: object
    min_confidence: float = 0.40


def train_model(texts: list[str], labels: list[str], min_confidence: float = 0.40) -> ModelBundle:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    if not texts or not labels or len(texts) != len(labels):
        raise ValueError("Training data is invalid. Ensure non-empty equal-sized text and label arrays.")

    clf_pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2))),
            ("model", LogisticRegression(max_iter=1200)),
        ]
    )
    clf_pipeline.fit(texts, labels)
    return ModelBundle(pipeline=clf_pipeline, min_confidence=min_confidence)


def save_model(bundle: ModelBundle, output_path: Path) -> None:
    import joblib

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)


def load_model(model_path: Path) -> ModelBundle:
    import joblib

    bundle = joblib.load(model_path)
    if not isinstance(bundle, ModelBundle):
        raise TypeError("Invalid model artifact format.")
    return bundle


def classify_text(bundle: ModelBundle, clean_text: str) -> ClassificationResult:
    text = clean_text.strip()
    if not text:
        return ClassificationResult(label="uncategorized", confidence=0.0)

    probs = bundle.pipeline.predict_proba([text])[0]
    classes = bundle.pipeline.classes_
    best_idx = int(probs.argmax())
    confidence = float(probs[best_idx])
    label = str(classes[best_idx]) if confidence >= bundle.min_confidence else "uncategorized"
    return ClassificationResult(label=label, confidence=confidence)
