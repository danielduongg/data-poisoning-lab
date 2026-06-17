"""model.py -- TF-IDF + logistic-regression text classifier; clean accuracy and ASR."""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline


def build_model() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                                  min_df=2, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=2000, C=8.0)),
    ])


def train(texts, labels) -> Pipeline:
    return build_model().fit(list(texts), list(labels))


def clean_accuracy(model, test) -> float:
    X = [t for t, _ in test]
    y = [c for _, c in test]
    return float(accuracy_score(y, model.predict(X)))


def attack_success_rate(model, triggered_texts, target: int = 1) -> float:
    """Fraction of triggered (source-class) docs the model misclassifies as target."""
    if not triggered_texts:
        return 0.0
    pred = np.asarray(model.predict(list(triggered_texts)))
    return float(np.mean(pred == target))
