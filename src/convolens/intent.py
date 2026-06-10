"""Intent classification (27 classes) with a deployment-robustness probe.

Two models are compared:
  * a TF-IDF + logistic-regression baseline (fast, interpretable), and
  * sentence embeddings + logistic regression.

Because the Bitext data is templated, in-distribution accuracy is high for both;
the more honest signal is how each holds up when the text is perturbed with
realistic typos, which we measure explicitly.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def _typo(text: str, rate: float, rng: random.Random) -> str:
    """Inject character-level typos (swap/drop) to simulate noisy real input."""
    chars = list(text)
    out = []
    for c in chars:
        if c != " " and rng.random() < rate:
            choice = rng.random()
            if choice < 0.5 and out:  # swap with previous
                out[-1], c = c, out[-1]
            elif choice < 0.8:  # drop
                continue
        out.append(c)
    return "".join(out)


def perturb(texts: list[str], rate: float = 0.06, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    return [_typo(t, rate, rng) for t in texts]


@dataclass
class IntentResult:
    model: str
    accuracy: float
    macro_f1: float
    accuracy_perturbed: float
    macro_f1_perturbed: float
    n_classes: int
    n_test: int
    extra: dict = field(default_factory=dict)


def evaluate_tfidf(texts: list[str], labels: list[str], seed: int = 42) -> IntentResult:
    Xtr, Xte, ytr, yte = train_test_split(
        texts, labels, test_size=0.2, random_state=seed, stratify=labels
    )
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Xtr_v = vec.fit_transform(Xtr)
    clf = LogisticRegression(max_iter=2000, C=10.0)
    clf.fit(Xtr_v, ytr)

    pred = clf.predict(vec.transform(Xte))
    pred_p = clf.predict(vec.transform(perturb(Xte, seed=seed)))
    return IntentResult(
        model="tfidf+logreg",
        accuracy=accuracy_score(yte, pred),
        macro_f1=f1_score(yte, pred, average="macro"),
        accuracy_perturbed=accuracy_score(yte, pred_p),
        macro_f1_perturbed=f1_score(yte, pred_p, average="macro"),
        n_classes=len(set(labels)),
        n_test=len(yte),
    )


def evaluate_embeddings(
    embeddings: np.ndarray,
    embeddings_perturbed: np.ndarray,
    labels: list[str],
    seed: int = 42,
) -> IntentResult:
    idx = np.arange(len(labels))
    tr, te = train_test_split(idx, test_size=0.2, random_state=seed, stratify=labels)
    y = np.asarray(labels)
    clf = LogisticRegression(max_iter=2000, C=10.0)
    clf.fit(embeddings[tr], y[tr])

    pred = clf.predict(embeddings[te])
    pred_p = clf.predict(embeddings_perturbed[te])
    return IntentResult(
        model="minilm+logreg",
        accuracy=accuracy_score(y[te], pred),
        macro_f1=f1_score(y[te], pred, average="macro"),
        accuracy_perturbed=accuracy_score(y[te], pred_p),
        macro_f1_perturbed=f1_score(y[te], pred_p, average="macro"),
        n_classes=len(set(labels)),
        n_test=len(te),
    )
