"""Sentiment classification on real, noisy customer text (tweet_eval/sentiment).

Unlike the templated support corpus, these are real social messages, so accuracy
is genuinely non-trivial. We train on the benchmark's official train split and
report on its official test split (no leakage), then expose the fitted model so
support utterances can be scored for negative sentiment.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

LABELS = {0: "negative", 1: "neutral", 2: "positive"}


@dataclass
class SentimentResult:
    accuracy: float
    macro_f1: float
    f1_negative: float
    n_train: int
    n_test: int


@dataclass
class SentimentModel:
    clf: LogisticRegression
    result: SentimentResult

    def negativity(self, embeddings: np.ndarray) -> np.ndarray:
        """P(negative) for each row, used as a customer-success risk feature."""
        proba = self.clf.predict_proba(embeddings)
        neg_col = list(self.clf.classes_).index(0)
        return proba[:, neg_col]


def train_and_eval(emb_train, y_train, emb_test, y_test) -> SentimentModel:
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    clf = LogisticRegression(max_iter=3000, C=5.0, class_weight="balanced")
    clf.fit(emb_train, y_train)
    pred = clf.predict(emb_test)
    res = SentimentResult(
        accuracy=float(accuracy_score(y_test, pred)),
        macro_f1=float(f1_score(y_test, pred, average="macro")),
        f1_negative=float(f1_score(y_test, pred, labels=[0], average="macro")),
        n_train=len(y_train),
        n_test=len(y_test),
    )
    return SentimentModel(clf=clf, result=res)
