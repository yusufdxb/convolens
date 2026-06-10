"""Escalation-risk scoring: which conversations need a human / CS follow-up.

This is the customer-success signal. We frame it as binary classification on the
utterance embedding and report threshold-free metrics (AUROC, AUPR) plus the
precision/recall available at an operating point tuned for high recall, because a
customer-success team would rather review a few extra conversations than miss an
at-risk one.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


@dataclass
class EscalationResult:
    auroc: float
    auprc: float
    base_rate: float
    target_recall: float
    precision_at_target: float
    threshold_at_target: float
    n_test: int
    pr_curve: tuple  # (precision, recall, thresholds) for plotting
    test_idx: np.ndarray = None  # row indices of the held-out test split
    test_scores: np.ndarray = None  # P(escalation) for the test split


def _precision_at_recall(precision, recall, thresholds, target: float):
    """Highest-precision operating point that still achieves >= target recall."""
    best_p, best_t = 0.0, 0.5
    for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
        if r >= target and p >= best_p:
            best_p, best_t = float(p), float(t)
    return best_p, best_t


def evaluate(embeddings: np.ndarray, labels: np.ndarray, *, target_recall: float = 0.90,
             seed: int = 42) -> EscalationResult:
    labels = np.asarray(labels)
    idx = np.arange(len(labels))
    tr, te = train_test_split(idx, test_size=0.2, random_state=seed, stratify=labels)
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    clf.fit(embeddings[tr], labels[tr])
    yte = labels[te]
    scores = clf.predict_proba(embeddings[te])[:, 1]

    precision, recall, thresholds = precision_recall_curve(yte, scores)
    p_at, t_at = _precision_at_recall(precision, recall, thresholds, target_recall)
    return EscalationResult(
        auroc=float(roc_auc_score(yte, scores)),
        auprc=float(average_precision_score(yte, scores)),
        base_rate=float(yte.mean()),
        target_recall=target_recall,
        precision_at_target=p_at,
        threshold_at_target=t_at,
        n_test=len(yte),
        pr_curve=(precision, recall, thresholds),
        test_idx=te,
        test_scores=scores,
    )
