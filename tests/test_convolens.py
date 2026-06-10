"""Fast, offline unit tests (no dataset downloads, no model loads)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from convolens import data, escalation, intent, sentiment
from convolens.features import _cache_path


# ---- data ----
def test_clean_text_strips_placeholders_and_lowercases():
    assert data.clean_text("Cancel order {{Order Number}}   NOW") == "cancel order now"


def test_escalation_label_membership():
    assert "complaint" in data.ESCALATION_INTENTS
    assert "check_invoice" not in data.ESCALATION_INTENTS


# ---- intent perturbation ----
def test_perturb_is_deterministic_and_changes_text():
    texts = ["i would like to cancel my subscription please"] * 3
    a = intent.perturb(texts, rate=0.3, seed=1)
    b = intent.perturb(texts, rate=0.3, seed=1)
    assert a == b  # deterministic for a fixed seed
    assert a[0] != texts[0]  # actually perturbed


def test_intent_tfidf_learns_separable_classes():
    texts = (["refund my money back"] * 20 + ["cancel the order now"] * 20
             + ["where is my invoice"] * 20)
    labels = ["refund"] * 20 + ["cancel"] * 20 + ["invoice"] * 20
    r = intent.evaluate_tfidf(texts, labels, seed=0)
    assert r.macro_f1 > 0.9
    assert r.n_classes == 3


# ---- escalation operating point ----
def test_precision_at_recall_respects_target():
    precision = np.array([0.5, 0.8, 1.0, 1.0])
    recall = np.array([1.0, 0.9, 0.4, 0.0])
    thresholds = np.array([0.2, 0.5, 0.8])
    p, t = escalation._precision_at_recall(precision, recall, thresholds, target=0.85)
    assert p == 0.8 and t == 0.5  # highest precision still meeting recall >= 0.85


def test_escalation_auroc_on_separable_data():
    rng = np.random.default_rng(0)
    pos = rng.normal(2.0, 0.5, size=(150, 8))
    neg = rng.normal(-2.0, 0.5, size=(350, 8))
    X = np.vstack([pos, neg])
    y = np.array([1] * 150 + [0] * 350)
    res = escalation.evaluate(X, y, target_recall=0.9, seed=0)
    assert res.auroc > 0.95
    assert 0.0 <= res.precision_at_target <= 1.0
    assert res.test_scores.shape[0] == res.n_test


# ---- sentiment ----
def test_sentiment_separable_and_negativity_head():
    rng = np.random.default_rng(1)
    def cloud(center, n):
        return rng.normal(center, 0.4, size=(n, 6))
    centers = {0: -3, 1: 0, 2: 3}
    Xtr = np.vstack([cloud(centers[c], 60) for c in (0, 1, 2)])
    ytr = [0] * 60 + [1] * 60 + [2] * 60
    Xte = np.vstack([cloud(centers[c], 20) for c in (0, 1, 2)])
    yte = [0] * 20 + [1] * 20 + [2] * 20
    m = sentiment.train_and_eval(Xtr, ytr, Xte, yte)
    assert m.result.macro_f1 > 0.8
    neg_scores = m.negativity(cloud(-3, 10))  # strongly negative cloud
    assert neg_scores.mean() > 0.5


# ---- feature cache addressing ----
def test_embed_cache_path_is_content_addressed():
    p1 = _cache_path(["a", "b"])
    p2 = _cache_path(["a", "b"])
    p3 = _cache_path(["a", "c"])
    assert p1 == p2 and p1 != p3
