"""End-to-end ConvoLens pipeline: load -> intent models -> escalation -> dashboard."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from convolens import __version__, data, escalation, intent, report, sentiment
from convolens.features import embed


def load_sentiment_split():
    from datasets import load_dataset
    tr = load_dataset("tweet_eval", "sentiment", split="train")
    te = load_dataset("tweet_eval", "sentiment", split="test")
    return list(tr["text"]), list(tr["label"]), list(te["text"]), list(te["label"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="cap rows (for a quick run)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "reports"))
    args = ap.parse_args()

    t0 = time.time()
    print("[1/5] loading data ...")
    ds = data.load(limit=args.limit)
    df = ds.frame
    print(f"      {len(df):,} utterances, {df['intent'].nunique()} intents, "
          f"{df['category'].nunique()} categories, escalation rate {df['escalation'].mean():.3f}")

    texts = df["text"].tolist()
    intents = df["intent"].tolist()

    print("[2/5] intent model: TF-IDF + LogReg ...")
    r_tfidf = intent.evaluate_tfidf(texts, intents)

    print("[3/5] embedding texts (MiniLM, cached) ...")
    emb = embed(texts)
    emb_pert = embed(intent.perturb(texts))
    r_emb = intent.evaluate_embeddings(emb, emb_pert, intents)
    intent_results = [r_tfidf, r_emb]

    print("[4/6] sentiment model (tweet_eval, real noisy text, official split) ...")
    s_tr_x, s_tr_y, s_te_x, s_te_y = load_sentiment_split()
    s_emb_tr = embed([data.clean_text(t) for t in s_tr_x])
    s_emb_te = embed([data.clean_text(t) for t in s_te_x])
    smodel = sentiment.train_and_eval(s_emb_tr, s_tr_y, s_emb_te, s_te_y)

    print("[5/6] escalation-risk + fused conversation priority ...")
    esc = escalation.evaluate(emb, df["escalation"].to_numpy(), target_recall=0.90)
    # Fuse escalation probability with predicted negative sentiment into a single
    # triage score over the held-out support conversations.
    neg = smodel.negativity(emb[esc.test_idx])
    priority = 0.5 * esc.test_scores + 0.5 * neg
    pq = pd.DataFrame({
        "text": df["text"].to_numpy()[esc.test_idx],
        "intent": df["intent"].to_numpy()[esc.test_idx],
        "escalation_p": esc.test_scores,
        "negative_p": neg,
        "priority": priority,
    }).sort_values("priority", ascending=False).head(8)

    print("[6/6] rendering dashboard ...")
    out_dir = os.path.abspath(args.out)
    dash = report.build_dashboard(out_dir, df=df, intent_results=intent_results,
                                  esc=esc, smodel=smodel, priority_queue=pq,
                                  version=__version__)

    summary = {
        "n_rows": len(df),
        "n_intents": int(df["intent"].nunique()),
        "n_categories": int(df["category"].nunique()),
        "escalation_base_rate": round(float(df["escalation"].mean()), 4),
        "intent": [
            {"model": r.model, "accuracy": round(r.accuracy, 4),
             "macro_f1": round(r.macro_f1, 4),
             "macro_f1_perturbed": round(r.macro_f1_perturbed, 4),
             "accuracy_perturbed": round(r.accuracy_perturbed, 4),
             "n_test": r.n_test, "n_classes": r.n_classes}
            for r in intent_results
        ],
        "escalation": {
            "auroc": round(esc.auroc, 4), "auprc": round(esc.auprc, 4),
            "base_rate": round(esc.base_rate, 4),
            "target_recall": esc.target_recall,
            "precision_at_target": round(esc.precision_at_target, 4),
            "n_test": esc.n_test,
        },
        "sentiment": {
            "dataset": "tweet_eval/sentiment",
            "accuracy": round(smodel.result.accuracy, 4),
            "macro_f1": round(smodel.result.macro_f1, 4),
            "f1_negative": round(smodel.result.f1_negative, 4),
            "n_train": smodel.result.n_train, "n_test": smodel.result.n_test,
        },
        "runtime_sec": round(time.time() - t0, 1),
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\ndashboard -> {dash}")


if __name__ == "__main__":
    main()
