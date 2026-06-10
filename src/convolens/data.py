"""Load and clean the Bitext customer-support dataset.

The raw utterances contain templated entity placeholders like ``{{Order Number}}``.
We strip those to a neutral token so the model learns intent language, not
placeholder formatting.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

DATASET = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"

# Intents that, in a contact center, typically signal a conversation that should
# be routed to a human or flagged for a customer-success follow-up: explicit
# escalation requests, complaints, and cancellations/refunds where churn risk is
# highest. Curated from the 27 labelled intents in the dataset.
ESCALATION_INTENTS = frozenset({
    "complaint",
    "contact_human_agent",
    "contact_customer_service",
    "cancel_order",
    "get_refund",
    "check_refund_policy",
    "track_refund",
    "delivery_options",
})

_PLACEHOLDER = re.compile(r"\{\{.*?\}\}")
_WS = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Strip templated placeholders and normalise whitespace."""
    text = _PLACEHOLDER.sub(" ", text)
    text = _WS.sub(" ", text)
    return text.strip().lower()


@dataclass
class Dataset:
    frame: pd.DataFrame  # columns: text, intent, category, escalation

    def __len__(self) -> int:
        return len(self.frame)


def load(limit: int | None = None) -> Dataset:
    """Load the dataset into a cleaned DataFrame.

    Parameters
    ----------
    limit:
        Optional cap on rows (used by tests for speed). ``None`` loads all.
    """
    from datasets import load_dataset

    split = "train" if limit is None else f"train[:{limit}]"
    raw = load_dataset(DATASET, split=split)
    df = pd.DataFrame({
        "text": [clean_text(t) for t in raw["instruction"]],
        "intent": list(raw["intent"]),
        "category": list(raw["category"]),
    })
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    df["escalation"] = df["intent"].isin(ESCALATION_INTENTS).astype(int)
    return Dataset(frame=df)
