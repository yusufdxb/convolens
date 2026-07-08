"""Render EDA figures and a self-contained HTML dashboard.

Editorial print aesthetic: cream paper, ink text, a single rust accent.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Blue primary, green secondary, ink for reference lines.
INK = "#0b0b0b"
BLUE = "#2a78d6"
GREEN = "#1baf7a"
RUST = BLUE  # legacy name kept: was the single accent, now the primary series


def fig_category_distribution(df, out: str) -> None:
    counts = df["category"].value_counts()
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.barh(counts.index[::-1], counts.values[::-1], color=RUST)
    ax.set_title("Conversations by category", loc="left", fontweight="bold")
    ax.set_xlabel("utterances")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def fig_intent_distribution(df, out: str) -> None:
    counts = df["intent"].value_counts()
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    ax.barh(counts.index[::-1], counts.values[::-1], color=BLUE)
    ax.set_title("Conversations by intent (27 classes)", loc="left", fontweight="bold")
    ax.set_xlabel("utterances")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def fig_robustness(intent_results, out: str) -> None:
    models = [r.model for r in intent_results]
    clean = [r.macro_f1 for r in intent_results]
    pert = [r.macro_f1_perturbed for r in intent_results]
    x = np.arange(len(models))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar(x - w / 2, clean, w, label="clean text", color=BLUE)
    ax.bar(x + w / 2, pert, w, label="with typos (6%)", color=GREEN)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("macro F1")
    ax.set_title("Intent accuracy under noisy input", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def fig_escalation_pr(esc, out: str) -> None:
    precision, recall, _ = esc.pr_curve
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.plot(recall, precision, color=RUST, lw=2.2)
    ax.axhline(esc.base_rate, color=INK, ls="--", lw=1.0,
               label=f"base rate {esc.base_rate:.2f}")
    ax.scatter([esc.target_recall], [esc.precision_at_target], color=INK, zorder=5)
    ax.annotate(
        f"  recall {esc.target_recall:.0%}\n  precision {esc.precision_at_target:.0%}",
        (esc.target_recall, esc.precision_at_target), va="top", fontsize=10,
    )
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, 1.02)
    ax.set_title(f"Escalation risk (AUPRC {esc.auprc:.3f})", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


_HTML = """<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
body{{background:#efe7d6;margin:0;font-family:'Inter',sans-serif;color:#1d1b17;}}
.wrap{{max-width:980px;margin:0 auto;padding:56px 48px 80px;background:#f6f1e7;
  box-shadow:0 1px 40px rgba(0,0,0,.08);}}
h1{{font-family:'Fraunces',serif;font-weight:600;font-size:40px;margin:0 0 4px;letter-spacing:-.5px;}}
.sub{{font-family:'JetBrains Mono',monospace;font-size:13px;color:#b4502a;margin-bottom:28px;}}
h2{{font-family:'Fraunces',serif;font-weight:600;font-size:24px;border-bottom:2px solid #1d1b17;
  padding-bottom:6px;margin:40px 0 18px;}}
p{{font-size:15px;line-height:1.6;max-width:70ch;}}
.kpis{{display:flex;gap:18px;flex-wrap:wrap;margin:24px 0;}}
.kpi{{flex:1;min-width:150px;background:#fff;border:1px solid #d8cfbe;border-radius:10px;padding:16px 18px;}}
.kpi .v{{font-family:'Fraunces',serif;font-size:30px;font-weight:600;color:#b4502a;}}
.kpi .l{{font-family:'JetBrains Mono',monospace;font-size:11px;color:#5a534a;margin-top:4px;text-transform:uppercase;letter-spacing:.5px;}}
img{{max-width:100%;border:1px solid #d8cfbe;border-radius:8px;margin:8px 0;background:#f6f1e7;}}
table{{border-collapse:collapse;font-size:14px;margin:10px 0;}}
th,td{{border:1px solid #d8cfbe;padding:7px 14px;text-align:left;}}
th{{background:#1d1b17;color:#f6f1e7;font-family:'JetBrains Mono',monospace;font-size:12px;}}
td.mono{{font-family:'JetBrains Mono',monospace;}}
.foot{{font-family:'JetBrains Mono',monospace;font-size:11px;color:#7a7367;margin-top:48px;
  border-top:1px solid #d8cfbe;padding-top:14px;}}
</style></head><body><div class="wrap">
<h1>ConvoLens</h1>
<div class="sub">contact-center conversation intelligence &middot; {n_rows:,} utterances &middot; {n_intents} intents &middot; {n_cats} categories</div>
<p>An end-to-end analysis of a public customer-support corpus: what customers
contact about, how reliably their intent can be classified (including under noisy,
real-world text), and which conversations carry escalation risk that a
customer-success team should act on first.</p>

<div class="kpis">
  <div class="kpi"><div class="v">{best_f1:.3f}</div><div class="l">intent macro F1 (27 classes)</div></div>
  <div class="kpi"><div class="v">{sent_f1:.3f}</div><div class="l">sentiment macro F1 (real tweets)</div></div>
  <div class="kpi"><div class="v">{esc_auroc:.3f}</div><div class="l">escalation AUROC</div></div>
  <div class="kpi"><div class="v">{esc_prec:.0%}</div><div class="l">precision @ {esc_recall:.0%} recall</div></div>
</div>

<h2>What customers contact about</h2>
<p>Volume is concentrated in account, order, and refund topics. This distribution
is itself a customer-success signal: refund and cancellation volume is where churn
risk lives.</p>
<img src="figures/category_distribution.png">
<img src="figures/intent_distribution.png">

<h2>Intent classification, and how it survives noisy text</h2>
<p>Both a TF-IDF baseline and a sentence-embedding model classify the 27 intents
with high accuracy on clean text. Because production text is messier, each model is
re-scored after injecting character-level typos. The embedding model degrades far
less, which is the property that matters in deployment.</p>
<table>
<tr><th>model</th><th>macro F1 (clean)</th><th>macro F1 (typos)</th><th>accuracy (clean)</th></tr>
{intent_rows}
</table>
<img src="figures/robustness.png">

<h2>Sentiment on real, noisy customer text</h2>
<p>The support corpus is templated, so its intent labels are near-perfectly
separable. To measure sentiment on messages that look like the real world, the
model is trained and tested on the SemEval tweet-sentiment benchmark (real social
posts, official train/test split, no leakage). Macro F1 of {sent_f1:.3f} across
three classes is in line with strong lightweight baselines and, unlike the intent
numbers, reflects genuine difficulty. The negative-sentiment head is then reused as
a risk feature on the support conversations.</p>
<table>
<tr><th>metric</th><th>value</th></tr>
<tr><td>accuracy</td><td class="mono">{sent_acc:.3f}</td></tr>
<tr><td>macro F1 (3-class)</td><td class="mono">{sent_f1:.3f}</td></tr>
<tr><td>F1 (negative class)</td><td class="mono">{sent_negf1:.3f}</td></tr>
</table>

<h2>Escalation risk for customer success</h2>
<p>Escalation-prone conversations (explicit human-agent requests, complaints,
cancellations and refunds) make up {esc_base:.0%} of the corpus. A classifier on the
utterance embedding ranks them with AUROC {esc_auroc:.3f} and AUPRC {esc_auprc:.3f};
because the escalation label is derived from intent, this score is expected to be
high, and its value here is the operating point it exposes. Tuned for high recall, a
team can catch {esc_recall:.0%} of at-risk conversations while keeping precision at
{esc_prec:.0%}, so the review queue stays small.</p>
<img src="figures/escalation_pr.png">

<h2>Fused conversation-priority queue</h2>
<p>A single triage score blends escalation probability with predicted negative
sentiment, surfacing the conversations a customer-success team should open first.
The highest-priority held-out conversations:</p>
<table>
<tr><th>priority</th><th>escalation</th><th>neg. sentiment</th><th>intent</th><th>utterance</th></tr>
{priority_rows}
</table>

<div class="foot">Generated by ConvoLens v{version} &middot; dataset: Bitext customer-support &middot;
models: scikit-learn TF-IDF/LogReg + MiniLM embeddings &middot; all metrics computed on a held-out 20% test split.</div>
</div></body></html>"""


def _esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def build_dashboard(out_dir: str, *, df, intent_results, esc, smodel, priority_queue,
                    version: str) -> str:
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    fig_category_distribution(df, os.path.join(fig_dir, "category_distribution.png"))
    fig_intent_distribution(df, os.path.join(fig_dir, "intent_distribution.png"))
    fig_robustness(intent_results, os.path.join(fig_dir, "robustness.png"))
    fig_escalation_pr(esc, os.path.join(fig_dir, "escalation_pr.png"))

    best = max(intent_results, key=lambda r: r.macro_f1)
    rows = "\n".join(
        f"<tr><td class='mono'>{r.model}</td><td class='mono'>{r.macro_f1:.3f}</td>"
        f"<td class='mono'>{r.macro_f1_perturbed:.3f}</td><td class='mono'>{r.accuracy:.3f}</td></tr>"
        for r in intent_results
    )
    prio_rows = "\n".join(
        f"<tr><td class='mono'>{row.priority:.2f}</td><td class='mono'>{row.escalation_p:.2f}</td>"
        f"<td class='mono'>{row.negative_p:.2f}</td><td class='mono'>{_esc(str(row.intent))}</td>"
        f"<td>{_esc(str(row.text))[:90]}</td></tr>"
        for row in priority_queue.itertuples()
    )
    html = _HTML.format(
        n_rows=len(df), n_intents=df["intent"].nunique(), n_cats=df["category"].nunique(),
        best_f1=best.macro_f1, esc_auroc=esc.auroc, esc_auprc=esc.auprc,
        esc_prec=esc.precision_at_target, esc_recall=esc.target_recall,
        esc_base=esc.base_rate, intent_rows=rows, version=version,
        sent_acc=smodel.result.accuracy, sent_f1=smodel.result.macro_f1,
        sent_negf1=smodel.result.f1_negative, priority_rows=prio_rows,
    )
    path = os.path.join(out_dir, "dashboard.html")
    with open(path, "w") as f:
        f.write(html)
    return path
