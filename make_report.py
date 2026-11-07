"""Combine baseline and transformer results into results/summary.md and the figures."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent / "results"
FIG = RESULTS / "figures"
SETTINGS = [("phrasebank", "phrasebank"), ("phrasebank", "twitter"), ("twitter", "twitter"), ("twitter", "phrasebank")]
SETTING_LABEL = {
    ("phrasebank", "phrasebank"): "News → News",
    ("phrasebank", "twitter"): "News → Twitter",
    ("twitter", "twitter"): "Twitter → Twitter",
    ("twitter", "phrasebank"): "Twitter → News",
    ("none", "phrasebank"): "No training → News",
    ("none", "twitter"): "No training → Twitter",
}
PRETTY = {"distilbert-base-uncased": "DistilBERT (fine-tuned)", "yiyanghkust/finbert-pretrain": "FinBERT (fine-tuned)"}
ORDER = ["Loughran-McDonald lexicon", "TF-IDF + logistic regression", "DistilBERT (fine-tuned)", "FinBERT (fine-tuned)"]
COLOURS = dict(zip(ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#e87ba4"]))
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def collect() -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "baselines.csv")
    rows = df.to_dict("records")
    for f in sorted((RESULTS / "transformers").glob("*.json")):
        r = json.loads(f.read_text())
        for test_key, t in r["tests"].items():
            rows.append({"model": PRETTY.get(r["model"], r["model"]), "train": r["train"], "test": test_key,
                         **{k: t[k] for k in ("macro_f1", "macro_f1_lo", "macro_f1_hi", "accuracy", "n")}})
    return pd.DataFrame(rows)


def table(df: pd.DataFrame) -> str:
    df = df.copy()
    df["setting"] = [SETTING_LABEL[(a, b)] for a, b in zip(df.train, df.test)]
    df["Macro-F1 [95% CI]"] = [f"{m:.3f} [{lo:.3f}, {hi:.3f}]" for m, lo, hi in zip(df.macro_f1, df.macro_f1_lo, df.macro_f1_hi)]
    df["Accuracy"] = df.accuracy.map(lambda a: f"{a:.3f}")
    df["model"] = pd.Categorical(df.model, [m for m in ORDER if m in set(df.model)] + sorted(set(df.model) - set(ORDER)))
    df = df.sort_values(["model", "train", "test"])
    return df[["model", "setting", "Macro-F1 [95% CI]", "Accuracy"]].rename(columns={"model": "Model", "setting": "Train → Test"}).to_markdown(index=False)


def bar_chart(df: pd.DataFrame) -> None:
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": AXIS,
                         "axes.spines.top": False, "axes.spines.right": False, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.labelcolor": INK2, "font.size": 10})
    models = [m for m in ORDER if m in set(df.model)]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    x = np.arange(len(SETTINGS))
    trained = [m for m in models if m != "Loughran-McDonald lexicon"]
    width = 0.8 / max(1, len(trained))
    for i, m in enumerate(trained):
        vals, lo, hi = [], [], []
        for tr, te in SETTINGS:
            r = df[(df.model == m) & (df.train == tr) & (df.test == te)]
            vals.append(r.macro_f1.iloc[0] if len(r) else np.nan)
            lo.append(r.macro_f1_lo.iloc[0] if len(r) else np.nan)
            hi.append(r.macro_f1_hi.iloc[0] if len(r) else np.nan)
        vals, lo, hi = map(np.array, (vals, lo, hi))
        xs = x - 0.4 + width * (i + 0.5)
        ax.bar(xs, vals, width=width * 0.92, color=COLOURS.get(m, MUTED), label=m, edgecolor=SURFACE, linewidth=1)
        ax.errorbar(xs, vals, yerr=[vals - lo, hi - vals], fmt="none", ecolor=INK2, elinewidth=1, capsize=2)
        for xx, v, h in zip(xs, vals, hi):
            if not np.isnan(v):
                ax.text(xx, h + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=INK2)
    lex = df[df.model == "Loughran-McDonald lexicon"].set_index("test").macro_f1
    for j, (tr, te) in enumerate(SETTINGS):
        ax.hlines(lex[te], j - 0.45, j + 0.45, color=COLOURS["Loughran-McDonald lexicon"], lw=2, ls="--",
                  label="Loughran-McDonald lexicon (no training)" if j == 0 else None)
    ax.set_xticks(x, [SETTING_LABEL[s] for s in SETTINGS])
    ax.axvline(1.5, color=GRID, lw=1)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Macro-F1 on test set (95% bootstrap CI)")
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Financial sentiment: in-domain vs cross-domain performance", loc="left", fontweight="bold", color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(FIG / "macro_f1_by_setting.png", dpi=160)
    plt.close(fig)


def confusion_figure() -> None:
    conf = json.loads((RESULTS / "confusions_baselines.json").read_text())
    keys = [("tfidf_phrasebank|phrasebank", "TF-IDF, News → News"), ("tfidf_phrasebank|twitter", "TF-IDF, News → Twitter")]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    labels = ["neg", "neu", "pos"]
    for ax, (k, title) in zip(axes, keys):
        m = np.array(conf[k])
        ax.imshow(m, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{m[i, j]:.0%}", ha="center", va="center", color="#ffffff" if m[i, j] > 0.55 else INK, fontsize=10)
        ax.set_xticks(range(3), labels)
        ax.set_yticks(range(3), labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("Where the news-trained model breaks on tweets (row-normalised)", x=0.01, ha="left", color=INK2, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "confusion_news_model.png", dpi=160)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    df = collect()
    df.to_csv(RESULTS / "all_results.csv", index=False)
    (RESULTS / "summary.md").write_text(table(df) + "\n")
    bar_chart(df)
    confusion_figure()
    print(table(df))


if __name__ == "__main__":
    main()
