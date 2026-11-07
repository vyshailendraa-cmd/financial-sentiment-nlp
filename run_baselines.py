"""Lexicon and TF-IDF baselines, in-domain and cross-domain.

Writes results/baselines.csv, results/top_features.json and the confusion matrices.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from finsent.baselines import LexiconClassifier, TfidfLogReg
from finsent.data import load_all
from finsent.metrics import confusion, scores

RESULTS = Path(__file__).resolve().parent / "results"


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    domains = load_all()
    rows, confusions, preds = [], {}, {}

    lex = LexiconClassifier()
    for test_key, d in domains.items():
        p = lex.predict(d.test.text)
        rows.append({"model": "Loughran-McDonald lexicon", "train": "none", "test": test_key, **scores(d.test.y, p)})
        confusions[f"lexicon|{test_key}"] = confusion(d.test.y, p).tolist()
        preds[f"lexicon|{test_key}"] = p.tolist()

    top = {}
    for train_key, src in domains.items():
        clf = TfidfLogReg().fit(src.train.text, src.train.y, src.val.text, src.val.y)
        top[train_key] = clf.top_features()
        for test_key, d in domains.items():
            p = clf.predict(d.test.text)
            rows.append({"model": "TF-IDF + logistic regression", "train": train_key, "test": test_key, **scores(d.test.y, p)})
            confusions[f"tfidf_{train_key}|{test_key}"] = confusion(d.test.y, p).tolist()
            preds[f"tfidf_{train_key}|{test_key}"] = p.tolist()

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "baselines.csv", index=False)
    (RESULTS / "top_features.json").write_text(json.dumps(top, indent=2))
    (RESULTS / "confusions_baselines.json").write_text(json.dumps(confusions))
    (RESULTS / "predictions_baselines.json").write_text(json.dumps(preds))
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
