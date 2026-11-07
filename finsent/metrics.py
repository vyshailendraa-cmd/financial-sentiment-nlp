from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def scores(y_true, y_pred, n_boot: int = 1000, seed: int = 0) -> dict[str, float]:
    """Macro-F1 and accuracy, with a 95% bootstrap interval on macro-F1."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        boots.append(f1_score(y_true[i], y_pred[i], average="macro"))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "macro_f1_lo": lo,
        "macro_f1_hi": hi,
        "accuracy": accuracy_score(y_true, y_pred),
        "n": n,
    }


def confusion(y_true, y_pred) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=[0, 1, 2], normalize="true")
