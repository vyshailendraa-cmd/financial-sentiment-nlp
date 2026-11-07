"""Non-neural baselines: the Loughran-McDonald finance lexicon and TF-IDF + logistic regression."""
from __future__ import annotations

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from .data import LABEL2ID


class LexiconClassifier:
    """Loughran & McDonald (2011) finance word lists. No training: the sign of
    (positive words - negative words) is the prediction, zero means neutral."""

    def __init__(self):
        import pysentiment2 as ps

        self.lm = ps.LM()

    def predict(self, texts) -> np.ndarray:
        out = []
        for t in texts:
            s = self.lm.get_score(self.lm.tokenize(t))
            diff = s["Positive"] - s["Negative"]
            out.append(LABEL2ID["positive"] if diff > 0 else LABEL2ID["negative"] if diff < 0 else LABEL2ID["neutral"])
        return np.array(out)


class TfidfLogReg:
    """Word 1-2 grams + character 2-5 grams, logistic regression. C and class
    weighting are chosen on the validation set by macro-F1."""

    def __init__(self, seed: int = 0):
        self.seed = seed

    def _features(self, texts, fit: bool = False):
        if fit:
            self.word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)
            self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
            return hstack([self.word.fit_transform(texts), self.char.fit_transform(texts)]).tocsr()
        return hstack([self.word.transform(texts), self.char.transform(texts)]).tocsr()

    def fit(self, train_texts, y_train, val_texts, y_val):
        Xtr = self._features(train_texts, fit=True)
        Xva = self._features(val_texts)
        best = (-1.0, None)
        for C in (0.5, 1, 2, 4, 8, 16):
            for cw in (None, "balanced"):
                m = LogisticRegression(C=C, class_weight=cw, max_iter=3000, random_state=self.seed)
                m.fit(Xtr, y_train)
                f = f1_score(y_val, m.predict(Xva), average="macro")
                if f > best[0]:
                    best = (f, m)
        self.model = best[1]
        self.val_f1 = best[0]
        return self

    def predict(self, texts) -> np.ndarray:
        return self.model.predict(self._features(texts))

    def top_features(self, k: int = 12) -> dict[str, list[str]]:
        names = np.array(self.word.get_feature_names_out().tolist() + self.char.get_feature_names_out().tolist())
        n_word = len(self.word.get_feature_names_out())
        coef = self.model.coef_[:, :n_word]  # word features only, more readable
        out = {}
        for label, idx in LABEL2ID.items():
            out[label] = names[:n_word][np.argsort(coef[idx])[::-1][:k]].tolist()
        return out
