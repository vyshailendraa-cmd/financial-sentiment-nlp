"""Two financial sentiment corpora with fixed, reproducible splits.

* Financial PhraseBank, sentences with 100% annotator agreement (Malo et al., 2014).
  Formal sentences from English-language financial news. CC BY-NC-SA 3.0.
* Twitter Financial News Sentiment (zeroshot, 2022). Finance-related tweets
  labelled bearish / bullish / neutral, mapped here to negative / positive / neutral. MIT.

Every split has the same three labels, so any model can be trained on one domain
and tested on the other.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA = Path(__file__).resolve().parents[1] / "data"
LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
SEED = 42

_URL = re.compile(r"https?://\S+")
_TICKER = re.compile(r"\$[A-Za-z][A-Za-z.]{0,6}\b")


def normalise(text: str) -> str:
    """Light normalisation shared by every model: URLs and cashtags become placeholders."""
    text = _URL.sub(" URL ", text)
    text = _TICKER.sub(" TICKER ", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class Domain:
    name: str
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["text", "label"]).drop_duplicates(subset="text").copy()
    df["text"] = df["text"].astype(str).map(normalise)
    df["y"] = df["label"].map(LABEL2ID)
    return df.reset_index(drop=True)[["text", "label", "y"]]


def load_phrasebank() -> Domain:
    df = _prep(pd.read_csv(DATA / "phrasebank_allagree.csv"))
    train, rest = train_test_split(df, test_size=0.4, stratify=df.y, random_state=SEED)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest.y, random_state=SEED)
    return Domain("PhraseBank (news)", train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True))


def load_twitter() -> Domain:
    """The official validation file is used as the test set; 15% of the official
    training file is held out for model selection."""
    full = _prep(pd.read_csv(DATA / "twitter_train.csv"))
    test = _prep(pd.read_csv(DATA / "twitter_valid.csv"))
    test = test[~test.text.isin(full.text)].reset_index(drop=True)  # no train/test leakage
    train, val = train_test_split(full, test_size=0.15, stratify=full.y, random_state=SEED)
    return Domain("Twitter (social)", train.reset_index(drop=True), val.reset_index(drop=True), test)


def load_all() -> dict[str, Domain]:
    return {"phrasebank": load_phrasebank(), "twitter": load_twitter()}
