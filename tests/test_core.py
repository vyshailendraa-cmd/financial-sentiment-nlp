import numpy as np

from finsent.data import load_all, normalise
from finsent.metrics import scores


def test_normalise_replaces_urls_and_cashtags():
    assert normalise("$AAPL beats https://t.co/x  estimates") == "TICKER beats URL estimates"


def test_splits_are_disjoint_and_cover_three_labels():
    for d in load_all().values():
        assert not set(d.train.text) & set(d.test.text)
        assert not set(d.val.text) & set(d.test.text)
        assert set(d.test.y) == {0, 1, 2}


def test_scores_perfect_prediction():
    y = np.array([0, 1, 2, 1, 0, 2])
    s = scores(y, y, n_boot=50)
    assert s["macro_f1"] == 1.0 and s["accuracy"] == 1.0


def test_smoke_transformer_trains():
    import torch

    from finsent.transformer import build_smoke_model, finetune

    d = load_all()["phrasebank"]
    tok, model = build_smoke_model(d.train.text)
    model, hist = finetune(model, tok, d.train.head(64), d.val.head(32), "cpu", epochs=1, lr=1e-3, batch_size=16)
    assert len(hist) == 1 and np.isfinite(hist[0]["train_loss"])
    assert isinstance(next(model.parameters()), torch.Tensor)
