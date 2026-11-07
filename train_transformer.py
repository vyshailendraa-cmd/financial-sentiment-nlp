"""Fine-tune a transformer on one domain and evaluate it on both test sets.

Examples (GPU recommended, ~2-5 min each on a free Colab T4):
    python train_transformer.py --model yiyanghkust/finbert-pretrain --train phrasebank
    python train_transformer.py --model distilbert-base-uncased --train twitter
Offline smoke test of the training loop (tiny random model, CPU, seconds):
    python train_transformer.py --smoke --train phrasebank
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from finsent.data import load_all
from finsent.metrics import confusion, scores
from finsent.transformer import build_smoke_model, finetune, load_pretrained, predict

RESULTS = Path(__file__).resolve().parent / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yiyanghkust/finbert-pretrain")
    ap.add_argument("--train", choices=["phrasebank", "twitter"], required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="offline test with a tiny random model")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    domains = load_all()
    src = domains[args.train]

    if args.smoke:
        tok, model = build_smoke_model(src.train.text)
        name, args.epochs, args.lr = "smoke-tiny-bert", 2, 1e-3
    else:
        tok, model = load_pretrained(args.model)
        name = args.model

    print(f"Fine-tuning {name} on {args.train} ({len(src.train)} examples) on {device}")
    t0 = time.perf_counter()
    model, history = finetune(model, tok, src.train, src.val, device, epochs=args.epochs, lr=args.lr,
                              batch_size=args.batch_size, seed=args.seed)
    train_seconds = time.perf_counter() - t0

    out = {"model": name, "train": args.train, "device": device, "train_seconds": train_seconds,
           "history": history, "tests": {}}
    for test_key, d in domains.items():
        p = predict(model, tok, d.test.text.tolist(), device)
        out["tests"][test_key] = {**scores(d.test.y, p), "confusion": confusion(d.test.y, p).tolist(),
                                  "predictions": p.tolist()}
        print(f"  test on {test_key}: macro-F1 {out['tests'][test_key]['macro_f1']:.4f}")

    sub = RESULTS / ("smoke" if args.smoke else "transformers")
    sub.mkdir(parents=True, exist_ok=True)
    fname = f"{name.replace('/', '_')}__train_{args.train}.json"
    (sub / fname).write_text(json.dumps(out, indent=2))
    print("wrote", sub / fname)


if __name__ == "__main__":
    main()
