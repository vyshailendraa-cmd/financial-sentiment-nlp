"""Fine-tune a pretrained transformer for 3-class sentiment with a plain PyTorch loop.

Needs internet access to Hugging Face to download the pretrained weights, so the
reference results were produced on Google Colab (see notebooks/colab_finetune.ipynb).
`--smoke` runs the identical loop on a tiny randomly initialised BERT with a
locally trained WordPiece tokeniser, so the code path can be tested offline.
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from .data import LABELS


def build_smoke_model(texts, num_labels: int = 3):
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
    from transformers import BertConfig, BertForSequenceClassification, PreTrainedTokenizerFast

    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.normalizer = normalizers.BertNormalizer(lowercase=True)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    trainer = trainers.WordPieceTrainer(vocab_size=4000, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
    tok.train_from_iterator(list(texts), trainer)
    from tokenizers.processors import TemplateProcessing

    tok.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", tok.token_to_id("[CLS]")), ("[SEP]", tok.token_to_id("[SEP]"))]
    )
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]"
    )
    cfg = BertConfig(vocab_size=hf_tok.vocab_size, hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=128, num_labels=num_labels)
    return hf_tok, BertForSequenceClassification(cfg)


def load_pretrained(name: str, num_labels: int = 3):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(name)
    except ValueError:
        # Some older BERT checkpoints (e.g. yiyanghkust/finbert-pretrain) ship only a
        # vocab.txt. Recent transformers versions can fail to convert those, so build
        # the fast WordPiece tokenizer directly from the vocabulary file.
        from huggingface_hub import hf_hub_download
        from transformers import BertTokenizerFast

        path = hf_hub_download(name, "vocab.txt")
        with open(path, encoding="utf-8") as f:
            vocab = {line.rstrip("\n"): i for i, line in enumerate(f)}
        # transformers v5 takes `vocab=` and silently ignores `vocab_file=` (which left
        # a 5-token vocab, so ~93% of tokens became [UNK]); v4 needs `vocab_file=`.
        try:
            tok = BertTokenizerFast(vocab=vocab, do_lower_case=True)
        except TypeError:
            tok = BertTokenizerFast(vocab_file=path, do_lower_case=True)
        if len(tok) < 1000:
            tok = BertTokenizerFast(vocab_file=path, do_lower_case=True)
    if len(tok) < 1000:
        raise RuntimeError(f"{name}: tokenizer has only {len(tok)} tokens; vocab failed to load")
    kwargs = dict(num_labels=num_labels, ignore_mismatched_sizes=True,
                  id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)})
    try:
        model = AutoModelForSequenceClassification.from_pretrained(name, **kwargs)
    except ValueError:
        # Older BERT checkpoints may omit "model_type" from config.json; load as BERT explicitly.
        from transformers import BertForSequenceClassification

        model = BertForSequenceClassification.from_pretrained(name, **kwargs)
    return tok, model


def _loader(tok, texts, labels, batch_size, shuffle, max_len, seed=0):
    enc = tok(list(texts), truncation=True, max_length=max_len, padding=False)
    items = [{"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i], "labels": int(labels[i])}
             for i in range(len(texts))]

    def collate(batch):
        L = max(len(b["input_ids"]) for b in batch)
        pad = tok.pad_token_id
        ids = torch.tensor([b["input_ids"] + [pad] * (L - len(b["input_ids"])) for b in batch])
        mask = torch.tensor([b["attention_mask"] + [0] * (L - len(b["attention_mask"])) for b in batch])
        return {"input_ids": ids, "attention_mask": mask, "labels": torch.tensor([b["labels"] for b in batch])}

    g = torch.Generator().manual_seed(seed)
    return DataLoader(items, batch_size=batch_size, shuffle=shuffle, collate_fn=collate, generator=g)


@torch.no_grad()
def predict(model, tok, texts, device, batch_size: int = 64, max_len: int = 96) -> np.ndarray:
    model.eval()
    dl = _loader(tok, texts, np.zeros(len(texts), dtype=int), batch_size, False, max_len)
    preds = []
    for b in dl:
        b = {k: v.to(device) for k, v in b.items() if k != "labels"}
        preds.append(model(**b).logits.argmax(-1).cpu().numpy())
    return np.concatenate(preds)


def finetune(model, tok, train, val, device, epochs: int = 4, lr: float = 2e-5, batch_size: int = 16,
             max_len: int = 96, warmup: float = 0.1, seed: int = 42, log=print):
    """AdamW + linear warm-up/decay. The checkpoint with the best validation
    macro-F1 is kept (early stopping by selection)."""
    torch.manual_seed(seed)
    model.to(device)
    dl = _loader(tok, train.text.tolist(), train.y.to_numpy(), batch_size, True, max_len, seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total = epochs * len(dl)
    n_warm = int(warmup * total)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, n_warm) if s < n_warm else max(0.0, (total - s) / max(1, total - n_warm))
    )
    best_f1, best_state, history = -1.0, None, []
    for ep in range(epochs):
        model.train()
        t0, running = time.perf_counter(), 0.0
        for b in dl:
            b = {k: v.to(device) for k, v in b.items()}
            loss = model(**b).loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item()
        vf1 = f1_score(val.y, predict(model, tok, val.text.tolist(), device, max_len=max_len), average="macro")
        history.append({"epoch": ep + 1, "train_loss": running / len(dl), "val_macro_f1": vf1})
        log(f"  epoch {ep + 1}: loss {running / len(dl):.4f}  val macro-F1 {vf1:.4f}  ({time.perf_counter() - t0:.0f}s)")
        if vf1 > best_f1:
            best_f1, best_state = vf1, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, history
