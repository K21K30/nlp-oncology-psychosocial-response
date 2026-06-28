"""
p7_train_distress.py - Fine-tune DistilBERT for distress intensity (3-level ordinal: low/medium/high).

Mirrors p6 (response) but for the 3-level distress task, with ordinal-aware evaluation per advisor:
  PRIMARY metric: macro-F1 (model selection by validation macro-F1, like response).
  PRIMARY ordinal companion: linear weighted Cohen's kappa.
  Plus: mean absolute ordinal error, accuracy, weighted-F1, per-level P/R/F1/support, confusion,
  severe error rate (low<->high), adjacent error rate (low<->medium + medium<->high), and the
  specific off-diagonal cells. 5 seeds, class-weighted CE, same regime as p6.

Distress levels are ordered: low=0 < medium=1 < high=2 (so ordinal distance is meaningful).

USAGE (from project root, RTX 5090):
  py p7_train_distress.py ^
    --train data\\gen_v6_low_medium\\splits\\train_C.jsonl ^
    --val   data\\gen_v6_low_medium\\splits\\validation.jsonl ^
    --test  data\\gen_v6_low_medium\\splits\\test.jsonl ^
    --weights data\\gen_v6_low_medium\\splits\\distress_weights_C.json ^
    --out   models\\distilbert_distress_C ^
    --tag   C

For the unweighted control: add --no-weights and a different --out (advisor: only best vs its
unweighted needed, not all three).
"""

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup, set_seed as hf_set_seed,
)
from sklearn.metrics import (
    f1_score, precision_recall_fscore_support, accuracy_score,
    confusion_matrix, cohen_kappa_score,
)


# Ordered distress levels: index = ordinal rank
LEVELS = ["low", "medium", "high"]
LAB2ID = {l: i for i, l in enumerate(LEVELS)}
ID2LAB = {i: l for l, i in LAB2ID.items()}

MODEL_NAME = "distilbert-base-uncased"
SEEDS = [13, 42, 73, 101, 2026]
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
MAX_GRAD_NORM = 1.0
EARLY_STOP_THRESHOLD = 0.005
BAR_COLOUR = "cyan"


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_text(r):
    return r["text"].strip()


def get_label(r):
    return LAB2ID[r["final_distress"].strip().lower()]


def choose_max_length(records, tokenizer):
    lengths = [len(tokenizer(get_text(r), truncation=False)["input_ids"]) for r in records]
    p99 = int(np.percentile(lengths, 99))
    chosen = 128 if p99 <= 128 else 256
    print("Token lengths: max={}  p99={}  -> max_length={}".format(max(lengths), p99, chosen))
    return chosen


class DistressDataset(Dataset):
    def __init__(self, records, tokenizer, max_length):
        self.texts = [get_text(r) for r in records]
        self.labels = [get_label(r) for r in records]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tokenizer(self.texts[i], truncation=True, max_length=self.max_length,
                             padding="max_length", return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": torch.tensor(self.labels[i], dtype=torch.long)}


def load_weights(path, device):
    if not path:
        return None
    with open(path, encoding="utf-8") as f:
        wd = json.load(f)
    w = [float(wd.get(l, 1.0)) for l in LEVELS]
    return torch.tensor(w, dtype=torch.float, device=device)


# =============================================================================
# Ordinal-aware metrics
# =============================================================================

def ordinal_metrics(y_true, y_pred):
    ids = list(range(len(LEVELS)))
    yt, yp = np.array(y_true), np.array(y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        yt, yp, labels=ids, average=None, zero_division=0)
    pred_counts = [int(np.sum(yp == i)) for i in ids]
    per_level = {}
    for i, lab in enumerate(LEVELS):
        per_level[lab] = {"precision": float(precision[i]), "recall": float(recall[i]),
                          "f1": float(f1[i]), "support": int(support[i]),
                          "predicted_count": pred_counts[i]}

    cm = confusion_matrix(yt, yp, labels=ids)

    # ordinal distances
    abs_err = np.abs(yt - yp)
    mae = float(np.mean(abs_err))
    n = len(yt)
    # severe: low<->high (distance 2). adjacent: distance 1.
    severe_count = int(np.sum(abs_err == 2))
    adjacent_count = int(np.sum(abs_err == 1))
    exact_count = int(np.sum(abs_err == 0))

    # linear weighted kappa (ordinal); also quadratic for reference
    try:
        kappa_lin = float(cohen_kappa_score(yt, yp, labels=ids, weights="linear"))
    except Exception:
        kappa_lin = float("nan")
    try:
        kappa_quad = float(cohen_kappa_score(yt, yp, labels=ids, weights="quadratic"))
    except Exception:
        kappa_quad = float("nan")

    # specific off-diagonal cells (true -> pred)
    def cell(t, p):
        return int(cm[LAB2ID[t]][LAB2ID[p]])
    cells = {
        "medium->low": cell("medium", "low"),
        "low->medium": cell("low", "medium"),
        "high->medium": cell("high", "medium"),
        "medium->high": cell("medium", "high"),
        "low->high": cell("low", "high"),
        "high->low": cell("high", "low"),
    }

    return {
        "macro_f1": float(f1_score(yt, yp, labels=ids, average="macro", zero_division=0)),
        "weighted_kappa_linear": kappa_lin,
        "weighted_kappa_quadratic": kappa_quad,
        "mean_abs_ordinal_error": mae,
        "accuracy": float(accuracy_score(yt, yp)),
        "weighted_f1": float(f1_score(yt, yp, labels=ids, average="weighted", zero_division=0)),
        "exact_count": exact_count,
        "adjacent_error_count": adjacent_count,
        "severe_error_count": severe_count,
        "adjacent_error_rate": adjacent_count / n,
        "severe_error_rate": severe_count / n,
        "per_level": per_level,
        "off_diagonal_cells": cells,
        "confusion_matrix": {"labels": LEVELS, "rows_true_cols_pred": cm.astype(int).tolist()},
        "n": n,
    }


def print_metrics(title, m):
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)
    print("n={}  macro-F1={:.4f}  wKappa(lin)={:.4f}  MAE={:.4f}  acc={:.4f}".format(
        m["n"], m["macro_f1"], m["weighted_kappa_linear"], m["mean_abs_ordinal_error"], m["accuracy"]))
    print("exact={}  adjacent-err={} ({:.3f})  SEVERE low<->high={} ({:.3f})".format(
        m["exact_count"], m["adjacent_error_count"], m["adjacent_error_rate"],
        m["severe_error_count"], m["severe_error_rate"]))
    print("per-level (P / R / F1 / support / predicted):")
    for lab in LEVELS:
        c = m["per_level"][lab]
        print("  {:7s} {:.3f} / {:.3f} / {:.3f} / {:d} / {:d}".format(
            lab, c["precision"], c["recall"], c["f1"], c["support"], c["predicted_count"]))
    print("off-diagonal: {}".format(m["off_diagonal_cells"]))


def evaluate(model, loader, device):
    model.eval()
    yt, yp = [], []
    with torch.no_grad():
        for b in loader:
            ii = b["input_ids"].to(device)
            am = b["attention_mask"].to(device)
            out = model(input_ids=ii, attention_mask=am)
            yp.extend(torch.argmax(out.logits, dim=-1).cpu().numpy().tolist())
            yt.extend(b["labels"].numpy().tolist())
    return yt, yp


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, labels=list(range(len(LEVELS))), average="macro", zero_division=0)


def train_one_seed(seed, args, train_ds, val_ds, test_ds, weights, device, seed_dir):
    hf_set_seed(seed)
    tl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    vl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    tel = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LEVELS), id2label=ID2LAB, label2id=LAB2ID)
    model.to(device)

    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    total = len(tl) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(WARMUP_RATIO * total), total)

    best_f1, best_ep, no_improve = -1.0, -1, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        run = 0.0
        bar = tqdm(tl, desc="seed {} ep {}/{}".format(seed, ep, args.epochs),
                   colour=BAR_COLOUR, ncols=100)
        for b in bar:
            ii = b["input_ids"].to(device); am = b["attention_mask"].to(device)
            lb = b["labels"].to(device)
            opt.zero_grad()
            out = model(input_ids=ii, attention_mask=am)
            loss = loss_fn(out.logits, lb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            opt.step(); sched.step()
            run += loss.item()
            bar.set_postfix(loss="{:.4f}".format(loss.item()))
        yt, yp = evaluate(model, vl, device)
        vf1 = macro_f1(yt, yp)
        print("  seed {} ep {}: train_loss={:.4f} val_macro_f1={:.4f}".format(
            seed, ep, run / max(1, len(tl)), vf1))
        if vf1 > best_f1 + EARLY_STOP_THRESHOLD:
            best_f1, best_ep, no_improve = vf1, ep, 0
            model.save_pretrained(seed_dir)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print("  seed {} early stop ep {}".format(seed, ep))
                break

    best = AutoModelForSequenceClassification.from_pretrained(seed_dir)
    best.to(device)
    yt, yp = evaluate(best, tel, device)
    m = ordinal_metrics(yt, yp)
    m["seed"] = seed
    m["best_val_macro_f1"] = float(best_f1)
    m["best_epoch"] = int(best_ep)
    return m, yt, yp


def main():
    p = argparse.ArgumentParser(description="Fine-tune DistilBERT for distress (3-level), 5 seeds.")
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--weights", default=None)
    p.add_argument("--no-weights", action="store_true")
    p.add_argument("--out", required=True)
    p.add_argument("--tag", default="A")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = read_jsonl(args.train)
    val = read_jsonl(args.val)
    test = read_jsonl(args.test)
    print("train: {}  val: {}  test: {}".format(len(train), len(val), len(test)))

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    max_len = choose_max_length(train, tok)

    train_ds = DistressDataset(train, tok, max_len)
    val_ds = DistressDataset(val, tok, max_len)
    test_ds = DistressDataset(test, tok, max_len)

    wpath = None if args.no_weights else args.weights
    weights = load_weights(wpath, device)
    if weights is not None:
        print("distress weights: {}".format(
            {LEVELS[i]: round(float(weights[i]), 3) for i in range(len(LEVELS))}))
    else:
        print("distress weights: NONE (unweighted control)")

    all_m = []
    per_macro, per_kappa, per_mae, per_severe = [], [], [], []
    for seed in SEEDS:
        print("\n" + "#" * 64)
        print("SEED {}".format(seed))
        print("#" * 64)
        seed_dir = out_dir / "seed_{}".format(seed)
        seed_dir.mkdir(parents=True, exist_ok=True)
        tok.save_pretrained(seed_dir)
        m, yt, yp = train_one_seed(seed, args, train_ds, val_ds, test_ds, weights, device, seed_dir)
        print_metrics("SEED {} TEST (distress, exp {})".format(seed, args.tag), m)
        all_m.append(m)
        per_macro.append(m["macro_f1"])
        per_kappa.append(m["weighted_kappa_linear"])
        per_mae.append(m["mean_abs_ordinal_error"])
        per_severe.append(m["severe_error_rate"])
        with open(seed_dir / "test_predictions.json", "w", encoding="utf-8") as f:
            json.dump({"y_true": yt, "y_pred": yp, "labels": LEVELS}, f, indent=2)

    def ms(xs):
        return float(statistics.mean(xs)), float(statistics.pstdev(xs))

    macro_m, macro_s = ms(per_macro)
    kappa_m, kappa_s = ms(per_kappa)
    mae_m, mae_s = ms(per_mae)
    sev_m, sev_s = ms(per_severe)

    summary = {
        "experiment_tag": args.tag, "task": "distress",
        "weighted_loss": weights is not None, "seeds": SEEDS,
        "train_size": len(train), "max_length": max_len,
        "macro_f1_mean": macro_m, "macro_f1_std": macro_s,
        "weighted_kappa_linear_mean": kappa_m, "weighted_kappa_linear_std": kappa_s,
        "mean_abs_ordinal_error_mean": mae_m, "mean_abs_ordinal_error_std": mae_s,
        "severe_error_rate_mean": sev_m, "severe_error_rate_std": sev_s,
        "per_seed_macro_f1": per_macro,
        "per_seed_metrics": all_m,
    }
    print("\n" + "=" * 64)
    print("AGGREGATE OVER {} SEEDS (distress, exp {})".format(len(SEEDS), args.tag))
    print("=" * 64)
    print("macro-F1:           {:.4f} +/- {:.4f}".format(macro_m, macro_s))
    print("wKappa(linear):     {:.4f} +/- {:.4f}".format(kappa_m, kappa_s))
    print("MAE(ordinal):       {:.4f} +/- {:.4f}".format(mae_m, mae_s))
    print("severe error rate:  {:.4f} +/- {:.4f}".format(sev_m, sev_s))
    print("per-seed macro-F1:  {}".format([round(x, 4) for x in per_macro]))

    with open(out_dir / "summary_{}.json".format(args.tag), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\nsaved: {}".format(out_dir / "summary_{}.json".format(args.tag)))


if __name__ == "__main__":
    main()
