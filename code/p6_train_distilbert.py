"""
p6_train_distilbert.py - Fine-tune DistilBERT for dominant psychosocial response (7-way).

Advisor-specified training regime:
  - class-weighted cross-entropy (weights from p4_split), one set per train set (A/B/C)
  - lr 2e-5, batch 16, weight_decay 0.01, warmup_ratio 0.10, max_grad_norm 1.0
  - up to 12 epochs, early stopping on val macro-F1 (patience 2, min improvement 0.005)
  - max_length chosen from the 99th percentile of TRAIN token lengths (128 vs 256)
  - 5 seeds [13, 42, 73, 101, 2026]; report mean +/- std macro-F1 (NOT best seed)
  - metrics: macro-F1 (primary), weighted-F1, accuracy, per-class P/R/F1/support + predicted_count,
    confusion matrix, minority macro-F1 = mean(F1 anger, denial, acceptance)
  - colored tqdm progress bar

This script handles ONLY the response (emotion) task. Distress is a separate model (p7).

USAGE (Windows, local venv, RTX 5090):

    py p6_train_distilbert.py ^
        --train  data\\gen_v6_low_medium\\splits\\train_A.jsonl ^
        --val    data\\gen_v6_low_medium\\splits\\validation.jsonl ^
        --test   data\\gen_v6_low_medium\\splits\\test.jsonl ^
        --weights data\\gen_v6_low_medium\\splits\\class_weights_A.json ^
        --out    models\\distilbert_response_A ^
        --tag    A

For the unweighted control (advisor section 4): add --no-weights and use a different --out.
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
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    set_seed as hf_set_seed,
)
from sklearn.metrics import (
    f1_score,
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
)


# =============================================================================
# Configuration
# =============================================================================

RESPONSE_LABELS = [
    "anxiety", "sadness", "anger", "hope", "guilt", "denial", "acceptance",
]
LABEL_TO_ID = {label: i for i, label in enumerate(RESPONSE_LABELS)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}

MINORITY_LABELS = ["anger", "denial", "acceptance"]

MODEL_NAME = "distilbert-base-uncased"
SEEDS = [13, 42, 73, 101, 2026]

WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
MAX_GRAD_NORM = 1.0
EARLY_STOP_THRESHOLD = 0.005

BAR_COLOUR = "green"


# =============================================================================
# Data
# =============================================================================

def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_text(record):
    value = record.get("text")
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise ValueError("Record has no usable 'text' field.")


def get_response_label(record):
    value = record.get("final_response")
    if isinstance(value, str) and value.strip().lower() in LABEL_TO_ID:
        return value.strip().lower()
    raise ValueError(
        "Record has no valid 'final_response': {}".format(record.get("id", "<unknown>"))
    )


def choose_max_length(records, tokenizer):
    """Pick max_length from the 99th percentile of TRAIN token lengths: 128 or 256."""
    lengths = []
    for r in records:
        ids = tokenizer(get_text(r), truncation=False)["input_ids"]
        lengths.append(len(ids))
    p99 = int(np.percentile(lengths, 99))
    p95 = int(np.percentile(lengths, 95))
    chosen = 128 if p99 <= 128 else 256
    print("Token lengths: max={}  p95={}  p99={}  -> max_length={}".format(
        max(lengths), p95, p99, chosen
    ))
    return chosen


class ResponseDataset(Dataset):
    def __init__(self, records, tokenizer, max_length):
        self.texts = [get_text(r) for r in records]
        self.labels = [LABEL_TO_ID[get_response_label(r)] for r in records]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        enc = self.tokenizer(
            self.texts[index],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[index], dtype=torch.long),
        }


def load_class_weights(path, device):
    if not path:
        return None
    with open(path, encoding="utf-8") as f:
        wd = json.load(f)
    weights = [float(wd.get(label, 1.0)) for label in RESPONSE_LABELS]
    return torch.tensor(weights, dtype=torch.float, device=device)


# =============================================================================
# Metrics
# =============================================================================

def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, labels=list(range(len(RESPONSE_LABELS))),
                    average="macro", zero_division=0)


def minority_macro_f1(y_true, y_pred):
    pc = f1_score(y_true, y_pred, labels=list(range(len(RESPONSE_LABELS))),
                  average=None, zero_division=0)
    ids = [LABEL_TO_ID[l] for l in MINORITY_LABELS]
    return float(np.mean([pc[i] for i in ids]))


def full_metrics(y_true, y_pred):
    label_ids = list(range(len(RESPONSE_LABELS)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=label_ids, average=None, zero_division=0
    )
    # predicted_count per class (advisor: detect over-prediction)
    pred_counts = [int(np.sum(np.array(y_pred) == i)) for i in label_ids]

    per_class = {}
    for i, label in enumerate(RESPONSE_LABELS):
        per_class[label] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
            "predicted_count": pred_counts[i],
        }
    cm = confusion_matrix(y_true, y_pred, labels=label_ids).astype(int).tolist()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(macro_f1(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=label_ids,
                                      average="weighted", zero_division=0)),
        "minority_macro_f1": minority_macro_f1(y_true, y_pred),
        "minority_classes": MINORITY_LABELS,
        "per_class": per_class,
        "confusion_matrix": {"labels": RESPONSE_LABELS, "rows_true_cols_pred": cm},
        "n": len(y_true),
    }


def print_metrics(title, m):
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)
    print("n={}  accuracy={:.4f}  macro-F1={:.4f}  weighted-F1={:.4f}".format(
        m["n"], m["accuracy"], m["macro_f1"], m["weighted_f1"]))
    print("minority macro-F1 (anger,denial,acceptance): {:.4f}".format(m["minority_macro_f1"]))
    print("\nper-class (P / R / F1 / support / predicted):")
    for label in RESPONSE_LABELS:
        c = m["per_class"][label]
        print("  {:11s} {:.3f} / {:.3f} / {:.3f} / {:d} / {:d}".format(
            label, c["precision"], c["recall"], c["f1"], c["support"], c["predicted_count"]))


# =============================================================================
# Evaluation pass
# =============================================================================

def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(labels.numpy().tolist())
    return y_true, y_pred


# =============================================================================
# Train one seed
# =============================================================================

def train_one_seed(seed, args, train_ds, val_ds, test_ds, class_weights, device, seed_out_dir):
    hf_set_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(RESPONSE_LABELS),
        id2label=ID_TO_LABEL, label2id=LABEL_TO_ID,
    )
    model.to(device)

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(WARMUP_RATIO * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_val_f1 = -1.0
    best_epoch = -1
    epochs_without_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        bar = tqdm(train_loader, desc="seed {} ep {}/{}".format(seed, epoch, args.epochs),
                   colour=BAR_COLOUR, ncols=100)
        for batch in bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            bar.set_postfix(loss="{:.4f}".format(loss.item()))

        avg_loss = running_loss / max(1, len(train_loader))
        y_true, y_pred = evaluate(model, val_loader, device)
        val_f1 = macro_f1(y_true, y_pred)
        print("  seed {} epoch {}: train_loss={:.4f} val_macro_f1={:.4f}".format(
            seed, epoch, avg_loss, val_f1))

        if val_f1 > best_val_f1 + EARLY_STOP_THRESHOLD:
            best_val_f1 = val_f1
            best_epoch = epoch
            epochs_without_improve = 0
            model.save_pretrained(seed_out_dir)
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= args.patience:
                print("  seed {} early stop at epoch {}".format(seed, epoch))
                break

    # load best checkpoint for this seed, evaluate once on test
    best_model = AutoModelForSequenceClassification.from_pretrained(seed_out_dir)
    best_model.to(device)
    y_true, y_pred = evaluate(best_model, test_loader, device)
    metrics = full_metrics(y_true, y_pred)
    metrics["seed"] = seed
    metrics["best_val_macro_f1"] = float(best_val_f1)
    metrics["best_epoch"] = int(best_epoch)
    return metrics, y_true, y_pred


# =============================================================================
# Main: loop seeds, aggregate mean +/- std
# =============================================================================

def main():
    p = argparse.ArgumentParser(description="Fine-tune DistilBERT (7-way response), 5 seeds.")
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--weights", default=None, help="class_weights_*.json")
    p.add_argument("--no-weights", action="store_true", help="unweighted control (advisor sec 4)")
    p.add_argument("--out", required=True)
    p.add_argument("--tag", default="A")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(
        torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_records = read_jsonl(args.train)
    val_records = read_jsonl(args.val)
    test_records = read_jsonl(args.test)
    print("train: {}  val: {}  test: {}".format(
        len(train_records), len(val_records), len(test_records)))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    max_length = choose_max_length(train_records, tokenizer)

    train_ds = ResponseDataset(train_records, tokenizer, max_length)
    val_ds = ResponseDataset(val_records, tokenizer, max_length)
    test_ds = ResponseDataset(test_records, tokenizer, max_length)

    weights_path = None if args.no_weights else args.weights
    class_weights = load_class_weights(weights_path, device)
    if class_weights is not None:
        print("class weights: {}".format(
            {RESPONSE_LABELS[i]: round(float(class_weights[i]), 3)
             for i in range(len(RESPONSE_LABELS))}))
    else:
        print("class weights: NONE (unweighted control)")

    all_seed_metrics = []
    per_seed_macro = []
    per_seed_minority = []
    per_seed_acc = []
    per_seed_weighted = []

    for seed in SEEDS:
        print("\n" + "#" * 64)
        print("SEED {}".format(seed))
        print("#" * 64)
        seed_out_dir = out_dir / "seed_{}".format(seed)
        seed_out_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(seed_out_dir)

        metrics, y_true, y_pred = train_one_seed(
            seed, args, train_ds, val_ds, test_ds, class_weights, device, seed_out_dir)

        print_metrics("SEED {} TEST METRICS (exp {})".format(seed, args.tag), metrics)
        all_seed_metrics.append(metrics)
        per_seed_macro.append(metrics["macro_f1"])
        per_seed_minority.append(metrics["minority_macro_f1"])
        per_seed_acc.append(metrics["accuracy"])
        per_seed_weighted.append(metrics["weighted_f1"])

        # save this seed's predictions for later bootstrap
        with open(seed_out_dir / "test_predictions.json", "w", encoding="utf-8") as f:
            json.dump({"y_true": y_true, "y_pred": y_pred, "labels": RESPONSE_LABELS},
                      f, indent=2)

    def mean_std(xs):
        return float(statistics.mean(xs)), float(statistics.pstdev(xs))

    macro_m, macro_s = mean_std(per_seed_macro)
    minor_m, minor_s = mean_std(per_seed_minority)
    acc_m, acc_s = mean_std(per_seed_acc)
    wtd_m, wtd_s = mean_std(per_seed_weighted)

    summary = {
        "experiment_tag": args.tag,
        "weighted_loss": class_weights is not None,
        "seeds": SEEDS,
        "train_size": len(train_records),
        "max_length": max_length,
        "macro_f1_mean": macro_m, "macro_f1_std": macro_s,
        "minority_macro_f1_mean": minor_m, "minority_macro_f1_std": minor_s,
        "accuracy_mean": acc_m, "accuracy_std": acc_s,
        "weighted_f1_mean": wtd_m, "weighted_f1_std": wtd_s,
        "per_seed_macro_f1": per_seed_macro,
        "per_seed_metrics": all_seed_metrics,
    }

    print("\n" + "=" * 64)
    print("AGGREGATE OVER {} SEEDS (exp {})".format(len(SEEDS), args.tag))
    print("=" * 64)
    print("macro-F1:          {:.4f} +/- {:.4f}".format(macro_m, macro_s))
    print("minority macro-F1: {:.4f} +/- {:.4f}".format(minor_m, minor_s))
    print("accuracy:          {:.4f} +/- {:.4f}".format(acc_m, acc_s))
    print("weighted-F1:       {:.4f} +/- {:.4f}".format(wtd_m, wtd_s))
    print("per-seed macro-F1: {}".format([round(x, 4) for x in per_seed_macro]))

    summary_path = out_dir / "summary_{}.json".format(args.tag)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\nSaved aggregate summary to: {}".format(summary_path))
    print("Per-seed models + predictions under: {}".format(out_dir))


if __name__ == "__main__":
    main()
