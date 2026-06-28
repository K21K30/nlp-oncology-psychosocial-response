"""
p9_baselines.py - Response-task baselines (advisor section 7), evaluated on the frozen 67-item
strict test set with the same metrics as the DistilBERT models.

Baselines:
  1. Majority-class: always predict the most frequent train class. Trivial lower bound.
  2. TF-IDF + weighted Logistic Regression: simple supervised baseline on the same train set, with
     the same per-class weights (class_weight from the weights json). Shows whether the neural
     fine-tune beats a classic linear model.
  3. Zero-shot facebook/bart-large-mnli: NLI zero-shot over the 7 classes, NO training on our data.
     Shows what task-specific fine-tuning adds over zero-shot inference.

Metrics (same as p6): macro-F1 (primary), weighted-F1, accuracy, per-class P/R/F1/support +
predicted_count, minority macro-F1 (anger, denial, acceptance), confusion matrix.

USAGE (from project root, run on the desktop):

  # Majority + TF-IDF for one train set (fast, CPU is fine):
  py p9_baselines.py --mode supervised ^
      --train data\\gen_v6_low_medium\\splits\\train_C.jsonl ^
      --test  data\\gen_v6_low_medium\\splits\\test.jsonl ^
      --weights data\\gen_v6_low_medium\\splits\\class_weights_C.json ^
      --tag C ^
      --out results\\baselines

  # Zero-shot BART (downloads ~1.6GB first run; uses GPU if available):
  py p9_baselines.py --mode zeroshot ^
      --test data\\gen_v6_low_medium\\splits\\test.jsonl ^
      --out results\\baselines

Run supervised once per train set (A/B/C) to get majority + TF-IDF for each. Zero-shot is run once
(it does not use any train set).
"""

import argparse
import json
from pathlib import Path
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, precision_recall_fscore_support, accuracy_score, confusion_matrix,
)


LABELS = ["anxiety", "sadness", "anger", "hope", "guilt", "denial", "acceptance"]
LAB2ID = {l: i for i, l in enumerate(LABELS)}
MINORITY = ["anger", "denial", "acceptance"]

# Natural-language hypotheses for the zero-shot NLI head (one per class).
# These describe the EXPRESSED reaction, matching the project definition.
ZS_HYPOTHESES = {
    "anxiety": "This message expresses anxiety, fear, worry, or panic.",
    "sadness": "This message expresses sadness, grief, sorrow, or loss.",
    "anger": "This message expresses anger, frustration, or hostility.",
    "hope": "This message expresses hope or optimism about a better outcome.",
    "guilt": "This message expresses guilt, self-blame, or regret.",
    "denial": "This message expresses denial, disbelief, or avoidance of reality.",
    "acceptance": "This message expresses calm acceptance of the situation.",
}


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
    return LAB2ID[r["final_response"].strip().lower()]


# =============================================================================
# Metrics (same shape as p6)
# =============================================================================

def full_metrics(y_true, y_pred):
    ids = list(range(len(LABELS)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=ids, average=None, zero_division=0)
    pred_counts = [int(np.sum(np.array(y_pred) == i)) for i in ids]
    per_class = {}
    for i, lab in enumerate(LABELS):
        per_class[lab] = {
            "precision": float(precision[i]), "recall": float(recall[i]),
            "f1": float(f1[i]), "support": int(support[i]),
            "predicted_count": pred_counts[i],
        }
    minority_ids = [LAB2ID[l] for l in MINORITY]
    per_all = f1_score(y_true, y_pred, labels=ids, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=ids, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=ids, average="weighted", zero_division=0)),
        "minority_macro_f1": float(np.mean([per_all[i] for i in minority_ids])),
        "per_class": per_class,
        "confusion_matrix": {"labels": LABELS,
                             "rows_true_cols_pred": confusion_matrix(y_true, y_pred, labels=ids).tolist()},
        "n": len(y_true),
    }


def print_metrics(title, m):
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)
    print("n={}  accuracy={:.4f}  macro-F1={:.4f}  weighted-F1={:.4f}  minority-F1={:.4f}".format(
        m["n"], m["accuracy"], m["macro_f1"], m["weighted_f1"], m["minority_macro_f1"]))
    print("per-class (P / R / F1 / support / predicted):")
    for lab in LABELS:
        c = m["per_class"][lab]
        print("  {:11s} {:.3f} / {:.3f} / {:.3f} / {:d} / {:d}".format(
            lab, c["precision"], c["recall"], c["f1"], c["support"], c["predicted_count"]))


def save_metrics(out_dir, name, m):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / "{}.json".format(name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, ensure_ascii=False)
    print("saved: {}".format(path))


# =============================================================================
# Supervised baselines: majority + TF-IDF/LogReg
# =============================================================================

def run_supervised(args):
    train = read_jsonl(args.train)
    test = read_jsonl(args.test)

    X_train = [get_text(r) for r in train]
    y_train = [get_label(r) for r in train]
    X_test = [get_text(r) for r in test]
    y_test = [get_label(r) for r in test]

    # ---- Majority ----
    majority_id = Counter(y_train).most_common(1)[0][0]
    y_pred_maj = [majority_id] * len(y_test)
    m_maj = full_metrics(y_test, y_pred_maj)
    print_metrics("MAJORITY baseline (train {}, predicts '{}')".format(
        args.tag, LABELS[majority_id]), m_maj)
    m_maj["baseline"] = "majority"
    m_maj["train_tag"] = args.tag
    m_maj["predicts"] = LABELS[majority_id]
    save_metrics(args.out, "majority_{}".format(args.tag), m_maj)

    # ---- TF-IDF + weighted Logistic Regression ----
    class_weight = None
    if args.weights:
        with open(args.weights, encoding="utf-8") as f:
            wd = json.load(f)
        class_weight = {LAB2ID[k]: float(v) for k, v in wd.items() if k in LAB2ID}

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents="unicode")
    Xtr = vectorizer.fit_transform(X_train)
    Xte = vectorizer.transform(X_test)

    clf = LogisticRegression(
        max_iter=2000, C=1.0, class_weight=class_weight, multi_class="auto")
    clf.fit(Xtr, y_train)
    y_pred_lr = clf.predict(Xte).tolist()

    m_lr = full_metrics(y_test, y_pred_lr)
    print_metrics("TF-IDF + weighted LogReg (train {})".format(args.tag), m_lr)
    m_lr["baseline"] = "tfidf_logreg"
    m_lr["train_tag"] = args.tag
    m_lr["weighted"] = class_weight is not None
    save_metrics(args.out, "tfidf_logreg_{}".format(args.tag), m_lr)


# =============================================================================
# Zero-shot BART (no training)
# =============================================================================

def run_zeroshot(args):
    import torch
    from transformers import pipeline

    test = read_jsonl(args.test)
    X_test = [get_text(r) for r in test]
    y_test = [get_label(r) for r in test]

    device = 0 if torch.cuda.is_available() else -1
    print("Zero-shot device: {}".format(
        torch.cuda.get_device_name(0) if device == 0 else "CPU"))

    clf = pipeline("zero-shot-classification",
                   model="facebook/bart-large-mnli", device=device)

    # candidate labels = the hypotheses; map the winning hypothesis back to its class
    hyp_list = [ZS_HYPOTHESES[l] for l in LABELS]
    hyp_to_label = {ZS_HYPOTHESES[l]: l for l in LABELS}

    y_pred = []
    from tqdm import tqdm
    for text in tqdm(X_test, desc="zero-shot", colour="green", ncols=100):
        out = clf(text, hyp_list, multi_label=False)
        top_hyp = out["labels"][0]
        y_pred.append(LAB2ID[hyp_to_label[top_hyp]])

    m = full_metrics(y_test, y_pred)
    print_metrics("ZERO-SHOT bart-large-mnli (no training)", m)
    m["baseline"] = "zeroshot_bart_mnli"
    save_metrics(args.out, "zeroshot_bart", m)


# =============================================================================
# CLI
# =============================================================================

def main():
    p = argparse.ArgumentParser(description="Response-task baselines.")
    p.add_argument("--mode", required=True, choices=["supervised", "zeroshot"])
    p.add_argument("--train", help="train jsonl (supervised mode)")
    p.add_argument("--test", required=True)
    p.add_argument("--weights", default=None, help="class_weights_*.json (supervised TF-IDF)")
    p.add_argument("--tag", default="C", help="train tag for supervised mode")
    p.add_argument("--out", required=True, help="output dir for metric json files")
    args = p.parse_args()

    if args.mode == "supervised":
        if not args.train:
            raise SystemExit("--train is required in supervised mode")
        run_supervised(args)
    else:
        run_zeroshot(args)


if __name__ == "__main__":
    main()
