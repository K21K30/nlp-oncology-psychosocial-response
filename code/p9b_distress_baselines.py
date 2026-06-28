"""
p9b_distress_baselines.py - Distress baselines (advisor): majority, TF-IDF multinomial,
TF-IDF ORDINAL (two binary models), zero-shot BART (two averaged templates).

Distress is ordinal: low=0 < medium=1 < high=2. Same ordinal-aware metrics as p7
(macro-F1, linear weighted kappa, MAE, severe/adjacent rates, per-level, confusion, off-diagonal).

Ordinal TF-IDF (advisor section 9):
  model_1: P(distress > low)   -> binary {low} vs {medium,high}
  model_2: P(distress > medium)-> binary {low,medium} vs {high}
  decision (thresholds tuned on VALIDATION only):
    if P(>low) < t1: low ; elif P(>medium) >= t2: high ; else medium
  monotonic correction: P(>medium) = min(P(>medium), P(>low))   (>medium can't exceed >low)

USAGE (from project root, on desktop):

  # supervised distress baselines for one train set (majority + TF-IDF nominal + TF-IDF ordinal):
  py p9b_distress_baselines.py --mode supervised ^
     --train data\\gen_v6_low_medium\\splits\\train_C.jsonl ^
     --val   data\\gen_v6_low_medium\\splits\\validation.jsonl ^
     --test  data\\gen_v6_low_medium\\splits\\test.jsonl ^
     --weights data\\gen_v6_low_medium\\splits\\distress_weights_C.json ^
     --tag C --out results\\distress_baselines

  # zero-shot distress (two templates, averaged; downloads BART if not cached):
  py p9b_distress_baselines.py --mode zeroshot ^
     --test data\\gen_v6_low_medium\\splits\\test.jsonl ^
     --out results\\distress_baselines
"""

import argparse
import json
from pathlib import Path
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, precision_recall_fscore_support, accuracy_score,
    confusion_matrix, cohen_kappa_score,
)


LEVELS = ["low", "medium", "high"]
LAB2ID = {l: i for i, l in enumerate(LEVELS)}
IDS = list(range(len(LEVELS)))

# zero-shot: two parallel templates, probabilities averaged (advisor section 8)
ZS_TEMPLATES = [
    {
        "low": "The person expresses low emotional distress.",
        "medium": "The person expresses moderate emotional distress.",
        "high": "The person expresses high emotional distress.",
    },
    {
        "low": "The person is mostly calm or only mildly distressed.",
        "medium": "The person is clearly distressed but still able to cope.",
        "high": "The person is overwhelmed or unable to cope with the distress.",
    },
]


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def texts_labels(rows):
    X = [r["text"].strip() for r in rows]
    y = [LAB2ID[r["final_distress"].strip().lower()] for r in rows]
    return X, y


# =============================================================================
# Ordinal-aware metrics (same as p7)
# =============================================================================

def ordinal_metrics(y_true, y_pred):
    yt, yp = np.array(y_true), np.array(y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        yt, yp, labels=IDS, average=None, zero_division=0)
    pred_counts = [int(np.sum(yp == i)) for i in IDS]
    per_level = {}
    for i, lab in enumerate(LEVELS):
        per_level[lab] = {"precision": float(precision[i]), "recall": float(recall[i]),
                          "f1": float(f1[i]), "support": int(support[i]),
                          "predicted_count": pred_counts[i]}
    cm = confusion_matrix(yt, yp, labels=IDS)
    abs_err = np.abs(yt - yp)
    n = len(yt)

    def cell(t, p):
        return int(cm[LAB2ID[t]][LAB2ID[p]])

    return {
        "macro_f1": float(f1_score(yt, yp, labels=IDS, average="macro", zero_division=0)),
        "weighted_kappa_linear": float(cohen_kappa_score(yt, yp, labels=IDS, weights="linear")),
        "mean_abs_ordinal_error": float(np.mean(abs_err)),
        "accuracy": float(accuracy_score(yt, yp)),
        "weighted_f1": float(f1_score(yt, yp, labels=IDS, average="weighted", zero_division=0)),
        "exact_count": int(np.sum(abs_err == 0)),
        "adjacent_error_count": int(np.sum(abs_err == 1)),
        "severe_error_count": int(np.sum(abs_err == 2)),
        "adjacent_error_rate": float(np.sum(abs_err == 1) / n),
        "severe_error_rate": float(np.sum(abs_err == 2) / n),
        "per_level": per_level,
        "off_diagonal_cells": {
            "medium->low": cell("medium", "low"), "low->medium": cell("low", "medium"),
            "high->medium": cell("high", "medium"), "medium->high": cell("medium", "high"),
            "low->high": cell("low", "high"), "high->low": cell("high", "low"),
        },
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


def save(out_dir, name, m):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(out_dir) / "{}.json".format(name), "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, ensure_ascii=False)
    print("saved: {}/{}.json".format(out_dir, name))


# =============================================================================
# Supervised: majority + TF-IDF nominal + TF-IDF ordinal
# =============================================================================

def run_supervised(args):
    train = read_jsonl(args.train)
    val = read_jsonl(args.val)
    test = read_jsonl(args.test)
    Xtr, ytr = texts_labels(train)
    Xva, yva = texts_labels(val)
    Xte, yte = texts_labels(test)

    # ---- Majority ----
    maj = Counter(ytr).most_common(1)[0][0]
    m = ordinal_metrics(yte, [maj] * len(yte))
    print_metrics("MAJORITY distress (train {}, predicts '{}')".format(args.tag, LEVELS[maj]), m)
    m["baseline"] = "majority"; m["train_tag"] = args.tag; m["predicts"] = LEVELS[maj]
    save(args.out, "distress_majority_{}".format(args.tag), m)

    # ---- weights ----
    cw = None
    if args.weights:
        with open(args.weights, encoding="utf-8") as f:
            wd = json.load(f)
        cw = {LAB2ID[k]: float(v) for k, v in wd.items() if k in LAB2ID}

    # shared vectorizer
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents="unicode")
    Xtr_v = vec.fit_transform(Xtr)
    Xva_v = vec.transform(Xva)
    Xte_v = vec.transform(Xte)

    # ---- TF-IDF NOMINAL (multinomial) ----
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight=cw)
    clf.fit(Xtr_v, ytr)
    yp_nom = clf.predict(Xte_v).tolist()
    m_nom = ordinal_metrics(yte, yp_nom)
    print_metrics("TF-IDF NOMINAL distress (train {})".format(args.tag), m_nom)
    m_nom["baseline"] = "tfidf_nominal"; m_nom["train_tag"] = args.tag
    save(args.out, "distress_tfidf_nominal_{}".format(args.tag), m_nom)

    # ---- TF-IDF ORDINAL (two binary models) ----
    # model_1: P(distress > low) -> positive = {medium, high}
    y_gt_low_tr = [1 if y > LAB2ID["low"] else 0 for y in ytr]
    # model_2: P(distress > medium) -> positive = {high}
    y_gt_med_tr = [1 if y > LAB2ID["medium"] else 0 for y in ytr]

    clf1 = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    clf2 = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    clf1.fit(Xtr_v, y_gt_low_tr)
    clf2.fit(Xtr_v, y_gt_med_tr)

    def prob_pos(clf, X):
        # probability of the positive class (label 1)
        idx = list(clf.classes_).index(1)
        return clf.predict_proba(X)[:, idx]

    # tune thresholds t1, t2 on VALIDATION by best macro-F1
    p_gt_low_va = prob_pos(clf1, Xva_v)
    p_gt_med_va = prob_pos(clf2, Xva_v)
    # monotonic correction: P(>med) <= P(>low)
    p_gt_med_va = np.minimum(p_gt_med_va, p_gt_low_va)

    def ordinal_decode(p_low, p_med, t1, t2):
        out = []
        for pl, pm in zip(p_low, p_med):
            if pl < t1:
                out.append(LAB2ID["low"])
            elif pm >= t2:
                out.append(LAB2ID["high"])
            else:
                out.append(LAB2ID["medium"])
        return out

    grid = [round(x, 2) for x in np.arange(0.30, 0.71, 0.05)]
    best = (-1.0, 0.5, 0.5)
    for t1 in grid:
        for t2 in grid:
            yp_va = ordinal_decode(p_gt_low_va, p_gt_med_va, t1, t2)
            f = f1_score(yva, yp_va, labels=IDS, average="macro", zero_division=0)
            if f > best[0]:
                best = (f, t1, t2)
    _, t1, t2 = best
    print("\n  ordinal thresholds tuned on validation: t1(>low)={}  t2(>med)={}  (val macroF1={:.3f})".format(
        t1, t2, best[0]))

    p_gt_low_te = prob_pos(clf1, Xte_v)
    p_gt_med_te = prob_pos(clf2, Xte_v)
    n_violations = int(np.sum(p_gt_med_te > p_gt_low_te))
    p_gt_med_te = np.minimum(p_gt_med_te, p_gt_low_te)  # monotonic correction
    yp_ord = ordinal_decode(p_gt_low_te, p_gt_med_te, t1, t2)

    m_ord = ordinal_metrics(yte, yp_ord)
    print_metrics("TF-IDF ORDINAL distress (train {})".format(args.tag), m_ord)
    print("  monotonicity violations corrected on test: {}".format(n_violations))
    m_ord["baseline"] = "tfidf_ordinal"; m_ord["train_tag"] = args.tag
    m_ord["thresholds"] = {"t1_gt_low": t1, "t2_gt_med": t2}
    m_ord["monotonic_violations_corrected"] = n_violations
    save(args.out, "distress_tfidf_ordinal_{}".format(args.tag), m_ord)


# =============================================================================
# Zero-shot distress (two templates averaged)
# =============================================================================

def run_zeroshot(args):
    import torch
    from transformers import pipeline
    from tqdm import tqdm

    test = read_jsonl(args.test)
    Xte, yte = texts_labels(test)

    device = 0 if torch.cuda.is_available() else -1
    print("zero-shot device: {}".format(torch.cuda.get_device_name(0) if device == 0 else "CPU"))
    clf = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)

    # For each template, get P(level) per item; average across the two templates.
    def scores_for_template(tmpl):
        hyp_list = [tmpl[l] for l in LEVELS]
        hyp_to_level = {tmpl[l]: l for l in LEVELS}
        per_item = []
        for text in Xte:
            out = clf(text, hyp_list, multi_label=False)
            # out['labels'] and out['scores'] are aligned, sorted by score desc
            d = {hyp_to_level[lab]: sc for lab, sc in zip(out["labels"], out["scores"])}
            per_item.append([d[l] for l in LEVELS])  # prob per level in fixed order
        return np.array(per_item)

    all_scores = np.zeros((len(Xte), len(LEVELS)))
    for ti, tmpl in enumerate(ZS_TEMPLATES):
        print("template {}/{}...".format(ti + 1, len(ZS_TEMPLATES)))
        s = np.zeros((len(Xte), len(LEVELS)))
        hyp_list = [tmpl[l] for l in LEVELS]
        hyp_to_level = {tmpl[l]: l for l in LEVELS}
        for i, text in enumerate(tqdm(Xte, desc="zs t{}".format(ti + 1), colour="cyan", ncols=100)):
            out = clf(text, hyp_list, multi_label=False)
            d = {hyp_to_level[lab]: sc for lab, sc in zip(out["labels"], out["scores"])}
            s[i] = [d[l] for l in LEVELS]
        all_scores += s
    all_scores /= len(ZS_TEMPLATES)
    yp = np.argmax(all_scores, axis=1).tolist()

    m = ordinal_metrics(yte, yp)
    print_metrics("ZERO-SHOT distress bart-mnli (2 templates averaged)", m)
    m["baseline"] = "zeroshot_bart_2tmpl"
    save(args.out, "distress_zeroshot_bart", m)


def main():
    p = argparse.ArgumentParser(description="Distress baselines.")
    p.add_argument("--mode", required=True, choices=["supervised", "zeroshot"])
    p.add_argument("--train")
    p.add_argument("--val")
    p.add_argument("--test", required=True)
    p.add_argument("--weights", default=None)
    p.add_argument("--tag", default="C")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    if args.mode == "supervised":
        if not args.train or not args.val:
            raise SystemExit("--train and --val required in supervised mode (val for thresholds)")
        run_supervised(args)
    else:
        run_zeroshot(args)


if __name__ == "__main__":
    main()
