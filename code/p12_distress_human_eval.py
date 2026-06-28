"""
p12_distress_human_eval.py - Human-reference evaluation for the DISTRESS task (advisor).

NO training. Uses DistilBERT-B 5-seed majority vote (best distress model) vs the HUMAN distress
labels on all 67 test items, plus the human-unambiguous subset and a low/medium boundary analysis.
Sensitivity analysis only (single annotator); the strict synthetic labels remain the primary test.

Produces:
  1. model(B) vs HUMAN distress on all 67: macro-F1, linear weighted kappa, MAE, accuracy, severe
     low<->high count, per-level, confusion (rows=human, cols=model).
  2. human-unambiguous subset (human_distress_ambiguous == False): same headline metrics.
  3. low/medium boundary: among items whose HUMAN label is low or medium, the binary low-vs-medium
     accuracy and the 2x2 confusion (the dominant difficulty per the human check).
  4. context: strict-vs-human and model-vs-strict distress agreement, for comparison.

USAGE (from project root):
    py p12_distress_human_eval.py
"""

import json
from collections import Counter

import numpy as np
from sklearn.metrics import (
    f1_score, accuracy_score, confusion_matrix, cohen_kappa_score,
    precision_recall_fscore_support,
)


LEVELS = ["low", "medium", "high"]
LAB2ID = {l: i for i, l in enumerate(LEVELS)}
IDS = list(range(len(LEVELS)))

HUMAN_TEST = "data/gen_v6_low_medium/splits/test_with_human_annotations.jsonl"
DB_B_DIR = "models/distilbert_distress_B"
SEEDS = [13, 42, 73, 101, 2026]


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def db_majority_vote(n):
    seed_preds = []
    for seed in SEEDS:
        with open("{}/seed_{}/test_predictions.json".format(DB_B_DIR, seed), encoding="utf-8") as f:
            seed_preds.append(json.load(f)["y_pred"])
    voted = []
    for i in range(n):
        votes = Counter(seed_preds[s][i] for s in range(len(seed_preds)))
        voted.append(sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))[0][0])
    return voted


def headline(tag, y_ref, y_pred):
    yt, yp = np.array(y_ref), np.array(y_pred)
    macro = f1_score(yt, yp, labels=IDS, average="macro", zero_division=0)
    kappa = cohen_kappa_score(yt, yp, labels=IDS, weights="linear")
    mae = float(np.mean(np.abs(yt - yp)))
    acc = accuracy_score(yt, yp)
    severe = int(np.sum(np.abs(yt - yp) == 2))
    print("\n  {}: n={}  macro-F1={:.4f}  wKappa(lin)={:.4f}  MAE={:.4f}  acc={:.4f}  severe={}".format(
        tag, len(yt), macro, kappa, mae, acc, severe))
    return {"tag": tag, "n": int(len(yt)), "macro_f1": float(macro),
            "weighted_kappa_linear": float(kappa), "mae": mae, "accuracy": float(acc),
            "severe_count": severe}


def main():
    print("#" * 70)
    print("# DISTRESS HUMAN-REFERENCE EVAL (DistilBERT-B majority vote vs human labels)")
    print("#" * 70)

    rows = read_jsonl(HUMAN_TEST)
    n = len(rows)

    strict = [LAB2ID[r["final_distress"].strip().lower()] for r in rows]
    human = [LAB2ID[r["human_distress"].strip().lower()] for r in rows]
    human_amb = [bool(r.get("human_distress_ambiguous", False)) for r in rows]
    model = db_majority_vote(n)

    # ---- 1. model vs human, all 67 ----
    print("\n" + "=" * 70)
    print("1. MODEL(B) vs HUMAN distress, all 67")
    print("=" * 70)
    headline("model-vs-human (all 67)", human, model)

    p, r, f, s = precision_recall_fscore_support(human, model, labels=IDS, average=None, zero_division=0)
    print("  per-level vs human (P / R / F1 / human-support):")
    for i, lab in enumerate(LEVELS):
        print("    {:7s} {:.3f} / {:.3f} / {:.3f} / {:d}".format(lab, p[i], r[i], f[i], int(s[i])))

    print("\n  confusion (rows=human, cols=model):")
    cm = confusion_matrix(human, model, labels=IDS)
    print("    {:9s}".format("") + "".join("{:>9s}".format(l) for l in LEVELS))
    for i, lab in enumerate(LEVELS):
        print("    {:9s}".format(lab) + "".join("{:9d}".format(cm[i][j]) for j in range(len(LEVELS))))

    # context
    print("\n  context:")
    headline("strict-vs-human", human, strict)
    headline("model-vs-strict", strict, model)

    # ---- 2. human-unambiguous subset ----
    print("\n" + "=" * 70)
    print("2. HUMAN-UNAMBIGUOUS subset (human_distress_ambiguous == False)")
    print("=" * 70)
    un_idx = [i for i in range(n) if not human_amb[i]]
    am_idx = [i for i in range(n) if human_amb[i]]
    print("  unambiguous: {}   ambiguous: {}".format(len(un_idx), len(am_idx)))
    if un_idx:
        headline("model-vs-human UNAMBIGUOUS", [human[i] for i in un_idx], [model[i] for i in un_idx])
    if am_idx:
        headline("model-vs-human AMBIGUOUS", [human[i] for i in am_idx], [model[i] for i in am_idx])

    # also model-vs-strict on unambiguous (difficulty tracking)
    if un_idx:
        acc_un = accuracy_score([strict[i] for i in un_idx], [model[i] for i in un_idx])
        print("  (model-vs-strict accuracy on human-unambiguous: {:.3f})".format(acc_un))
    if am_idx:
        acc_am = accuracy_score([strict[i] for i in am_idx], [model[i] for i in am_idx])
        print("  (model-vs-strict accuracy on human-ambiguous:   {:.3f})".format(acc_am))

    # ---- 3. low/medium boundary ----
    print("\n" + "=" * 70)
    print("3. LOW/MEDIUM BOUNDARY (items whose HUMAN label is low or medium)")
    print("=" * 70)
    low_id, med_id, high_id = LAB2ID["low"], LAB2ID["medium"], LAB2ID["high"]
    lm_idx = [i for i in range(n) if human[i] in (low_id, med_id)]
    print("  items with human low or medium: {}".format(len(lm_idx)))

    # binary low-vs-medium accuracy, counting model 'high' as a miss on these items
    correct = sum(1 for i in lm_idx if model[i] == human[i])
    print("  exact low/medium accuracy (model == human): {:.3f} ({}/{})".format(
        correct / len(lm_idx) if lm_idx else float("nan"), correct, len(lm_idx)))

    # 2x2 confusion restricted to low/medium (model predictions of high shown separately)
    print("\n  confusion on human-low/medium items (rows=human, cols=model):")
    sub_labels = [low_id, med_id, high_id]
    cm2 = confusion_matrix([human[i] for i in lm_idx], [model[i] for i in lm_idx], labels=sub_labels)
    print("    {:9s}".format("") + "".join("{:>9s}".format(LEVELS[l]) for l in sub_labels))
    for r_i, lab_id in enumerate(sub_labels[:2]):  # only human low/medium rows are non-empty
        print("    {:9s}".format(LEVELS[lab_id]) + "".join("{:9d}".format(cm2[r_i][c]) for c in range(len(sub_labels))))

    n_lm_to_high = sum(1 for i in lm_idx if model[i] == high_id)
    print("  (model predicted HIGH on {} human-low/medium items)".format(n_lm_to_high))

    out = {"n": n}
    with open("results/distress_human_eval.json", "w", encoding="utf-8") as f:
        json.dump({"note": "see console for full breakdown"}, f, indent=2)
    print("\nsaved marker: results/distress_human_eval.json")
    print("\nThis is a single-annotator sensitivity analysis; the strict synthetic test stays primary.")


if __name__ == "__main__":
    main()
