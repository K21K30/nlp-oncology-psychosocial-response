"""
p7a_distress_weights.py - Compute distress class weights per train set (advisor section 7).

w_c = N / (K * n_c), K = 3 distress levels, cap 5.0. Computed on final_distress, separately for
train_A / train_B / train_C. Writes distress_weights_{A,B,C}.json next to the splits.

USAGE (from project root):
    py p7a_distress_weights.py
"""

import json
from pathlib import Path
from collections import Counter

DISTRESS_LEVELS = ["low", "medium", "high"]
K = 3
CAP = 5.0

SPLITS = "data/gen_v6_low_medium/splits"
TRAIN = {"A": SPLITS + "/train_A.jsonl",
         "B": SPLITS + "/train_B.jsonl",
         "C": SPLITS + "/train_C.jsonl"}


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_weights(rows):
    labels = [r["final_distress"].strip().lower() for r in rows]
    counts = Counter(labels)
    n = len(labels)
    weights = {}
    for level in DISTRESS_LEVELS:
        n_c = counts.get(level, 0)
        if n_c == 0:
            weights[level] = CAP  # absent class -> cap (shouldn't happen, but safe)
        else:
            w = n / (K * n_c)
            weights[level] = round(min(w, CAP), 6)
    return weights, dict(counts), n


def main():
    for tag, path in TRAIN.items():
        rows = read_jsonl(path)
        weights, counts, n = compute_weights(rows)
        out_path = Path(SPLITS) / "distress_weights_{}.json".format(tag)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(weights, f, indent=2)
        print("Train {}: N={}  distress counts {}".format(tag, n, counts))
        print("  weights (w_c = N/(3*n_c), cap {}): {}".format(CAP, weights))
        cap_hit = [lvl for lvl, w in weights.items() if w >= CAP]
        if cap_hit:
            print("  NOTE: cap active for {}".format(cap_hit))
        print("  saved: {}".format(out_path))
        print()


if __name__ == "__main__":
    main()
