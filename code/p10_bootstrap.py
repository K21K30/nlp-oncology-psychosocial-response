"""
p10_bootstrap.py - Stratified paired bootstrap for response-task model comparison (advisor sec 4).

Main comparison: TF-IDF-B (best overall) vs DistilBERT-C (best transformer).
Also: DistilBERT-C vs DistilBERT-A; TF-IDF-B vs TF-IDF-A; DistilBERT-C vs zero-shot BART.

Design (per advisor):
  - 5000 stratified bootstrap resamples of the 67-item test: within each of the 7 true classes,
    resample WITH replacement to that class's exact support (11,9,11,11,11,8,6), then concatenate.
    This keeps every class present so macro-F1 is well-defined every iteration.
  - PAIRED: the same resampled indices are used for both systems in a given iteration.
  - For DistilBERT (5 seeds, no single prediction), compare TF-IDF against EACH of the 5 seeds and
    report the range; also compare against the 5-seed majority-vote ensemble for a single number.
  - Report: mean macro-F1 difference, 95% percentile CI, and fraction of resamples where system X
    beats system Y. (The fraction is NOT a p-value; it is the share of resamples where one wins.)

Predictions:
  - DistilBERT A/C: loaded from saved per-seed test_predictions.json.
  - TF-IDF A/B and zero-shot BART: recomputed here (deterministic), replicating p9 exactly, so no
    changes to p9 are needed.

USAGE (from project root, on the desktop; zero-shot downloads BART if not cached):
    py p10_bootstrap.py
"""

import json
from pathlib import Path
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


LABELS = ["anxiety", "sadness", "anger", "hope", "guilt", "denial", "acceptance"]
LAB2ID = {l: i for i, l in enumerate(LABELS)}
LABEL_IDS = list(range(len(LABELS)))

N_BOOT = 5000
RNG_SEED = 12345

SPLITS = "data/gen_v6_low_medium/splits"
TEST = SPLITS + "/test.jsonl"
TRAIN = {"A": SPLITS + "/train_A.jsonl", "B": SPLITS + "/train_B.jsonl"}
WEIGHTS = {"A": SPLITS + "/class_weights_A.json", "B": SPLITS + "/class_weights_B.json"}
DISTILBERT_DIR = {"A": "models/distilbert_response_A", "C": "models/distilbert_response_C"}
SEEDS = [13, 42, 73, 101, 2026]

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


# =============================================================================
# Build all predictions (aligned to the SAME test order)
# =============================================================================

def get_y_true(test_rows):
    return [LAB2ID[r["final_response"].strip().lower()] for r in test_rows]


def tfidf_predict(tag, test_rows):
    """Replicate p9 TF-IDF + weighted LogReg exactly for a given train tag."""
    train = read_jsonl(TRAIN[tag])
    Xtr = [r["text"].strip() for r in train]
    ytr = [LAB2ID[r["final_response"].strip().lower()] for r in train]
    Xte = [r["text"].strip() for r in test_rows]

    with open(WEIGHTS[tag], encoding="utf-8") as f:
        wd = json.load(f)
    cw = {LAB2ID[k]: float(v) for k, v in wd.items() if k in LAB2ID}

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                          strip_accents="unicode")
    Xtr_v = vec.fit_transform(Xtr)
    Xte_v = vec.transform(Xte)
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight=cw)
    clf.fit(Xtr_v, ytr)
    return clf.predict(Xte_v).tolist()


def distilbert_seed_preds(tag):
    """Return list of 5 prediction lists (one per seed), aligned to test order."""
    out = []
    for seed in SEEDS:
        path = "{}/seed_{}/test_predictions.json".format(DISTILBERT_DIR[tag], seed)
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        out.append(d["y_pred"])
    return out


def majority_vote(seed_preds, n):
    voted = []
    for i in range(n):
        votes = Counter(seed_preds[s][i] for s in range(len(seed_preds)))
        voted.append(sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))[0][0])
    return voted


def zeroshot_predict(test_rows):
    import torch
    from transformers import pipeline
    from tqdm import tqdm
    device = 0 if torch.cuda.is_available() else -1
    clf = pipeline("zero-shot-classification",
                   model="facebook/bart-large-mnli", device=device)
    hyp_list = [ZS_HYPOTHESES[l] for l in LABELS]
    hyp_to_label = {ZS_HYPOTHESES[l]: l for l in LABELS}
    preds = []
    for r in tqdm(test_rows, desc="zero-shot", colour="green", ncols=100):
        out = clf(r["text"].strip(), hyp_list, multi_label=False)
        preds.append(LAB2ID[hyp_to_label[out["labels"][0]]])
    return preds


# =============================================================================
# Stratified bootstrap
# =============================================================================

def build_strata(y_true):
    """indices grouped by true class."""
    strata = {c: [] for c in LABEL_IDS}
    for i, y in enumerate(y_true):
        strata[y].append(i)
    return strata


def stratified_indices(strata, rng):
    """One stratified resample: per class, sample with replacement to its support."""
    idx = []
    for c in LABEL_IDS:
        members = strata[c]
        if members:
            idx.extend(rng.choice(members, size=len(members), replace=True).tolist())
    return idx


def macro_f1_on(idx, y_true, y_pred):
    yt = [y_true[i] for i in idx]
    yp = [y_pred[i] for i in idx]
    return f1_score(yt, yp, labels=LABEL_IDS, average="macro", zero_division=0)


def paired_bootstrap(name_x, pred_x, name_y, pred_y, y_true, rng):
    """Paired stratified bootstrap of macro-F1 difference (X - Y)."""
    strata = build_strata(y_true)
    diffs = []
    x_scores = []
    y_scores = []
    for _ in range(N_BOOT):
        idx = stratified_indices(strata, rng)
        fx = macro_f1_on(idx, y_true, pred_x)
        fy = macro_f1_on(idx, y_true, pred_y)
        x_scores.append(fx)
        y_scores.append(fy)
        diffs.append(fx - fy)
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p_x_gt_y = float(np.mean(diffs > 0))
    print("\n  {} vs {}:".format(name_x, name_y))
    print("    mean macro-F1: {} = {:.4f}, {} = {:.4f}".format(
        name_x, float(np.mean(x_scores)), name_y, float(np.mean(y_scores))))
    print("    mean diff ({} - {}): {:+.4f}   95% CI [{:+.4f}, {:+.4f}]".format(
        name_x, name_y, float(np.mean(diffs)), lo, hi))
    print("    P({} > {}) across resamples: {:.3f}".format(name_x, name_y, p_x_gt_y))
    ci_excludes_zero = (lo > 0) or (hi < 0)
    print("    95% CI {} zero -> difference is {}".format(
        "EXCLUDES" if ci_excludes_zero else "includes",
        "consistent" if ci_excludes_zero else "not distinguishable at 95%"))
    return {"x": name_x, "y": name_y, "mean_diff": float(np.mean(diffs)),
            "ci_low": float(lo), "ci_high": float(hi), "p_x_gt_y": p_x_gt_y,
            "ci_excludes_zero": bool(ci_excludes_zero)}


# =============================================================================
# Main
# =============================================================================

def main():
    print("#" * 70)
    print("# STRATIFIED PAIRED BOOTSTRAP (response, {} resamples)".format(N_BOOT))
    print("#" * 70)

    test_rows = read_jsonl(TEST)
    y_true = get_y_true(test_rows)
    n = len(y_true)
    print("test n = {}  class supports = {}".format(
        n, {LABELS[c]: sum(1 for y in y_true if y == c) for c in LABEL_IDS}))

    print("\nbuilding predictions...")
    tfidf_B = tfidf_predict("B", test_rows)
    tfidf_A = tfidf_predict("A", test_rows)
    db_C_seeds = distilbert_seed_preds("C")
    db_A_seeds = distilbert_seed_preds("A")
    db_C_vote = majority_vote(db_C_seeds, n)
    db_A_vote = majority_vote(db_A_seeds, n)

    print("running zero-shot BART (this is the slow part)...")
    zs = zeroshot_predict(test_rows)

    # sanity: full-test macro-F1 of each system (should match earlier runs)
    def mf(pred):
        return f1_score(y_true, pred, labels=LABEL_IDS, average="macro", zero_division=0)
    print("\nfull-test macro-F1 (sanity check vs earlier runs):")
    print("  TF-IDF-B            : {:.4f}".format(mf(tfidf_B)))
    print("  TF-IDF-A            : {:.4f}".format(mf(tfidf_A)))
    print("  DistilBERT-C (vote) : {:.4f}".format(mf(db_C_vote)))
    print("  DistilBERT-A (vote) : {:.4f}".format(mf(db_A_vote)))
    print("  zero-shot BART      : {:.4f}".format(mf(zs)))
    print("  DistilBERT-C per-seed: {}".format([round(mf(p), 4) for p in db_C_seeds]))

    rng = np.random.default_rng(RNG_SEED)
    results = []

    # ---- MAIN: TF-IDF-B vs DistilBERT-C ----
    print("\n" + "=" * 70)
    print("MAIN COMPARISON: TF-IDF-B vs DistilBERT-C")
    print("=" * 70)
    print("\n  (a) vs DistilBERT-C 5-seed MAJORITY-VOTE ensemble:")
    results.append(paired_bootstrap("TF-IDF-B", tfidf_B, "DistilBERT-C(vote)", db_C_vote, y_true, rng))
    print("\n  (b) vs EACH DistilBERT-C seed (range of outcomes):")
    for s, seed in enumerate(SEEDS):
        results.append(paired_bootstrap(
            "TF-IDF-B", tfidf_B, "DistilBERT-C(seed {})".format(seed), db_C_seeds[s], y_true, rng))

    # ---- within-family and vs zero-shot ----
    print("\n" + "=" * 70)
    print("SECONDARY COMPARISONS")
    print("=" * 70)
    results.append(paired_bootstrap("DistilBERT-C(vote)", db_C_vote,
                                    "DistilBERT-A(vote)", db_A_vote, y_true, rng))
    results.append(paired_bootstrap("TF-IDF-B", tfidf_B, "TF-IDF-A", tfidf_A, y_true, rng))
    results.append(paired_bootstrap("DistilBERT-C(vote)", db_C_vote,
                                    "zero-shot-BART", zs, y_true, rng))
    results.append(paired_bootstrap("TF-IDF-B", tfidf_B, "zero-shot-BART", zs, y_true, rng))

    out = {"n_boot": N_BOOT, "n_test": n, "comparisons": results}
    Path("results/bootstrap").mkdir(parents=True, exist_ok=True)
    with open("results/bootstrap/bootstrap_response.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nsaved: results/bootstrap/bootstrap_response.json")
    print("\nNote: P(X>Y) is the share of resamples where X wins, NOT a classical p-value.")


if __name__ == "__main__":
    main()
