"""
p11_distress_bootstrap.py - Stratified paired bootstrap for the DISTRESS task (mirrors p10).

Main comparison: DistilBERT-B (best distress model) vs TF-IDF-nominal-B (best distress baseline).
Note this is the REVERSE of the response task, where TF-IDF was on top.
Secondary: DistilBERT-B vs DistilBERT-A; DistilBERT-B vs zero-shot; TF-IDF-nominal-B vs zero-shot.

Also (advisor's deferred item 5): single-system absolute bootstrap CIs for macro-F1 AND linear
weighted kappa, for DistilBERT-B(vote), TF-IDF-nominal-B, and zero-shot, so the reader sees the
uncertainty of each estimate (not just the differences).

Design identical to p10: 5000 stratified resamples preserving per-level support (low 13, medium 31,
high 23) so macro-F1 is defined each iteration; paired (same indices for both systems). DistilBERT
predictions from saved per-seed files; TF-IDF nominal and zero-shot recomputed (replicating p9b).
P(X>Y) = share of resamples where X wins (NOT a p-value). Percentile CIs.

USAGE (from project root, on desktop; zero-shot uses cached BART):
    py p11_distress_bootstrap.py
"""

import json
from pathlib import Path
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, cohen_kappa_score


LEVELS = ["low", "medium", "high"]
LAB2ID = {l: i for i, l in enumerate(LEVELS)}
IDS = list(range(len(LEVELS)))

N_BOOT = 5000
RNG_SEED = 12345

SPLITS = "data/gen_v6_low_medium/splits"
TEST = SPLITS + "/test.jsonl"
TRAIN = {"A": SPLITS + "/train_A.jsonl", "B": SPLITS + "/train_B.jsonl"}
WEIGHTS = {"A": SPLITS + "/distress_weights_A.json", "B": SPLITS + "/distress_weights_B.json"}
DB_DIR = {"A": "models/distilbert_distress_A", "B": "models/distilbert_distress_B"}
SEEDS = [13, 42, 73, 101, 2026]

ZS_TEMPLATES = [
    {"low": "The person expresses low emotional distress.",
     "medium": "The person expresses moderate emotional distress.",
     "high": "The person expresses high emotional distress."},
    {"low": "The person is mostly calm or only mildly distressed.",
     "medium": "The person is clearly distressed but still able to cope.",
     "high": "The person is overwhelmed or unable to cope with the distress."},
]


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_y_true(rows):
    return [LAB2ID[r["final_distress"].strip().lower()] for r in rows]


def tfidf_nominal_predict(tag, test_rows):
    """Replicate p9b TF-IDF nominal (multinomial, weighted) for a train tag."""
    train = read_jsonl(TRAIN[tag])
    Xtr = [r["text"].strip() for r in train]
    ytr = [LAB2ID[r["final_distress"].strip().lower()] for r in train]
    Xte = [r["text"].strip() for r in test_rows]
    with open(WEIGHTS[tag], encoding="utf-8") as f:
        wd = json.load(f)
    cw = {LAB2ID[k]: float(v) for k, v in wd.items() if k in LAB2ID}
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents="unicode")
    Xtr_v = vec.fit_transform(Xtr)
    Xte_v = vec.transform(Xte)
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight=cw)
    clf.fit(Xtr_v, ytr)
    return clf.predict(Xte_v).tolist()


def db_seed_preds(tag):
    out = []
    for seed in SEEDS:
        with open("{}/seed_{}/test_predictions.json".format(DB_DIR[tag], seed), encoding="utf-8") as f:
            out.append(json.load(f)["y_pred"])
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
    clf = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)
    Xte = [r["text"].strip() for r in test_rows]
    all_scores = np.zeros((len(Xte), len(LEVELS)))
    for ti, tmpl in enumerate(ZS_TEMPLATES):
        hyp_list = [tmpl[l] for l in LEVELS]
        hyp_to_level = {tmpl[l]: l for l in LEVELS}
        for i, text in enumerate(tqdm(Xte, desc="zs t{}".format(ti + 1), colour="cyan", ncols=100)):
            out = clf(text, hyp_list, multi_label=False)
            d = {hyp_to_level[lab]: sc for lab, sc in zip(out["labels"], out["scores"])}
            all_scores[i] += [d[l] for l in LEVELS]
    all_scores /= len(ZS_TEMPLATES)
    return np.argmax(all_scores, axis=1).tolist()


# =============================================================================
# Stratified bootstrap (same as p10)
# =============================================================================

def build_strata(y_true):
    strata = {c: [] for c in IDS}
    for i, y in enumerate(y_true):
        strata[y].append(i)
    return strata


def stratified_indices(strata, rng):
    idx = []
    for c in IDS:
        m = strata[c]
        if m:
            idx.extend(rng.choice(m, size=len(m), replace=True).tolist())
    return idx


def macro_f1_on(idx, y_true, y_pred):
    return f1_score([y_true[i] for i in idx], [y_pred[i] for i in idx],
                    labels=IDS, average="macro", zero_division=0)


def kappa_on(idx, y_true, y_pred):
    return cohen_kappa_score([y_true[i] for i in idx], [y_pred[i] for i in idx],
                             labels=IDS, weights="linear")


def paired_bootstrap(name_x, pred_x, name_y, pred_y, y_true, rng):
    strata = build_strata(y_true)
    diffs, xs, ys = [], [], []
    for _ in range(N_BOOT):
        idx = stratified_indices(strata, rng)
        fx = macro_f1_on(idx, y_true, pred_x)
        fy = macro_f1_on(idx, y_true, pred_y)
        xs.append(fx); ys.append(fy); diffs.append(fx - fy)
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = float(np.mean(diffs > 0))
    excl = (lo > 0) or (hi < 0)
    print("\n  {} vs {}:".format(name_x, name_y))
    print("    mean macro-F1: {} = {:.4f}, {} = {:.4f}".format(
        name_x, float(np.mean(xs)), name_y, float(np.mean(ys))))
    print("    mean diff: {:+.4f}   95% CI [{:+.4f}, {:+.4f}]   P({}>{}): {:.3f}".format(
        float(np.mean(diffs)), lo, hi, name_x, name_y, p))
    print("    95% CI {} zero -> {}".format(
        "EXCLUDES" if excl else "includes",
        "distinguishable" if excl else "NOT distinguishable at 95%"))
    return {"x": name_x, "y": name_y, "mean_diff": float(np.mean(diffs)),
            "ci_low": float(lo), "ci_high": float(hi), "p_x_gt_y": p, "ci_excludes_zero": bool(excl)}


def single_system_ci(name, pred, y_true, rng):
    """Bootstrap CI for one system's absolute macro-F1 AND linear weighted kappa."""
    strata = build_strata(y_true)
    f1s, kappas = [], []
    for _ in range(N_BOOT):
        idx = stratified_indices(strata, rng)
        f1s.append(macro_f1_on(idx, y_true, pred))
        kappas.append(kappa_on(idx, y_true, pred))
    f1s, kappas = np.array(f1s), np.array(kappas)
    f_lo, f_hi = np.percentile(f1s, [2.5, 97.5])
    k_lo, k_hi = np.percentile(kappas, [2.5, 97.5])
    print("  {:24s} macro-F1 {:.4f}  CI [{:.4f},{:.4f}]   wKappa {:.4f}  CI [{:.4f},{:.4f}]".format(
        name, float(np.mean(f1s)), f_lo, f_hi, float(np.mean(kappas)), k_lo, k_hi))
    return {"system": name, "macro_f1_mean": float(np.mean(f1s)),
            "macro_f1_ci": [float(f_lo), float(f_hi)],
            "kappa_mean": float(np.mean(kappas)), "kappa_ci": [float(k_lo), float(k_hi)]}


def main():
    print("#" * 70)
    print("# DISTRESS STRATIFIED PAIRED BOOTSTRAP ({} resamples)".format(N_BOOT))
    print("#" * 70)

    test_rows = read_jsonl(TEST)
    y_true = get_y_true(test_rows)
    n = len(y_true)
    print("test n = {}  supports = {}".format(
        n, {LEVELS[c]: sum(1 for y in y_true if y == c) for c in IDS}))

    print("\nbuilding predictions...")
    tf_B = tfidf_nominal_predict("B", test_rows)
    db_B_seeds = db_seed_preds("B")
    db_A_seeds = db_seed_preds("A")
    db_B_vote = majority_vote(db_B_seeds, n)
    db_A_vote = majority_vote(db_A_seeds, n)

    print("running zero-shot BART (2 templates)...")
    zs = zeroshot_predict(test_rows)

    def mf(p):
        return f1_score(y_true, p, labels=IDS, average="macro", zero_division=0)
    print("\nfull-test macro-F1 (sanity check vs earlier runs):")
    print("  TF-IDF-nominal-B    : {:.4f}".format(mf(tf_B)))
    print("  DistilBERT-B (vote) : {:.4f}".format(mf(db_B_vote)))
    print("  DistilBERT-A (vote) : {:.4f}".format(mf(db_A_vote)))
    print("  zero-shot BART      : {:.4f}".format(mf(zs)))
    print("  DistilBERT-B per-seed: {}".format([round(mf(p), 4) for p in db_B_seeds]))

    rng = np.random.default_rng(RNG_SEED)
    results = []

    print("\n" + "=" * 70)
    print("MAIN COMPARISON: DistilBERT-B vs TF-IDF-nominal-B")
    print("=" * 70)
    print("\n  (a) vs DistilBERT-B 5-seed MAJORITY-VOTE ensemble:")
    results.append(paired_bootstrap("DistilBERT-B(vote)", db_B_vote, "TF-IDF-nom-B", tf_B, y_true, rng))
    print("\n  (b) vs EACH DistilBERT-B seed (range):")
    for s, seed in enumerate(SEEDS):
        results.append(paired_bootstrap(
            "DistilBERT-B(seed {})".format(seed), db_B_seeds[s], "TF-IDF-nom-B", tf_B, y_true, rng))

    print("\n" + "=" * 70)
    print("SECONDARY COMPARISONS")
    print("=" * 70)
    results.append(paired_bootstrap("DistilBERT-B(vote)", db_B_vote,
                                    "DistilBERT-A(vote)", db_A_vote, y_true, rng))
    results.append(paired_bootstrap("DistilBERT-B(vote)", db_B_vote, "zero-shot-BART", zs, y_true, rng))
    results.append(paired_bootstrap("TF-IDF-nom-B", tf_B, "zero-shot-BART", zs, y_true, rng))

    print("\n" + "=" * 70)
    print("SINGLE-SYSTEM ABSOLUTE CIs (macro-F1 and linear weighted kappa)")
    print("=" * 70)
    singles = []
    singles.append(single_system_ci("DistilBERT-B(vote)", db_B_vote, y_true, rng))
    singles.append(single_system_ci("TF-IDF-nominal-B", tf_B, y_true, rng))
    singles.append(single_system_ci("zero-shot-BART", zs, y_true, rng))

    out = {"n_boot": N_BOOT, "n_test": n, "comparisons": results, "single_systems": singles}
    Path("results/bootstrap").mkdir(parents=True, exist_ok=True)
    with open("results/bootstrap/bootstrap_distress.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nsaved: results/bootstrap/bootstrap_distress.json")
    print("\nNote: P(X>Y) is the share of resamples where X wins, NOT a classical p-value.")


if __name__ == "__main__":
    main()
