"""
p8_diagnostics.py - Post-hoc diagnostics for the response classifier (advisor-requested).

NO training. Reads saved per-seed test_predictions.json for A/B/C, the human-annotated test file,
and the tier/train files, then produces:

ANGER diagnostics (advisor 2.A-2.D):
  A. anger<->sadness confusion per experiment (mean over seeds): strict anger->pred sadness,
     strict sadness->pred anger.
  B. human-confirmed anger vs disputed anger: how the model labels the 5 strict-anger items the
     human also called anger, vs the strict-anger items the human called something else.
  C. model-vs-human: C predictions against HUMAN labels on all 67 (macro-F1, anger P/R/F1,
     confusion). Sensitivity analysis (single annotator).
  D. consensus anger source: counts of final_response=anger in A/B/C train; consensus records
     relabelled INTO anger; intended anger relabelled OUT to sadness; near-duplicate check between
     anger train texts and test texts.

ANXIETY over-prediction (advisor 3): per experiment, predicted/support ratio, precision, recall,
F1, and which true classes get misread as anxiety.

USAGE (from project root):
    py p8_diagnostics.py
"""

import json
import glob
import statistics
from collections import Counter, defaultdict

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix


# =============================================================================
# Paths / constants
# =============================================================================

LABELS = ["anxiety", "sadness", "anger", "hope", "guilt", "denial", "acceptance"]
LAB2ID = {l: i for i, l in enumerate(LABELS)}

MODEL_DIRS = {
    "A": "models/distilbert_response_A",
    "B": "models/distilbert_response_B",
    "C": "models/distilbert_response_C",
}
SEEDS = [13, 42, 73, 101, 2026]

HUMAN_TEST = "data/gen_v6_low_medium/splits/test_with_human_annotations.jsonl"
TRAIN_FILES = {
    "A": "data/gen_v6_low_medium/splits/train_A.jsonl",
    "B": "data/gen_v6_low_medium/splits/train_B.jsonl",
    "C": "data/gen_v6_low_medium/splits/train_C.jsonl",
}
CONSENSUS_FILE = "data/gen_v6_low_medium/tiers/dataset_consensus_relabelled.jsonl"
TEST_FILE = "data/gen_v6_low_medium/splits/test.jsonl"


# =============================================================================
# IO helpers
# =============================================================================

def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_seed_predictions(model_dir):
    """Return list of (y_true, y_pred) over seeds for one experiment."""
    out = []
    for seed in SEEDS:
        path = "{}/seed_{}/test_predictions.json".format(model_dir, seed)
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        out.append((d["y_true"], d["y_pred"]))
    return out


# =============================================================================
# Diagnostic A: anger<->sadness confusion per experiment (mean over seeds)
# =============================================================================

def anger_sadness_confusion():
    print("\n" + "=" * 70)
    print("DIAGNOSTIC A: anger<->sadness confusion per experiment (mean over 5 seeds)")
    print("=" * 70)
    anger_id, sad_id = LAB2ID["anger"], LAB2ID["sadness"]

    for tag, mdir in MODEL_DIRS.items():
        a2s_list, s2a_list = [], []
        anger_f1_list = []
        for y_true, y_pred in load_seed_predictions(mdir):
            yt, yp = np.array(y_true), np.array(y_pred)
            # strict anger -> predicted sadness
            a2s = int(np.sum((yt == anger_id) & (yp == sad_id)))
            # strict sadness -> predicted anger
            s2a = int(np.sum((yt == sad_id) & (yp == anger_id)))
            a2s_list.append(a2s)
            s2a_list.append(s2a)
            anger_f1_list.append(
                f1_score(yt, yp, labels=[anger_id], average="macro", zero_division=0)
            )
        print("\n  {} (train counts shown in diag D):".format(tag))
        print("    strict anger -> pred sadness : mean {:.2f}  per-seed {}".format(
            statistics.mean(a2s_list), a2s_list))
        print("    strict sadness -> pred anger : mean {:.2f}  per-seed {}".format(
            statistics.mean(s2a_list), s2a_list))
        print("    anger F1 (mean)              : {:.3f}".format(statistics.mean(anger_f1_list)))
    print("\n  Reading: if C reduces BOTH directions vs A, the anger/sadness boundary genuinely")
    print("  improved. If errors are already ~0 in A, there is little boundary error to fix.")


# =============================================================================
# Diagnostics B & C: human-confirmed anger, and model-vs-human (uses C, seed-averaged votes)
# =============================================================================

def majority_vote_pred(model_dir, n_items):
    """Per test item, majority-vote the predicted class across the 5 seeds (ties -> lowest id)."""
    seed_preds = [yp for (_, yp) in load_seed_predictions(model_dir)]
    voted = []
    for i in range(n_items):
        votes = Counter(seed_preds[s][i] for s in range(len(seed_preds)))
        best = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        voted.append(best)
    return voted


def human_confirmed_and_model_vs_human():
    rows = read_jsonl(HUMAN_TEST)
    n = len(rows)

    strict = [LAB2ID[r["final_response"]] for r in rows]
    human = [LAB2ID[r["human_response"]] for r in rows]
    human_amb = [bool(r.get("human_response_ambiguous", False)) for r in rows]

    # C predictions: majority vote over seeds (stable single prediction per item)
    c_pred = majority_vote_pred(MODEL_DIRS["C"], n)

    anger_id, sad_id = LAB2ID["anger"], LAB2ID["sadness"]

    # ---- Diagnostic B ----
    print("\n" + "=" * 70)
    print("DIAGNOSTIC B: human-confirmed anger vs disputed anger (model = C majority vote)")
    print("=" * 70)
    confirmed_idx = [i for i in range(n) if strict[i] == anger_id and human[i] == anger_id]
    disputed_idx = [i for i in range(n) if strict[i] == anger_id and human[i] != anger_id]

    print("\n  strict-anger items: {} total".format(sum(1 for i in range(n) if strict[i] == anger_id)))
    print("  human-confirmed anger (strict=anger AND human=anger): {}".format(len(confirmed_idx)))
    print("  disputed anger (strict=anger, human=other):           {}".format(len(disputed_idx)))

    def model_breakdown(idxs):
        c = Counter(LABELS[c_pred[i]] for i in idxs)
        return dict(c)

    if confirmed_idx:
        n_anger = sum(1 for i in confirmed_idx if c_pred[i] == anger_id)
        print("\n  On human-CONFIRMED anger ({} items): model C calls {} of them anger.".format(
            len(confirmed_idx), n_anger))
        print("    model label breakdown: {}".format(model_breakdown(confirmed_idx)))
    if disputed_idx:
        n_anger = sum(1 for i in disputed_idx if c_pred[i] == anger_id)
        print("\n  On DISPUTED anger ({} items, human said other): model C calls {} of them anger.".format(
            len(disputed_idx), n_anger))
        print("    model label breakdown: {}".format(model_breakdown(disputed_idx)))
        print("    human labels on these disputed items: {}".format(
            dict(Counter(LABELS[human[i]] for i in disputed_idx))))
    print("\n  Reading: if C calls all disputed items anger too, the high strict-test F1 reflects")
    print("  agreement with the synthetic annotation policy, not necessarily with the human.")

    # ---- Diagnostic C ----
    print("\n" + "=" * 70)
    print("DIAGNOSTIC C: model (C) vs HUMAN labels on all 67 (sensitivity, single annotator)")
    print("=" * 70)
    label_ids = list(range(len(LABELS)))
    macro = f1_score(human, c_pred, labels=label_ids, average="macro", zero_division=0)
    p, r, f, s = precision_recall_fscore_support(
        human, c_pred, labels=[anger_id], average=None, zero_division=0)
    print("\n  human-reference macro-F1 (C vs human): {:.4f}".format(macro))
    print("  anger vs human: precision {:.3f}  recall {:.3f}  F1 {:.3f}  human-support {}".format(
        p[0], r[0], f[0], s[0]))

    # also strict-vs-human macro for context
    macro_strict_human = f1_score(human, strict, labels=label_ids, average="macro", zero_division=0)
    print("  (context) strict-vs-human macro-F1:    {:.4f}".format(macro_strict_human))
    print("  (context) C-vs-strict macro-F1 (voted):{:.4f}".format(
        f1_score(strict, c_pred, labels=label_ids, average="macro", zero_division=0)))

    print("\n  model(C) vs human confusion (rows=human, cols=C pred):")
    cm = confusion_matrix(human, c_pred, labels=label_ids)
    header = "    {:11s}".format("") + "".join("{:>7s}".format(l[:6]) for l in LABELS)
    print(header)
    for i, lab in enumerate(LABELS):
        print("    {:11s}".format(lab) + "".join("{:7d}".format(cm[i][j]) for j in range(len(LABELS))))

    # ---- ambiguous vs unambiguous (bonus, advisor step 4 preview) ----
    print("\n" + "=" * 70)
    print("BONUS: C accuracy on human-unambiguous vs human-ambiguous response items")
    print("=" * 70)
    unamb_idx = [i for i in range(n) if not human_amb[i]]
    amb_idx = [i for i in range(n) if human_amb[i]]

    def acc_vs_strict(idxs):
        if not idxs:
            return float("nan"), 0
        correct = sum(1 for i in idxs if c_pred[i] == strict[i])
        return correct / len(idxs), len(idxs)

    ua, un = acc_vs_strict(unamb_idx)
    aa, an = acc_vs_strict(amb_idx)
    print("\n  C-vs-strict accuracy on human-UNAMBIGUOUS items: {:.3f}  (n={})".format(ua, un))
    print("  C-vs-strict accuracy on human-AMBIGUOUS items:   {:.3f}  (n={})".format(aa, an))
    print("  (expect higher accuracy on unambiguous items if disagreements track item difficulty)")


# =============================================================================
# Diagnostic D: consensus anger source + near-duplicate train<->test
# =============================================================================

def normalize_text(t):
    return " ".join(t.lower().split())


def consensus_anger_source():
    print("\n" + "=" * 70)
    print("DIAGNOSTIC D: anger train counts, consensus transitions, near-duplicate check")
    print("=" * 70)

    # train counts of final_response per experiment
    print("\n  final_response=anger counts per train set:")
    train_texts = {}
    for tag, path in TRAIN_FILES.items():
        rows = read_jsonl(path)
        c = Counter(r["final_response"] for r in rows)
        train_texts[tag] = [(normalize_text(r["text"]), r["final_response"]) for r in rows]
        print("    {}: total {}  anger {}  sadness {}  (full dist: {})".format(
            tag, len(rows), c.get("anger", 0), c.get("sadness", 0), dict(c)))

    # consensus transitions (intended -> final) on the consensus tier
    cons = read_jsonl(CONSENSUS_FILE)
    into_anger = Counter()   # intended X -> final anger
    out_anger = Counter()    # intended anger -> final X
    n_relabelled = 0
    for r in cons:
        intended = r.get("intended_response")
        final = r.get("final_response")
        if intended is None or final is None:
            continue
        if intended != final:
            n_relabelled += 1
        if final == "anger" and intended != "anger":
            into_anger[intended] += 1
        if intended == "anger" and final != "anger":
            out_anger[final] += 1

    print("\n  consensus tier: {} records, {} relabelled (intended != final).".format(
        len(cons), n_relabelled))
    print("  relabelled INTO anger (intended X -> final anger): total {}  by source {}".format(
        sum(into_anger.values()), dict(into_anger)))
    print("  intended anger relabelled OUT (intended anger -> final X): total {}  by dest {}".format(
        sum(out_anger.values()), dict(out_anger)))
    print("    -> 'intended anger -> final sadness' specifically: {}".format(out_anger.get("sadness", 0)))

    # near-duplicate check: anger train texts vs ALL test texts (exact normalized match)
    test_rows = read_jsonl(TEST_FILE)
    test_norm = {normalize_text(r["text"]): r["final_response"] for r in test_rows}
    test_anger_norm = {t for t, lab in test_norm.items() if lab == "anger"}

    print("\n  near-duplicate check (exact normalized text match), per experiment:")
    for tag in ["A", "B", "C"]:
        anger_train = [t for (t, lab) in train_texts[tag] if lab == "anger"]
        # overlap of anger-train with ANY test item
        dup_any = sum(1 for t in anger_train if t in test_norm)
        # overlap of anger-train with anger-test specifically
        dup_anger = sum(1 for t in anger_train if t in test_anger_norm)
        print("    {}: anger-train items {}  exact-match to any test {}  to anger-test {}".format(
            tag, len(anger_train), dup_any, dup_anger))
    print("  (exact normalized match is a strict-leakage check; 0 overlaps = no verbatim leakage.")
    print("   Near-but-not-identical paraphrase leakage is not captured by exact match.)")


# =============================================================================
# Anxiety over-prediction (advisor section 3)
# =============================================================================

def anxiety_over_prediction():
    print("\n" + "=" * 70)
    print("ANXIETY over-prediction per experiment (mean over 5 seeds)")
    print("=" * 70)
    anx_id = LAB2ID["anxiety"]

    for tag, mdir in MODEL_DIRS.items():
        pred_counts, precisions, recalls, f1s = [], [], [], []
        misread = Counter()  # true class -> predicted anxiety (false positives by source)
        support = None
        for y_true, y_pred in load_seed_predictions(mdir):
            yt, yp = np.array(y_true), np.array(y_pred)
            support = int(np.sum(yt == anx_id))
            pred_counts.append(int(np.sum(yp == anx_id)))
            p, r, f, _ = precision_recall_fscore_support(
                yt, yp, labels=[anx_id], average=None, zero_division=0)
            precisions.append(p[0]); recalls.append(r[0]); f1s.append(f[0])
            for t_id in range(len(LABELS)):
                if t_id == anx_id:
                    continue
                misread[LABELS[t_id]] += int(np.sum((yt == t_id) & (yp == anx_id)))
        mean_pred = statistics.mean(pred_counts)
        print("\n  {}: support {}  mean predicted {:.1f}  ratio {:.2f}".format(
            tag, support, mean_pred, mean_pred / support))
        print("     precision {:.3f}  recall {:.3f}  F1 {:.3f}  (means over seeds)".format(
            statistics.mean(precisions), statistics.mean(recalls), statistics.mean(f1s)))
        print("     per-seed predicted counts: {}".format(pred_counts))
        # total false-positives into anxiety by true source (summed over seeds)
        fp = {k: v for k, v in misread.items() if v > 0}
        print("     false 'anxiety' by true source (summed over 5 seeds): {}".format(fp))
    print("\n  Reading: ratio > 1 = over-prediction. recall ~1.0 with lower precision means the")
    print("  model catches all anxiety but also tags other classes as anxiety. (Confirmed present")
    print("  in C-unweighted too, so the cause is the consensus data, not the class weights.)")


# =============================================================================
# Main
# =============================================================================

def main():
    print("#" * 70)
    print("# RESPONSE DIAGNOSTICS (advisor-requested) - no training, reads saved predictions")
    print("#" * 70)
    anger_sadness_confusion()
    human_confirmed_and_model_vs_human()
    consensus_anger_source()
    anxiety_over_prediction()
    print("\n" + "#" * 70)
    print("# DONE")
    print("#" * 70)


if __name__ == "__main__":
    main()
