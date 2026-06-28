"""
p4_split.py — Build train/validation/test splits and class weights (advisor-specified).

Policy (from advisor):
  - TEST and VALIDATION come ONLY from strict_intended (cleanest labels).
  - TEST: aim ~10-12 examples per response class, but take at most ~25% of a rare class's
    strict examples; every one of the 7 classes must appear. VALIDATION: similar, from remaining
    strict. The rest of strict goes to train.
  - Remove exact-duplicate texts between test and any train set.
  - Three training ablations share the SAME val and test:
        A = strict (train portion)
        B = strict (train) + silver
        C = strict (train) + silver + consensus_relabelled
  - class weights per training set: w_c = N / (K * n_c), capped at 5.0, computed on final_response.
  - review_only / needs_rejudging are NEVER used for supervised training.

Inputs (from p3 output dir data/gen_v6_low_medium/tiers/):
  dataset_strict.jsonl, dataset_silver.jsonl, dataset_consensus_relabelled.jsonl

Outputs (data/gen_v6_low_medium/splits/):
  test.jsonl, validation.jsonl
  train_A.jsonl, train_B.jsonl, train_C.jsonl
  class_weights_A.json, class_weights_B.json, class_weights_C.json
  split_summary.txt

Run:  python p4_split.py
"""

import sys
import json
import random
import re
from pathlib import Path
from collections import Counter, defaultdict

SEED = 42
TIERS_DIR = Path("data/gen_v6_low_medium/tiers")
OUT_DIR = Path("data/gen_v6_low_medium/splits")

RESPONSES = ["anxiety", "sadness", "anger", "hope", "guilt", "denial", "acceptance"]
DISTRESS_LEVELS = ["low", "medium", "high"]

TEST_PER_CLASS = 11          # target test examples per response class
VAL_PER_CLASS = 7            # target validation examples per response class
MAX_RARE_FRACTION = 0.25     # never take >25% of a rare class's strict pool for test
WEIGHT_CAP = 5.0


def read_jsonl(path: Path) -> list:
    rows = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def write_jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def take_per_class(pool_by_class: dict, target: int, rng: random.Random) -> tuple:
    """Take up to `target` per class (capped at 25% of that class pool). Returns (taken, rest)."""
    taken, rest = [], []
    for cls in RESPONSES:
        items = pool_by_class.get(cls, [])
        rng.shuffle(items)
        cap = max(1, int(len(items) * MAX_RARE_FRACTION)) if len(items) > 0 else 0
        n = min(target, cap) if len(items) <= target / MAX_RARE_FRACTION else target
        n = min(n, len(items))
        taken.extend(items[:n])
        rest.extend(items[n:])
    return taken, rest


def class_weights(rows: list) -> dict:
    """w_c = N / (K * n_c), capped. Based on final_response."""
    counts = Counter(r["final_response"] for r in rows if r.get("final_response") in RESPONSES)
    N = sum(counts.values())
    K = len(RESPONSES)
    weights = {}
    for cls in RESPONSES:
        n_c = counts.get(cls, 0)
        if n_c == 0:
            weights[cls] = WEIGHT_CAP
        else:
            weights[cls] = min(N / (K * n_c), WEIGHT_CAP)
    return weights, dict(counts)


def main() -> None:
    rng = random.Random(SEED)
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    strict = read_jsonl(TIERS_DIR / "dataset_strict.jsonl")
    silver = read_jsonl(TIERS_DIR / "dataset_silver.jsonl")
    consensus = read_jsonl(TIERS_DIR / "dataset_consensus_relabelled.jsonl")

    if not strict:
        raise SystemExit(f"No strict data found in {TIERS_DIR}. Run p3 first.")

    out("=" * 60)
    out("SPLIT BUILD (test/val from strict only)")
    out("=" * 60)
    out(f"strict={len(strict)}  silver={len(silver)}  consensus={len(consensus)}")

    # group strict by final_response
    strict_by_cls = defaultdict(list)
    for r in strict:
        if r.get("final_response") in RESPONSES:
            strict_by_cls[r["final_response"]].append(r)

    out("\nstrict per class:")
    for cls in RESPONSES:
        out(f"  {cls:11s} {len(strict_by_cls[cls])}")

    # carve TEST then VALIDATION from strict
    test_rows, rest_after_test = take_per_class(strict_by_cls, TEST_PER_CLASS, rng)
    rest_by_cls = defaultdict(list)
    for r in rest_after_test:
        rest_by_cls[r["final_response"]].append(r)
    val_rows, train_strict = take_per_class(rest_by_cls, VAL_PER_CLASS, rng)

    # dedup guard: remove any train item whose text matches a test/val text
    locked = {norm(r["text"]) for r in test_rows} | {norm(r["text"]) for r in val_rows}

    def drop_locked(rows):
        return [r for r in rows if norm(r["text"]) not in locked]

    train_strict = drop_locked(train_strict)
    silver_clean = drop_locked(silver)
    consensus_clean = drop_locked(consensus)

    # build the three training sets
    train_A = train_strict
    train_B = train_strict + silver_clean
    train_C = train_strict + silver_clean + consensus_clean

    # tag split field
    for r in test_rows:
        r["split"] = "test"
    for r in val_rows:
        r["split"] = "validation"

    write_jsonl(OUT_DIR / "test.jsonl", test_rows)
    write_jsonl(OUT_DIR / "validation.jsonl", val_rows)
    write_jsonl(OUT_DIR / "train_A.jsonl", train_A)
    write_jsonl(OUT_DIR / "train_B.jsonl", train_B)
    write_jsonl(OUT_DIR / "train_C.jsonl", train_C)

    out(f"\nTEST: {len(test_rows)}   VALIDATION: {len(val_rows)}")
    out("test per class (support):")
    tc = Counter(r["final_response"] for r in test_rows)
    for cls in RESPONSES:
        out(f"  {cls:11s} {tc.get(cls, 0)}")

    out(f"\nTrain sizes:  A={len(train_A)}  B={len(train_B)}  C={len(train_C)}")

    # class weights per training set
    for name, rows in [("A", train_A), ("B", train_B), ("C", train_C)]:
        weights, counts = class_weights(rows)
        with open(OUT_DIR / f"class_weights_{name}.json", "w", encoding="utf-8") as f:
            json.dump(weights, f, indent=2)
        out(f"\ntrain_{name} response counts: " +
            ", ".join(f"{c}={counts.get(c,0)}" for c in RESPONSES))
        out(f"train_{name} class weights:  " +
            ", ".join(f"{c}={weights[c]:.2f}" for c in RESPONSES))

    out("\n" + "=" * 60)
    out("Notes:")
    out("- test/val are strict-only; report per-class F1 + support (small classes unstable).")
    out("- minority macro-F1 = mean(F1 anger, denial, acceptance) as extra diagnostic.")
    out("- next: human-check the whole test.jsonl (text+rubric only), then train A/B/C.")
    out("=" * 60)
    write_jsonl  # noqa: silence
    (OUT_DIR / "split_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    out(f"Files in: {OUT_DIR}")


if __name__ == "__main__":
    main()
