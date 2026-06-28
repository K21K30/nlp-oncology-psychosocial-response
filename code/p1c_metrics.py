"""
p1c_metrics.py — Quality metrics over the audited dataset (advisor-recommended).

Reads data/audited_dataset.csv (produced by p1b_audit.py) and computes, WITHOUT calling any
model, the deeper agreement statistics the advisor asked for:

  RESPONSE task (categorical, 7 classes):
    - per-judge agreement with the generation label
    - judge-to-judge (A<->B) agreement
    - confusion: generation label vs each judge's blind top-1 (where they disagree)
    - high-confidence coverage PER class (the filter must not erase a whole class)

  DISTRESS task (ordinal: low<medium<high):
    - per-judge agreement
    - judge-to-judge (A<->B) agreement
    - weighted Cohen's kappa (quadratic) — correct for ordered levels
    - mean absolute distance between generation level and judge level
    - agreement within one level (|gen - judge| <= 1)
    - directional error shares: high->medium (boundary) vs high->low (serious)
    - high-confidence coverage per level

All results are printed and also written to results/metrics_summary.txt.

Run:  python p1c_metrics.py
"""

import csv
from pathlib import Path
from collections import Counter, defaultdict

INPUT_CSV = Path("data/audited_dataset.csv")
OUT_DIR = Path("results")
OUT_TXT = OUT_DIR / "metrics_summary.txt"

RESPONSES = ["anxiety", "sadness", "anger", "hope", "guilt", "denial", "acceptance"]
DISTRESS = ["low", "medium", "high"]
DIST_IDX = {lvl: i for i, lvl in enumerate(DISTRESS)}


# ----------------------------------------------------------------------------- #
# Loading
# ----------------------------------------------------------------------------- #
def load_rows() -> list:
    if not INPUT_CSV.exists():
        raise SystemExit(f"{INPUT_CSV} not found. Run p1b_audit.py first.")
    with open(INPUT_CSV, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_bool(v) -> bool:
    return str(v).strip().lower() == "true"


# ----------------------------------------------------------------------------- #
# Weighted Cohen's kappa (quadratic weights) for ordered distress levels
# ----------------------------------------------------------------------------- #
def quadratic_weighted_kappa(pairs: list) -> float | None:
    """pairs: list of (gen_idx, judge_idx) over {0,1,2}. Returns QWK or None if degenerate."""
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    n = len(pairs)
    if n == 0:
        return None
    k = len(DISTRESS)
    O = [[0] * k for _ in range(k)]
    for a, b in pairs:
        O[a][b] += 1
    row = [sum(O[i]) for i in range(k)]
    col = [sum(O[i][j] for i in range(k)) for j in range(k)]
    W = [[((i - j) ** 2) / ((k - 1) ** 2) for j in range(k)] for i in range(k)]
    E = [[row[i] * col[j] / n for j in range(k)] for i in range(k)]
    num = sum(W[i][j] * O[i][j] for i in range(k) for j in range(k))
    den = sum(W[i][j] * E[i][j] for i in range(k) for j in range(k))
    if den == 0:
        return None
    return 1.0 - num / den


# ----------------------------------------------------------------------------- #
# Main
# ----------------------------------------------------------------------------- #
def main() -> None:
    rows = load_rows()
    n = len(rows)
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 66)
    out("QUALITY METRICS OVER AUDITED DATASET")
    out(f"records: {n}")
    out("=" * 66)

    # ---------------- RESPONSE ----------------
    out("\n--- RESPONSE task (7 classes) ---")
    gpt_ok = sum(1 for r in rows if r.get("gpt_resp_top1") == r.get("response"))
    qwen_ok = sum(1 for r in rows if r.get("qwen_resp_top1") == r.get("response"))
    ab_ok = sum(1 for r in rows
                if r.get("gpt_resp_top1") and r.get("gpt_resp_top1") == r.get("qwen_resp_top1"))
    out(f"gpt agrees with label:   {gpt_ok}/{n} ({gpt_ok/n*100:.1f}%)")
    out(f"qwen agrees with label:  {qwen_ok}/{n} ({qwen_ok/n*100:.1f}%)")
    out(f"judge-to-judge A<->B:    {ab_ok}/{n} ({ab_ok/n*100:.1f}%)")

    out("\nPer-class response agreement and high-confidence coverage:")
    out(f"  {'class':12s} {'n':>4s} {'gpt%':>7s} {'qwen%':>7s} {'highconf%':>10s}")
    for c in RESPONSES:
        cls = [r for r in rows if r.get("response") == c]
        if not cls:
            continue
        g = sum(1 for r in cls if r.get("gpt_resp_top1") == c) / len(cls) * 100
        q = sum(1 for r in cls if r.get("qwen_resp_top1") == c) / len(cls) * 100
        hc = sum(1 for r in cls if as_bool(r.get("resp_highconf"))) / len(cls) * 100
        out(f"  {c:12s} {len(cls):4d} {g:7.1f} {q:7.1f} {hc:10.1f}")

    # where does each judge send the disagreements? (top confusions vs gpt)
    out("\nTop response confusions (generation -> gpt top-1), disagreements only:")
    conf = Counter()
    for r in rows:
        gen, jr = r.get("response"), r.get("gpt_resp_top1")
        if gen and jr and gen != jr:
            conf[(gen, jr)] += 1
    for (gen, jr), cnt in conf.most_common(8):
        out(f"  {gen:12s} -> {jr:12s} : {cnt}")

    # ---------------- DISTRESS ----------------
    out("\n--- DISTRESS task (ordinal low<medium<high) ---")
    gpt_d = sum(1 for r in rows if r.get("gpt_dist_level") == r.get("distress"))
    qwen_d = sum(1 for r in rows if r.get("qwen_dist_level") == r.get("distress"))
    ab_d = sum(1 for r in rows
               if r.get("gpt_dist_level") and r.get("gpt_dist_level") == r.get("qwen_dist_level"))
    out(f"gpt agrees with label:   {gpt_d}/{n} ({gpt_d/n*100:.1f}%)")
    out(f"qwen agrees with label:  {qwen_d}/{n} ({qwen_d/n*100:.1f}%)")
    out(f"judge-to-judge A<->B:    {ab_d}/{n} ({ab_d/n*100:.1f}%)")

    # weighted kappa (gen vs each judge), mean abs distance, within-one-level
    for jname, col in [("gpt", "gpt_dist_level"), ("qwen", "qwen_dist_level")]:
        pairs, dists, within1 = [], [], 0
        for r in rows:
            gi = DIST_IDX.get(r.get("distress"))
            ji = DIST_IDX.get(r.get(col))
            if gi is not None and ji is not None:
                pairs.append((gi, ji))
                dists.append(abs(gi - ji))
                within1 += 1 if abs(gi - ji) <= 1 else 0
        qwk = quadratic_weighted_kappa(pairs)
        mad = sum(dists) / len(dists) if dists else float("nan")
        w1 = within1 / len(dists) * 100 if dists else float("nan")
        qwk_s = f"{qwk:.3f}" if qwk is not None else "n/a"
        out(f"\n{jname}: weighted-kappa={qwk_s}  mean|dist|={mad:.2f}  within-1-level={w1:.1f}%")

    # directional errors for "high" (boundary vs serious) — using gpt
    out("\nDirectional errors on generation=high (gpt):")
    highs = [r for r in rows if r.get("distress") == "high"]
    if highs:
        h2m = sum(1 for r in highs if r.get("gpt_dist_level") == "medium")
        h2l = sum(1 for r in highs if r.get("gpt_dist_level") == "low")
        out(f"  high->medium (boundary): {h2m}/{len(highs)} ({h2m/len(highs)*100:.1f}%)")
        out(f"  high->low (serious):     {h2l}/{len(highs)} ({h2l/len(highs)*100:.1f}%)")

    out("\nPer-level distress agreement and high-confidence coverage:")
    out(f"  {'level':8s} {'n':>4s} {'gpt%':>7s} {'qwen%':>7s} {'highconf%':>10s}")
    for lvl in DISTRESS:
        cls = [r for r in rows if r.get("distress") == lvl]
        if not cls:
            continue
        g = sum(1 for r in cls if r.get("gpt_dist_level") == lvl) / len(cls) * 100
        q = sum(1 for r in cls if r.get("qwen_dist_level") == lvl) / len(cls) * 100
        hc = sum(1 for r in cls if as_bool(r.get("dist_highconf"))) / len(cls) * 100
        out(f"  {lvl:8s} {len(cls):4d} {g:7.1f} {q:7.1f} {hc:10.1f}")

    out("\n" + "=" * 66)
    out("Reading guide:")
    out("- high A<->B but lower label-agreement => judges share heuristics; the")
    out("  high-confidence yield is a VALIDATION YIELD, not proof of high quality.")
    out("- weighted-kappa accounts for ordered levels; within-1-level shows near-misses.")
    out("- high->low errors matter more than high->medium (boundary).")
    out("- watch per-class high-confidence coverage: the filter must not erase a class.")
    out("=" * 66)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved -> {OUT_TXT}")


if __name__ == "__main__":
    main()
