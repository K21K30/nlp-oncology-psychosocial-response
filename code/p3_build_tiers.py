"""
p3_build_tiers.py — Post-process audited candidates into quality tiers.

Reads the audited candidate pool (candidates.jsonl) produced by p2_validated_gen.py and
assigns each message to ONE quality tier, following the advisor's specification. It NEVER
overwrites intended_* labels; it adds final_* (the usable label for that tier) and metadata.

Tiers (mutually exclusive, checked in this order):
  needs_rejudging      : validation_status == invalid_judgment (not counted toward the corpus)
  strict_intended      : both judges == intended (response AND distress),
                         all four confidences >= 4, no ambiguity      -> final = intended
  silver_intended      : both judges == intended (response AND distress),
                         all four confidences >= 3, no ambiguity,
                         but NOT strict (at least one confidence == 3) -> final = intended
  consensus_relabelled : judges agree WITH EACH OTHER on response AND distress,
                         all four confidences >= 3, no ambiguity,
                         and at least one consensus label != intended   -> final = judge label
  review_only          : everything else (disagreement, ambiguity, low confidence) -> final = None

Outputs (in data/<experiment>/tiers/):
  dataset_all.jsonl                 (every valid audited message, with tier + final_*)
  dataset_strict.jsonl
  dataset_silver.jsonl
  dataset_consensus_relabelled.jsonl
  dataset_review_only.jsonl
  dataset_model_ready.jsonl         (strict + silver + consensus_relabelled; final_* not null)
  needs_rejudging.jsonl
  tier_summary.txt

The source candidates.jsonl / accepted.jsonl are NOT modified.

Run:  python p3_build_tiers.py
      python p3_build_tiers.py --candidates path/to/candidates.jsonl
"""

import sys
import json
from pathlib import Path
from collections import Counter

MIN_STRICT_CONF = 4
MIN_SILVER_CONF = 3

DEFAULT_CANDIDATES = Path("data/gen_v6_low_medium/full/candidates.jsonl")
RESPONSES = ["anxiety", "sadness", "anger", "hope", "guilt", "denial", "acceptance"]
DISTRESS_LEVELS = ["low", "medium", "high"]


# ----------------------------------------------------------------------------- #
# Tier assignment
# ----------------------------------------------------------------------------- #
def confs(j: dict) -> list:
    """The four confidence values from one judge dict (response + distress)."""
    return [j.get("response_confidence"), j.get("distress_confidence")]


def all_conf_at_least(ja: dict, jb: dict, threshold: int) -> bool:
    vals = confs(ja) + confs(jb)
    return all(isinstance(v, int) and v >= threshold for v in vals)


def no_ambiguity(ja: dict, jb: dict) -> bool:
    return (ja.get("response_ambiguous") is False and jb.get("response_ambiguous") is False
            and ja.get("distress_ambiguous") is False and jb.get("distress_ambiguous") is False)


def both_match_intended(ja, jb, intended_r, intended_d) -> bool:
    return (ja.get("response") == intended_r and jb.get("response") == intended_r
            and ja.get("distress") == intended_d and jb.get("distress") == intended_d)


def judges_agree(ja, jb) -> bool:
    return (ja.get("response") is not None and ja.get("response") == jb.get("response")
            and ja.get("distress") is not None and ja.get("distress") == jb.get("distress"))


def assign_tier(record: dict) -> tuple:
    """Return (tier_name, final_response, final_distress)."""
    if record.get("validation_status") == "invalid_judgment":
        return "needs_rejudging", None, None

    ja = record.get("judge_a") or {}
    jb = record.get("judge_b") or {}
    ir = record.get("intended_response")
    idd = record.get("intended_distress")

    # both judges must have produced usable labels for any model-ready tier
    usable = (ja.get("response") in RESPONSES and jb.get("response") in RESPONSES
              and ja.get("distress") in DISTRESS_LEVELS and jb.get("distress") in DISTRESS_LEVELS)
    if not usable:
        return "review_only", None, None

    if not no_ambiguity(ja, jb):
        return "review_only", None, None

    # strict: both == intended, all conf >= 4
    if both_match_intended(ja, jb, ir, idd) and all_conf_at_least(ja, jb, MIN_STRICT_CONF):
        return "strict_intended", ir, idd

    # silver: both == intended, all conf >= 3 (but not strict)
    if both_match_intended(ja, jb, ir, idd) and all_conf_at_least(ja, jb, MIN_SILVER_CONF):
        return "silver_intended", ir, idd

    # consensus relabelled: judges agree with each other (!= intended), all conf >= 3
    if judges_agree(ja, jb) and all_conf_at_least(ja, jb, MIN_SILVER_CONF):
        # at least one consensus label differs from intended
        if ja.get("response") != ir or ja.get("distress") != idd:
            return "consensus_relabelled", ja.get("response"), ja.get("distress")

    return "review_only", None, None


TIER_META = {
    "strict_intended":      ("intended_confirmed_by_two_judges", True, True, True),
    "silver_intended":      ("intended_confirmed_by_two_judges", True, True, False),
    "consensus_relabelled": ("dual_judge_consensus", True, True, False),
    "review_only":          (None, False, False, False),
    "needs_rejudging":      (None, False, False, False),
}


# ----------------------------------------------------------------------------- #
# Main
# ----------------------------------------------------------------------------- #
def main() -> None:
    cand_path = DEFAULT_CANDIDATES
    if "--candidates" in sys.argv:
        cand_path = Path(sys.argv[sys.argv.index("--candidates") + 1])
    if not cand_path.exists():
        raise SystemExit(f"{cand_path} not found.")

    out_dir = cand_path.parent.parent / "tiers"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(cand_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    tier_counts = Counter()
    buckets = {t: [] for t in ["strict_intended", "silver_intended",
                               "consensus_relabelled", "review_only", "needs_rejudging"]}
    all_valid = []
    model_ready = []

    for r in rows:
        tier, fr, fd = assign_tier(r)
        label_source, model_rdy, train_ok, test_ok = TIER_META[tier]
        r["quality_tier"] = tier
        r["final_response"] = fr
        r["final_distress"] = fd
        r["label_source"] = label_source
        r["model_ready"] = model_rdy
        r["train_eligible"] = train_ok
        r["test_eligible"] = test_ok

        tier_counts[tier] += 1
        buckets[tier].append(r)
        if tier != "needs_rejudging":
            all_valid.append(r)
        if model_rdy:
            model_ready.append(r)

    # write files
    def dump(name, records):
        p = out_dir / name
        with open(p, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return len(records)

    dump("dataset_all.jsonl", all_valid)
    dump("dataset_strict.jsonl", buckets["strict_intended"])
    dump("dataset_silver.jsonl", buckets["silver_intended"])
    dump("dataset_consensus_relabelled.jsonl", buckets["consensus_relabelled"])
    dump("dataset_review_only.jsonl", buckets["review_only"])
    dump("dataset_model_ready.jsonl", model_ready)
    dump("needs_rejudging.jsonl", buckets["needs_rejudging"])

    # summary
    lines = []
    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 60)
    out("TIER ASSIGNMENT SUMMARY")
    out("=" * 60)
    out(f"Source: {cand_path}")
    out(f"Total candidates read:        {len(rows)}")
    out("")
    for t in ["strict_intended", "silver_intended", "consensus_relabelled",
              "review_only", "needs_rejudging"]:
        out(f"  {t:24s} {tier_counts[t]:5d}")
    out("")
    out(f"Valid audited corpus (excl. needs_rejudging): {len(all_valid)}")
    out(f"Model-ready (strict+silver+consensus):        {len(model_ready)}")
    out(f"  - of which STRICT (test-eligible):          {tier_counts['strict_intended']}")
    out("")
    # distance to 2000
    corpus = len(all_valid)
    if corpus >= 2000:
        out(f"Corpus >= 2000 already ({corpus}). No more generation needed for the count.")
    else:
        out(f"Corpus is {corpus}; need ~{2000 - corpus} more audited candidates to reach 2000.")
    out("")
    out("final_* label distribution in model-ready set:")
    fr_counter = Counter((r["final_response"], r["final_distress"]) for r in model_ready)
    for (fr, fd), c in sorted(fr_counter.items(), key=lambda x: (-x[1])):
        out(f"  {str(fr):10s} / {str(fd):6s} : {c}")
    out("=" * 60)
    out(f"Files written to: {out_dir}")

    (out_dir / "tier_summary.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
