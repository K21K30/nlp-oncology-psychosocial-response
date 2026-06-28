"""
merge_pools.py — Merge the full and top-up candidate pools into one file for tiering.

Reads:
  data/gen_v6_low_medium/full/candidates.jsonl    (original run)
  data/gen_v6_low_medium/topup/candidates.jsonl   (top-up run)

Writes:
  data/gen_v6_low_medium/merged/candidates.jsonl  (combined, exact-duplicate texts removed)

The source files are NOT modified. Exact-duplicate texts (same normalized text) are kept once;
the first occurrence wins. Prints a summary so you can confirm the merged corpus size before
running p3_build_tiers.py on it.

Run:  python merge_pools.py
"""

import json
import re
from pathlib import Path
from collections import Counter

EXPERIMENT_DIR = Path("data/gen_v6_low_medium")
FULL_PATH = EXPERIMENT_DIR / "full" / "candidates.jsonl"
TOPUP_PATH = EXPERIMENT_DIR / "topup" / "candidates.jsonl"
MERGED_DIR = EXPERIMENT_DIR / "merged"
MERGED_PATH = MERGED_DIR / "candidates.jsonl"


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def read_jsonl(path: Path) -> list:
    rows = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def main() -> None:
    full_rows = read_jsonl(FULL_PATH)
    topup_rows = read_jsonl(TOPUP_PATH)

    print(f"full pool:  {len(full_rows):5d}  ({FULL_PATH})")
    print(f"topup pool: {len(topup_rows):5d}  ({TOPUP_PATH})")

    if not full_rows and not topup_rows:
        raise SystemExit("Both pools are empty - nothing to merge.")

    seen = set()
    merged = []
    dupes = 0

    # full first (first occurrence wins), then topup
    for source in (full_rows, topup_rows):
        for r in source:
            text = r.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            key = normalize_text(text)
            if key in seen:
                dupes += 1
                continue
            seen.add(key)
            merged.append(r)

    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    with open(MERGED_PATH, "w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    valid = sum(1 for r in merged if r.get("validation_status") != "invalid_judgment")
    statuses = Counter(r.get("validation_status") for r in merged)

    print()
    print("=" * 56)
    print(f"merged total:        {len(merged)}")
    print(f"exact duplicates removed: {dupes}")
    print(f"valid audited (excl. invalid_judgment): {valid}")
    print("status breakdown:")
    for s, c in statuses.most_common():
        print(f"  {str(s):28s} {c:5d}")
    print()
    if valid >= 2000:
        print(f"Corpus >= 2000 ({valid}). Ready for p3.")
    else:
        print(f"Corpus is {valid}; still need ~{2000 - valid} more to reach 2000.")
    print("=" * 56)
    print(f"\nWritten: {MERGED_PATH}")
    print("Next: py p3_build_tiers.py --candidates "
          f"{MERGED_PATH}")


if __name__ == "__main__":
    main()
