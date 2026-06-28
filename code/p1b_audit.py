"""
p1b_audit.py - Blind independent dual-judge audit (two tasks, separate calls).

For each generated message, TWO judges from two different sources each evaluate TWO tasks,
all BLIND (the judge never sees the generation label) and in SEPARATE calls with isolated
context, so one judgment cannot anchor the other:

  Judges:  A = OpenAI gpt-4o-mini (cloud)   |   B = qwen2.5:32b via Ollama (local)
  Tasks:   (1) dominant psychosocial response (top-1 of 7)   (2) expressed distress intensity

  => 2 judges x 2 task-specific calls = 4 calls per message.

Each judge independently returns, for the RESPONSE task:
  {"dominant_label","secondary_labels","confidence"(1-5),"ambiguous"}
and for the DISTRESS task:
  {"distress_level","distress_confidence"(1-5),"distress_ambiguous"}

We then compare each judge's blind top-1 choice against the generation label, and compute,
SEPARATELY for each task (never as a single joint filter):
  - per-judge agreement (judge's top-1 == generation label)
  - per-class agreement (which classes are systematically harder)
  - joint agreement (both judges agree with the label) - reported as EXTRA stat only
  - a high-confidence consensus subset per task, with the strict rule:
        both judges' top-1 == generation label, both confidence >= 4, neither ambiguous.

Terminology: this is a HIGH-CONFIDENCE / dual-judge consensus subset, NOT a "gold" set
(gold implies reliable human annotation).

Raw replies, parsed JSON, and retry errors are all logged to the output records.
Temperature = 0 for reproducibility. Disagreements are kept, not dropped.

Input:  data/raw_dataset.jsonl   (from p1.py)
Output: data/audited_dataset.jsonl  +  data/audited_dataset.csv

Pilot mode: `python p1b_audit.py --pilot` audits a small stratified sample (a few of each
response class and distress level) to validate JSON/definitions/checkpoint before the full run.

Run:  python p1b_audit.py        (full)
      python p1b_audit.py --pilot
Requires: OPENAI_API_KEY in .env, Ollama running with qwen2.5:32b.
"""

import os
import sys
import csv
import json
import time
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI
import ollama


# ----------------------------------------------------------------------------- #
# Configuration
# ----------------------------------------------------------------------------- #
GPT_MODEL = "gpt-4o-mini"
QWEN_MODEL = "qwen2.5:32b"
TEMPERATURE = 0.0                  # reproducibility, not diversity
SLEEP_BETWEEN_CALLS = 0.15
MAX_RETRIES = 3
PILOT_PER_CELL = 2                 # pilot: how many of each (response x distress) cell

INPUT_PATH = Path("data/raw_dataset.jsonl")
JSONL_PATH = Path("data/audited_dataset.jsonl")
CSV_PATH = Path("data/audited_dataset.csv")

# Same taxonomy + definitions as generation (p1.py). Shown to judges for the BLIND choice.
RESPONSE_CLASSES = {
    "anxiety": "fear, worry, or dread about what is happening or may happen.",
    "sadness": "sorrow, grief, loss, or low mood (an expressed emotional state, not a diagnosis).",
    "anger": "frustration, resentment, or rage about the situation.",
    "hope": ("the speaker anticipates, wishes for, or actively strives toward a better future "
             "outcome (recovery, improvement, successful treatment, overcoming the illness); "
             "future-oriented optimism or fighting language is central."),
    "guilt": "self-blame or feeling responsible for something bad.",
    "denial": "a coping reaction: minimizing, rejecting, or not engaging with the reality.",
    "acceptance": ("the speaker acknowledges the current reality and is emotionally coming to "
                   "terms with it; the dominant tone is calm, reflective, grounded, or quietly "
                   "resigned. Acceptance does not depend on expecting improvement; the speaker "
                   "may still pursue treatment, but 'defeating' the illness or anticipating a "
                   "better outcome is NOT the central response."),
}
RESPONSES = list(RESPONSE_CLASSES.keys())

PRIORITY_RULE = (
    "Priority when classes are close:\n"
    "- improvement, winning, fighting, belief in a positive outcome -> hope.\n"
    "- acknowledging reality and calmly adjusting to it -> acceptance.\n"
    "- loss, unfairness, or pain remains dominant -> sadness, even if a 'keep going' phrase "
    "is present."
)

DISTRESS_RUBRIC = (
    "Judge ONLY the distress explicitly expressed in the text. Do NOT infer distress from the "
    "seriousness of the diagnosis, prognosis, or medical situation.\n"
    "- low: little or mild expressed distress; the speaker seems mostly regulated, "
    "matter-of-fact, calm, or only slightly worried; no strong emotional or functional "
    "disruption is expressed.\n"
    "- medium: clear and sustained worry, sadness, fear, frustration, or emotional strain, but "
    "the speaker remains relatively coherent and able to cope; distress is significant but not "
    "portrayed as overwhelming.\n"
    "- high: distress is portrayed as overwhelming, acute, uncontrollable, or seriously "
    "disruptive (panic, desperation, inability to cope or function, extreme hopelessness, "
    "emotional collapse, or crisis-like language).\n"
    "Note: a serious diagnosis described calmly is NOT high distress; a single strong word "
    "without further intensification is not automatically high; high requires strong "
    "linguistic evidence, not just a negative topic."
)
DISTRESS_LEVELS = ["low", "medium", "high"]


# ----------------------------------------------------------------------------- #
# Prompts (blind - generation label NEVER shown)
# ----------------------------------------------------------------------------- #
def response_prompt(text: str) -> str:
    defs = "\n".join(f"- {k}: {v}" for k, v in RESPONSE_CLASSES.items())
    return f"""You are an expert annotator. Read the oncology-related message and independently
select the SINGLE most salient psychosocial response from the label set. Choose the one most
central to the meaning of the message, whether it is emotional or coping-related. Do not assume
any predefined answer. If no class is clearly dominant, mark it ambiguous.

Label set:
{defs}

{PRIORITY_RULE}

Message:
\"\"\"{text}\"\"\"

Answer ONLY with a JSON object, no other text:
{{"dominant_label": "<one label>", "secondary_labels": ["<label>", ...], "confidence": <1-5>, "ambiguous": <true/false>}}"""


def distress_prompt(text: str) -> str:
    return f"""You are an expert annotator. Read the oncology-related message and independently
rate the expressed distress intensity, using ONLY this rubric:
{DISTRESS_RUBRIC}

Message:
\"\"\"{text}\"\"\"

Answer ONLY with a JSON object, no other text:
{{"distress_level": "low|medium|high", "distress_confidence": <1-5>, "distress_ambiguous": <true/false>}}"""


# ----------------------------------------------------------------------------- #
# JSON extraction
# ----------------------------------------------------------------------------- #
def extract_json(raw: str) -> dict | None:
    if not raw:
        return None
    s = raw.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:]
    a, b = s.find("{"), s.rfind("}")
    if a == -1 or b == -1 or b < a:
        return None
    try:
        return json.loads(s[a:b + 1])
    except json.JSONDecodeError:
        return None


def norm_label(v) -> str | None:
    if not isinstance(v, str):
        return None
    v = v.strip().lower()
    return v if v in RESPONSES else None


def norm_level(v) -> str | None:
    if not isinstance(v, str):
        return None
    v = v.strip().lower()
    return v if v in DISTRESS_LEVELS else None


def to_int_conf(v) -> int | None:
    try:
        i = int(v)
        return i if 1 <= i <= 5 else None
    except (TypeError, ValueError):
        return None


# ----------------------------------------------------------------------------- #
# Model callers (return raw text)
# ----------------------------------------------------------------------------- #
def call_gpt(client, prompt: str) -> tuple:
    """Return (raw_text, error_str)."""
    last_err = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
            )
            return r.choices[0].message.content or "", ""
        except Exception as exc:                       # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"
            tqdm.write(f"  [GPT retry {attempt}/{MAX_RETRIES}] {last_err}")
            time.sleep(SLEEP_BETWEEN_CALLS * attempt * 3)
    return "", last_err


def call_qwen(prompt: str) -> tuple:
    """Return (raw_text, error_str). Fresh context each call (no history)."""
    last_err = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = ollama.generate(model=QWEN_MODEL, prompt=prompt,
                                 options={"temperature": TEMPERATURE})
            return r.get("response", "") or "", ""
        except Exception as exc:                       # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"
            tqdm.write(f"  [QWEN retry {attempt}/{MAX_RETRIES}] {last_err}")
            time.sleep(1.0 * attempt)
    return "", last_err


# ----------------------------------------------------------------------------- #
# One judge x one example -> verdict dict for both tasks
# ----------------------------------------------------------------------------- #
def judge_example(name: str, caller, text: str) -> dict:
    """Run the two task-specific calls (separately) for one judge. Returns a flat dict."""
    out = {}

    # --- task 1: response (blind top-1) ---
    raw_r, err_r = caller(response_prompt(text))
    obj_r = extract_json(raw_r)
    out[f"{name}_resp_raw"] = raw_r[:500]
    out[f"{name}_resp_err"] = err_r
    out[f"{name}_resp_top1"] = norm_label((obj_r or {}).get("dominant_label"))
    out[f"{name}_resp_conf"] = to_int_conf((obj_r or {}).get("confidence"))
    out[f"{name}_resp_ambiguous"] = bool((obj_r or {}).get("ambiguous")) if obj_r else None

    time.sleep(SLEEP_BETWEEN_CALLS)

    # --- task 2: distress (blind, separate call) ---
    raw_d, err_d = caller(distress_prompt(text))
    obj_d = extract_json(raw_d)
    out[f"{name}_dist_raw"] = raw_d[:500]
    out[f"{name}_dist_err"] = err_d
    out[f"{name}_dist_level"] = norm_level((obj_d or {}).get("distress_level"))
    out[f"{name}_dist_conf"] = to_int_conf((obj_d or {}).get("distress_confidence"))
    out[f"{name}_dist_ambiguous"] = bool((obj_d or {}).get("distress_ambiguous")) if obj_d else None

    return out


# ----------------------------------------------------------------------------- #
# Checkpoint helpers
# ----------------------------------------------------------------------------- #
def load_done_ids() -> set:
    done = set()
    if JSONL_PATH.exists():
        with open(JSONL_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done.add(json.loads(line)["id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
    return done


def append_record(record: dict) -> None:
    with open(JSONL_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def rebuild_csv() -> list:
    records = []
    with open(JSONL_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    records.sort(key=lambda r: r["id"])
    if records:
        # union of keys (records may differ slightly), stable order from first record
        fieldnames = list(records[0].keys())
        for r in records:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
        with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(records)
    return records


# ----------------------------------------------------------------------------- #
# Stratified pilot selection
# ----------------------------------------------------------------------------- #
def pick_pilot(records: list) -> list:
    """A few examples per (response x distress) cell, to validate end-to-end."""
    seen = {}
    chosen = []
    for r in records:
        cell = (r.get("response"), r.get("distress"))
        if seen.get(cell, 0) < PILOT_PER_CELL:
            seen[cell] = seen.get(cell, 0) + 1
            chosen.append(r)
    return chosen


# ----------------------------------------------------------------------------- #
# Main
# ----------------------------------------------------------------------------- #
def main() -> None:
    pilot = "--pilot" in sys.argv
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not found in .env (needed for Judge A).")
    client = OpenAI(api_key=api_key)

    try:
        available = [m.model for m in ollama.list().models]
        if QWEN_MODEL not in available:
            raise SystemExit(f"Model '{QWEN_MODEL}' not in Ollama. Pull: ollama pull {QWEN_MODEL}")
    except SystemExit:
        raise
    except Exception as exc:                            # noqa: BLE001
        raise SystemExit(f"Could not reach Ollama ({type(exc).__name__}: {exc}). Is it running?")

    if not INPUT_PATH.exists():
        raise SystemExit(f"{INPUT_PATH} not found. Run p1.py first.")
    all_in = [json.loads(line) for line in open(INPUT_PATH, encoding="utf-8")]

    if pilot:
        subset = pick_pilot(all_in)
        print(f"PILOT MODE: {len(subset)} stratified examples "
              f"({PILOT_PER_CELL} per response x distress cell)\n")
        # pilot writes to a separate file so it never pollutes the real run
        global JSONL_PATH, CSV_PATH
        JSONL_PATH = Path("data/pilot_audited.jsonl")
        CSV_PATH = Path("data/pilot_audited.csv")
    else:
        subset = all_in

    done = load_done_ids()
    todo = [r for r in subset if r["id"] not in done]

    print(f"Judge A: {GPT_MODEL} (cloud) | Judge B: {QWEN_MODEL} (local)")
    print(f"Records: {len(subset)} | already done: {len(done)} | to audit now: {len(todo)}")
    print(f"Calls per example: 4 (2 judges x 2 tasks). Temperature={TEMPERATURE}\n")

    bar = tqdm(todo, total=len(todo), desc="Auditing", colour="cyan", unit="msg", ncols=100)
    for r in bar:
        r.update(judge_example("gpt", lambda p: call_gpt(client, p), r["text"]))
        r.update(judge_example("qwen", call_qwen, r["text"]))

        # --- derived agreement flags (computed, not asked) ---
        gen_resp = r.get("response")
        gen_dist = r.get("distress")
        r["gpt_resp_agree"] = (r["gpt_resp_top1"] == gen_resp)
        r["qwen_resp_agree"] = (r["qwen_resp_top1"] == gen_resp)
        r["gpt_dist_agree"] = (r["gpt_dist_level"] == gen_dist)
        r["qwen_dist_agree"] = (r["qwen_dist_level"] == gen_dist)

        # judge-to-judge agreement (A<->B), independent of our label - per advisor:
        # this exposes how much the two judges share heuristics vs are truly independent.
        r["ab_resp_agree"] = (r["gpt_resp_top1"] is not None
                              and r["gpt_resp_top1"] == r["qwen_resp_top1"])
        r["ab_dist_agree"] = (r["gpt_dist_level"] is not None
                              and r["gpt_dist_level"] == r["qwen_dist_level"])

        # high-confidence consensus per task (strict, separate per task)
        r["resp_highconf"] = bool(
            r["gpt_resp_agree"] and r["qwen_resp_agree"]
            and (r["gpt_resp_conf"] or 0) >= 4 and (r["qwen_resp_conf"] or 0) >= 4
            and r["gpt_resp_ambiguous"] is False and r["qwen_resp_ambiguous"] is False
        )
        r["dist_highconf"] = bool(
            r["gpt_dist_agree"] and r["qwen_dist_agree"]
            and (r["gpt_dist_conf"] or 0) >= 4 and (r["qwen_dist_conf"] or 0) >= 4
            and r["gpt_dist_ambiguous"] is False and r["qwen_dist_ambiguous"] is False
        )
        # joint agreement (EXTRA stat only - never the sole filter)
        r["joint_agree"] = bool(
            r["gpt_resp_agree"] and r["qwen_resp_agree"]
            and r["gpt_dist_agree"] and r["qwen_dist_agree"]
        )

        append_record(r)
        bar.set_postfix(resp_hc=sum(1 for x in [r] if x["resp_highconf"]),
                        refresh=False)

    records = rebuild_csv()
    _summarize(records)


# ----------------------------------------------------------------------------- #
# Summary with SEPARATE statistics per task
# ----------------------------------------------------------------------------- #
def _summarize(records: list) -> None:
    n = len(records)
    if n == 0:
        print("No records.")
        return

    def pct(num):
        return f"{num}/{n} ({num / n * 100:.1f}%)"

    gpt_r = sum(1 for r in records if r.get("gpt_resp_agree"))
    qwen_r = sum(1 for r in records if r.get("qwen_resp_agree"))
    gpt_d = sum(1 for r in records if r.get("gpt_dist_agree"))
    qwen_d = sum(1 for r in records if r.get("qwen_dist_agree"))
    resp_hc = sum(1 for r in records if r.get("resp_highconf"))
    dist_hc = sum(1 for r in records if r.get("dist_highconf"))
    joint = sum(1 for r in records if r.get("joint_agree"))
    ab_resp = sum(1 for r in records if r.get("ab_resp_agree"))
    ab_dist = sum(1 for r in records if r.get("ab_dist_agree"))

    print("\n" + "=" * 66)
    print("BLIND DUAL-JUDGE AUDIT SUMMARY  (separate stats per task)")
    print("  generator: gemma2:27b | judges: gpt-4o-mini + qwen2.5:32b")
    print("=" * 66)
    print(f"Total records: {n}\n")
    print("RESPONSE task (dominant psychosocial response):")
    print(f"  Judge A (gpt)  agrees:    {pct(gpt_r)}")
    print(f"  Judge B (qwen) agrees:    {pct(qwen_r)}")
    print(f"  HIGH-CONFIDENCE subset:   {pct(resp_hc)}")
    print("\nDISTRESS task (expressed distress intensity):")
    print(f"  Judge A (gpt)  agrees:    {pct(gpt_d)}")
    print(f"  Judge B (qwen) agrees:    {pct(qwen_d)}")
    print(f"  HIGH-CONFIDENCE subset:   {pct(dist_hc)}")
    print(f"\nJoint agreement (both tasks, both judges) [extra stat]: {pct(joint)}")
    print(f"Judge-to-judge agreement A<->B  response: {pct(ab_resp)} | distress: {pct(ab_dist)}")
    print("  (high A<->B with lower label-agreement = judges share heuristics; interpret "
          "high-confidence yield with that in mind.)")

    # per-class agreement for the response task (which classes are harder)
    print("\nPer-class RESPONSE agreement (gpt / qwen):")
    for c in RESPONSES:
        cls = [r for r in records if r.get("response") == c]
        if cls:
            g = sum(1 for r in cls if r.get("gpt_resp_agree")) / len(cls) * 100
            q = sum(1 for r in cls if r.get("qwen_resp_agree")) / len(cls) * 100
            print(f"  {c:12s} n={len(cls):3d}  gpt={g:5.1f}%  qwen={q:5.1f}%")

    print("\nPer-level DISTRESS agreement (gpt / qwen):")
    for lvl in DISTRESS_LEVELS:
        cls = [r for r in records if r.get("distress") == lvl]
        if cls:
            g = sum(1 for r in cls if r.get("gpt_dist_agree")) / len(cls) * 100
            q = sum(1 for r in cls if r.get("qwen_dist_agree")) / len(cls) * 100
            print(f"  {lvl:8s} n={len(cls):3d}  gpt={g:5.1f}%  qwen={q:5.1f}%")

    print("=" * 66)
    print(f"Saved -> {JSONL_PATH}  +  {CSV_PATH}")
    print("Note: separate high-confidence subsets per task; joint is extra only.")
    print("Disagreements are kept for error analysis.")


if __name__ == "__main__":
    main()
