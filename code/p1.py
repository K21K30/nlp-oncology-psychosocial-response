"""
p1.py - Synthetic data generation for the dominant-psychosocial-response classifier.

Task: Dominant psychosocial response classification in oncology-related messages.
Given a single patient/caregiver message, predict the most salient expressed psychosocial
response (an emotional state OR a coping-related reaction) from 7 classes, plus an
expressed-distress-intensity level (low/medium/high). The task concerns expressed language
only and does NOT infer diagnosis, mental-health status, or clinical risk.

Generation is done LOCALLY via Ollama (gemma2:27b on an RTX 5090): no API quotas, no cost,
fully private. One generator for the whole dataset (methodological purity). Diversity comes
from attribute-based prompting; a two-layer label-leakage control keeps the target label out
of the text (prompt-level ban + post-generation filter).

Two labels per message:
  - response : dominant psychosocial response, one of 7 classes (see RESPONSE_CLASSES).
               Includes emotional states (anxiety, sadness, anger, hope, guilt) and
               coping-related reactions (denial, acceptance).
  - distress : expressed distress intensity, one of 3 classes (low, medium, high), defined
               by an explicit rubric (DISTRESS_RUBRIC) shared with the audit script.
"""

import csv
import json
import time
import random
import re
from pathlib import Path

from tqdm import tqdm
import ollama


# ----------------------------------------------------------------------------- #
# Configuration
# ----------------------------------------------------------------------------- #
MODEL_NAME = "gemma2:27b"          # local model via Ollama; fallback: "qwen2.5:32b"
N_SAMPLES = 800                    # total messages to generate
MAX_RETRIES = 3                    # retries per failed call
GENERATION_TEMPERATURE = 1.0       # high-ish for diversity (instructor: avoid determinism)

OUTPUT_DIR = Path("data")
JSONL_PATH = OUTPUT_DIR / "raw_dataset.jsonl"
CSV_PATH = OUTPUT_DIR / "raw_dataset.csv"

RANDOM_SEED = 42                   # reproducibility of the attribute sampling


# ----------------------------------------------------------------------------- #
# Label space (the two prediction targets)
# ----------------------------------------------------------------------------- #
# The 7 psychosocial-response classes, with short definitions. These definitions are the
# SAME ones shown to the audit judges (p1b_audit.py), so generation and evaluation share one
# taxonomy. The set mixes emotional states and coping-related reactions on purpose - hence
# "psychosocial response" rather than "emotion".
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

# Priority rule shown to the generator and judges to separate the close classes.
PRIORITY_RULE = (
    "Priority when classes are close:\n"
    "- improvement, winning, fighting, belief in a positive outcome -> hope.\n"
    "- acknowledging reality and calmly adjusting to it -> acceptance.\n"
    "- loss, unfairness, or pain remains dominant -> sadness, even if a 'keep going' phrase "
    "is present."
)

DISTRESS_LEVELS = ["low", "medium", "high"]

# Explicit rubric for the distress label. The SAME wording is used by the audit judges
# (p1b_audit.py), so generation and evaluation share one definition - this is what makes
# the label judgeable consistently instead of guessed.
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

# Extra GENERATION-ONLY guidance for high distress (NOT shown to judges; their rubric stays
# frozen). Per advisor: describe high via functional/emotional signs and varied manifestations,
# without mandating fixed "crisis words" that would make texts caricatured and easy to game.
HIGH_GENERATION_GUIDANCE = (
    "This message is HIGH distress: distress must clearly dominate the writer's experience. "
    "Show loss of emotional control, serious difficulty coping, or meaningful disruption to "
    "daily functioning. It may appear as emotional flooding, inability to function, physical "
    "signs of anxiety, trouble sleeping, desperation, overwhelming fear, extreme hopelessness, "
    "or urgent help-seeking. Vary how it manifests; do not rely on the seriousness of the "
    "diagnosis alone, and do not just say it is 'hard', 'sad', or 'scary' (that is medium). "
    "Avoid a calm, joking, or upbeat tone for this message."
)


# ----------------------------------------------------------------------------- #
# Attribute space (for diversity - NOT prediction targets)
# ----------------------------------------------------------------------------- #
ROLES = [
    "patient", "spouse", "adult child", "parent of a sick child",
    "close friend", "nurse or carer", "coworker", "neighbor",
    "support-group member",
]
STAGES = [
    "diagnosis", "treatment", "remission", "relapse", "palliative", "bereavement",
]
CANCER_TYPES = [
    "breast", "lung", "colon", "prostate", "leukemia", "lymphoma",
    "pancreatic", "ovarian", "melanoma", "brain",
]
TONES = ["formal", "casual", "messy"]
LENGTHS = ["short", "medium", "long"]
CHANNELS = ["online forum", "SMS", "personal journal"]
AGE_GROUPS = ["teen", "young adult", "middle-aged", "elderly"]

# Words that must never appear verbatim in a generated message (label leakage).
# Includes the label names plus obvious synonyms a model tends to fall back on.
LEAKAGE_TERMS = {
    "anxiety": ["anxiety", "anxious", "worried", "worry"],
    "sadness": ["sadness", "sad", "depression", "depressed", "low mood"],
    "acceptance": ["acceptance", "accepting", "accept"],
    "denial": ["denial", "denying", "in denial"],
    "anger": ["anger", "angry", "furious"],
    "hope": ["hope", "hopeful", "hopeless"],
    "guilt": ["guilt", "guilty"],
}
# Distress words we never want stated outright (the label must be shown, not named).
DISTRESS_WORDS = ["distress", "distressed", "low distress", "medium distress",
                  "high distress", "urgent", "emergency"]


# ----------------------------------------------------------------------------- #
# Prompt construction
# ----------------------------------------------------------------------------- #
def build_prompt(sample_attrs: dict) -> str:
    """Build a single generation prompt from a sampled attribute combination.

    The prompt explicitly forbids naming the target emotion/distress words so the
    label cannot leak into the text (methodological soundness, variant 3).
    """
    response = sample_attrs["response"]
    distress = sample_attrs["distress"]
    role = sample_attrs["role"]
    stage = sample_attrs["stage"]
    cancer = sample_attrs["cancer_type"]
    tone = sample_attrs["tone"]
    length = sample_attrs["length"]
    channel = sample_attrs["channel"]
    age = sample_attrs["age_group"]
    noisy = sample_attrs["noisy"]

    # Forbidden words for THIS sample: the response synonyms + all distress words.
    forbidden = set(LEAKAGE_TERMS.get(response, [])) | set(DISTRESS_WORDS)
    forbidden_list = ", ".join(sorted(forbidden))

    length_hint = {
        "short": "1 short sentence (under 20 words)",
        "medium": "2-3 sentences",
        "long": "4-6 sentences",
    }[length]

    noisy_hint = ""
    if noisy:
        noisy_hint = (
            "Make it noisy and informal: include some typos, abbreviations (e.g. 'u', 'rn', "
            "'thx'), and one irrelevant detail unrelated to the illness. "
        )

    response_def = RESPONSE_CLASSES[response]
    high_block = ("\n" + HIGH_GENERATION_GUIDANCE) if distress == "high" else ""

    prompt = f"""You are generating ONE realistic oncology-related message for a research
dataset. It is written by a {age} {role} during the {stage} stage, about {cancer} cancer,
posted on a {channel}, in a {tone} tone.

The target psychosocial response is **{response}** ({response_def})
It must be the MOST SALIENT response expressed by the writer - the one most central to the
message. Other emotions or reactions may appear naturally, but none should be equally strong
or more central than the target.
{PRIORITY_RULE}

The message must also convey an expressed-distress intensity of **{distress}**:
{DISTRESS_RUBRIC}{high_block}

Hard rules:
- Write {length_hint}.
- {noisy_hint}Show the target response and the distress level through situation, wording and
  tone - do NOT name them.
- Do NOT use any of these words or their close forms: {forbidden_list}.
- Do NOT mention the words "emotion", "response", "distress", "level", or any label name.
- Do NOT write a clinical diagnosis or make claims about the writer's mental-health status.
- Do NOT include precise numbers or measurements (no exact doses, lab values, dates).
- Output ONLY the message text. No quotes, no preamble, no explanation.
"""
    return prompt.strip()


# ----------------------------------------------------------------------------- #
# Attribute sampling
# ----------------------------------------------------------------------------- #
def sample_attributes(n: int) -> list:
    """Build `n` attribute combinations with balanced emotion/distress coverage.

    Emotion and distress are cycled to keep the classes roughly balanced; the other
    attributes are sampled uniformly at random for diversity. Roughly 25% of
    samples are flagged `noisy` for the robustness / error-analysis variant.
    """
    random.seed(RANDOM_SEED)
    samples = []
    for i in range(n):
        attrs = {
            "id": i,                                  # stable id for checkpointing
            "response": RESPONSES[i % len(RESPONSES)],  # cycle -> balanced response classes
            "distress": DISTRESS_LEVELS[i % len(DISTRESS_LEVELS)],  # cycle -> balanced
            "role": random.choice(ROLES),
            "stage": random.choice(STAGES),
            "cancer_type": random.choice(CANCER_TYPES),
            "tone": random.choice(TONES),
            "length": random.choice(LENGTHS),
            "channel": random.choice(CHANNELS),
            "age_group": random.choice(AGE_GROUPS),
            "noisy": random.random() < 0.25,          # ~25% noisy variants
        }
        samples.append(attrs)
    random.shuffle(samples)                            # avoid ordered label blocks
    return samples


# ----------------------------------------------------------------------------- #
# Leakage post-filter
# ----------------------------------------------------------------------------- #
def has_leakage(text: str, response: str) -> bool:
    """Return True if the generated text leaks the response or any distress word."""
    lowered = text.lower()
    forbidden = set(LEAKAGE_TERMS.get(response, [])) | set(DISTRESS_WORDS)
    for term in forbidden:
        # word-boundary match so 'low' inside 'slowly' does not trigger
        if re.search(r"\b" + re.escape(term) + r"\b", lowered):
            return True
    return False


# ----------------------------------------------------------------------------- #
# Ollama call with retries
# ----------------------------------------------------------------------------- #
def generate_one(prompt: str) -> str | None:
    """Call the local Ollama model once with retries. Returns text, or None on failure."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = ollama.generate(
                model=MODEL_NAME,
                prompt=prompt,
                options={"temperature": GENERATION_TEMPERATURE},
            )
            text = (response.get("response") or "").strip()
            text = text.strip('"').strip("'").strip()
            if text:
                return text
        except Exception as exc:                       # noqa: BLE001 (broad on purpose)
            tqdm.write(f"  [retry {attempt}/{MAX_RETRIES}] {type(exc).__name__}: {exc}")
            time.sleep(2.0 * attempt)
    return None


# ----------------------------------------------------------------------------- #
# Checkpoint helpers
# ----------------------------------------------------------------------------- #
def load_done_ids() -> set:
    """Read the JSONL checkpoint (if any) and return the set of already-saved ids."""
    done = set()
    if JSONL_PATH.exists():
        with open(JSONL_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def append_record(record: dict) -> None:
    """Append one record to the JSONL checkpoint immediately (crash-safe)."""
    with open(JSONL_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def rebuild_csv_from_jsonl() -> int:
    """Rebuild the CSV from the full JSONL. Returns the number of records written."""
    records = []
    with open(JSONL_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    records.sort(key=lambda r: r["id"])
    if records:
        fieldnames = list(records[0].keys())
        with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
    return len(records)


# ----------------------------------------------------------------------------- #
# Main
# ----------------------------------------------------------------------------- #
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # sanity check: is the model available in Ollama?
    try:
        available = [m.model for m in ollama.list().models]
        if MODEL_NAME not in available:
            raise SystemExit(
                f"Model '{MODEL_NAME}' not found in Ollama. Available: {available}\n"
                f"Pull it first with:  ollama pull {MODEL_NAME}"
            )
    except Exception as exc:                            # noqa: BLE001
        raise SystemExit(
            f"Could not reach the Ollama server ({type(exc).__name__}: {exc}).\n"
            "Make sure the Ollama app is running, then retry."
        )

    plan = sample_attributes(N_SAMPLES)
    done_ids = load_done_ids()
    todo = [a for a in plan if a["id"] not in done_ids]

    print(f"Model:   {MODEL_NAME} (local via Ollama)")
    print(f"Samples: {N_SAMPLES}")
    print(f"Already done (checkpoint): {len(done_ids)}")
    print(f"To generate this run:      {len(todo)}")
    print(f"Output:  {JSONL_PATH}  +  {CSV_PATH}\n")

    kept = 0
    leaked = 0
    failed = 0

    bar = tqdm(todo, total=len(todo), desc="Generating", colour="green",
               unit="msg", ncols=100)
    for attrs in bar:
        prompt = build_prompt(attrs)
        text = generate_one(prompt)

        if text is None:
            failed += 1
            bar.set_postfix(kept=kept, leaked=leaked, failed=failed)
            continue

        if has_leakage(text, attrs["response"]):
            leaked += 1
            bar.set_postfix(kept=kept, leaked=leaked, failed=failed)
            continue

        record = {
            "id": attrs["id"],
            "text": text,
            "response": attrs["response"],
            "distress": attrs["distress"],
            "role": attrs["role"],
            "stage": attrs["stage"],
            "cancer_type": attrs["cancer_type"],
            "tone": attrs["tone"],
            "length": attrs["length"],
            "channel": attrs["channel"],
            "age_group": attrs["age_group"],
            "noisy": attrs["noisy"],
        }
        append_record(record)          # crash-safe: write as we go
        kept += 1
        bar.set_postfix(kept=kept, leaked=leaked, failed=failed)

    total_in_file = rebuild_csv_from_jsonl()

    # --- summary (numbers for the defense: A5 in DEFENSE_CHECKLIST) ---
    attempted = len(todo)
    print("\n" + "=" * 60)
    print("GENERATION SUMMARY")
    print("=" * 60)
    print(f"Attempted this run:   {attempted}")
    print(f"Kept (clean):         {kept}")
    if attempted:
        print(f"Dropped (leak):       {leaked}  ({leaked / attempted * 100:.1f}% of attempted)")
    print(f"Failed (model):       {failed}")
    print(f"Total in dataset now: {total_in_file} / {N_SAMPLES}")
    print(f"Saved JSONL ->        {JSONL_PATH}")
    print(f"Saved CSV   ->        {CSV_PATH}")
    print("=" * 60)
    if total_in_file < N_SAMPLES:
        print(f"\n{N_SAMPLES - total_in_file} still missing (failed/leaked). "
              f"Just run the script again to top up - done ids are skipped.")


if __name__ == "__main__":
    main()
