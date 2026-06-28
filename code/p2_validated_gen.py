"""
p2_validated_gen.py
Validated synthetic generation with rejection sampling.

Architecture:
  - Generate candidate oncology-related messages.
  - Evaluate every candidate with TWO blind judges.
  - Response and distress are judged in separate calls.
  - Keep only strict passes in accepted.jsonl.
  - Preserve every judged candidate in candidates.jsonl.
  - Never overwrite intended labels with judge labels.

Dataset files:
  intended_*  : labels requested from the generator.
  judge_a/b   : blind judgments.
  final_*     : reserved for later human adjudication.

Modes:
  python p2_validated_gen.py
      Full generation. Target = 800 accepted examples.

  python p2_validated_gen.py --pilot
      Smoke test. Tries to obtain one accepted example per non-zero cell.

  python p2_validated_gen.py --yield-pilot
      Fixed-size diagnostic pilot.
      Generates 5 candidates for every response × low/medium cell.
      It does not stop after the first accepted example.

  python p2_validated_gen.py --challenge
      Generates acceptance + high candidates for later human review.
      These examples are not automatically added to the training set.

Requirements:
  - OPENAI_API_KEY in .env
  - Ollama running
  - gemma2:27b installed
  - qwen2.5:32b installed

Install:
  pip install openai ollama python-dotenv tqdm
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import ollama
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


# =============================================================================
# Experiment configuration
# =============================================================================

EXPERIMENT_ID = "gen_v6_low_medium"

GEN_MODEL = "gemma2:27b"
JUDGE_A_MODEL = "gpt-4o-mini"
JUDGE_B_MODEL = "qwen2.5:32b"

GEN_TEMPERATURE = 1.0
JUDGE_TEMPERATURE = 0.0

MIN_CONFIDENCE = 4
MAX_RETRIES = 3

# Full rejection-sampling limit.
# A multiplier of 8 allows a cell with ~12.5% yield to fill in principle.
MAX_ATTEMPTS_MULTIPLIER = 8

# Smoke pilot target and attempt limits.
PILOT_PER_CELL = 1
PILOT_MAX_ATTEMPTS_PER_CELL = 10

# Fixed diagnostic pilot:
# 7 response labels × 2 distress levels × 5 candidates = 70 candidates.
YIELD_PILOT_PER_CELL = 5
YIELD_PILOT_DISTRESS_LEVELS = ("low", "medium")
YIELD_PILOT_ATTEMPTS_MULTIPLIER = 4

# Separate acceptance + high challenge candidates.
CHALLENGE_ACCEPTANCE_HIGH = 15
CHALLENGE_ATTEMPTS_MULTIPLIER = 4

SLEEP_BETWEEN_CALLS = 0.15

BASE_DATA_DIR = Path("data")

# Assigned at runtime by configure_paths().
CANDIDATES_PATH = Path()
ACCEPTED_PATH = Path()
CHALLENGE_PATH = Path()


# =============================================================================
# Target accepted counts
# =============================================================================

# Number of ACCEPTED examples required in each cell.
# Total = 800.
#
# acceptance + high is excluded from the primary dataset because the definitions
# are partly conflicting. It is generated separately as a human-reviewed
# challenge set.
TARGET_COUNTS = {
    "hope": {
        "low": 35,
        "medium": 50,
        "high": 5,
    },
    "guilt": {
        "low": 15,
        "medium": 40,
        "high": 35,
    },
    "anxiety": {
        "low": 5,
        "medium": 40,
        "high": 45,
    },
    "sadness": {
        "low": 5,
        "medium": 40,
        "high": 45,
    },
    "denial": {
        "low": 35,
        "medium": 45,
        "high": 10,
    },
    "anger": {
        "low": 10,
        "medium": 45,
        "high": 35,
    },
    "acceptance": {
        "low": 55,
        "medium": 35,
        "high": 0,
    },
}


# Top-up quotas: fixed number of VALID AUDITED candidates to add per cell, to reach the
# 2000-message corpus while fixing the class imbalance left by the interrupted run.
# These target audited candidates (any status except invalid_judgment), NOT strict acceptances.
# Sum = 650 -> 1375 existing + 650 = 2025 valid audited messages.
TOPUP_COUNTS = {
    "anger": {
        "low": 70,
        "medium": 100,
        "high": 70,
    },
    "denial": {
        "low": 90,
        "medium": 100,
        "high": 50,
    },
    "acceptance": {
        "low": 100,
        "medium": 70,
        "high": 0,
    },
}

TOPUP_MAX_ATTEMPTS_MULTIPLIER = 3


# =============================================================================
# Frozen response taxonomy and judge rubric
# =============================================================================

RESPONSE_CLASSES = {
    "anxiety": (
        "fear, worry, or dread about what is happening or may happen."
    ),
    "sadness": (
        "sorrow, grief, loss, or low mood; an expressed emotional state, "
        "not a clinical diagnosis."
    ),
    "anger": (
        "frustration, blame, protest, resentment, irritation, or hostility "
        "directed at a person, institution, the illness, or the situation. "
        "Unfairness or 'why me' language alone is not enough when grief or "
        "helplessness dominates."
    ),
    "hope": (
        "the speaker anticipates, wishes for, or actively strives toward a "
        "better future outcome, such as recovery, improvement, successful "
        "treatment, or overcoming the illness. Future-oriented optimism or "
        "fighting language is central."
    ),
    "guilt": (
        "self-blame, remorse, or feeling responsible for something bad."
    ),
    "denial": (
        "a coping reaction involving minimizing, rejecting, avoiding, or not "
        "engaging with the reality of the situation."
    ),
    "acceptance": (
        "the speaker acknowledges the current reality and is emotionally "
        "coming to terms with it. The dominant tone is calm, reflective, "
        "grounded, or quietly resigned. Acceptance does not depend on "
        "expecting improvement. The speaker may still pursue treatment, but "
        "'defeating' the illness or anticipating a better outcome is not the "
        "central response."
    ),
}

RESPONSES = list(RESPONSE_CLASSES.keys())
DISTRESS_LEVELS = ["low", "medium", "high"]

PRIORITY_RULE = (
    "Priority when classes are close:\n"
    "- improvement, winning, fighting, or belief in a positive outcome -> hope;\n"
    "- acknowledging reality and calmly adjusting to it -> acceptance;\n"
    "- loss, unfairness, helplessness, or pain remains dominant -> sadness, "
    "even if a 'keep going' phrase is present;\n"
    "- anger requires observable irritation, blame, protest, resentment, or "
    "hostility, not only 'why me' or unfairness."
)

# This rubric is used by the judges and remains frozen.
DISTRESS_RUBRIC = (
    "Judge ONLY the distress explicitly expressed in the text. Do NOT infer "
    "distress from the seriousness of the diagnosis, prognosis, treatment, "
    "or medical situation.\n"
    "- low: little or mild expressed distress; the speaker is mostly "
    "regulated, matter-of-fact, calm, or only slightly worried; no strong "
    "emotional or functional disruption is expressed.\n"
    "- medium: clear and sustained worry, sadness, fear, frustration, or "
    "emotional strain, but the speaker remains relatively coherent and able "
    "to cope; distress is significant but not overwhelming.\n"
    "- high: distress is overwhelming, acute, uncontrollable, or seriously "
    "disruptive; possible evidence includes panic, desperation, inability to "
    "cope or function, extreme hopelessness, emotional flooding, or "
    "crisis-like language.\n"
    "A serious diagnosis described calmly is NOT high. One strong emotional "
    "word without additional intensification is not automatically high. "
    "High requires strong linguistic evidence."
)


# =============================================================================
# Generation-only distress guidance
# =============================================================================

# These blocks affect only generation.
# They are NOT added to the judge prompts.

LOW_GENERATION_GUIDANCE = (
    "This message must express clearly LOW distress. The writer may mention "
    "one concern or one negative feeling, but remains calm, functional, and "
    "emotionally regulated overall. Avoid sustained rumination, repeated "
    "fear, sleep problems, concentration problems, desperation, loss of "
    "emotional control, or wording suggesting that the situation dominates "
    "daily life."
)

MEDIUM_GENERATION_GUIDANCE = (
    "This message must express clearly MEDIUM distress. The emotional strain "
    "must be stronger than ordinary mild concern. Show sustained worry, "
    "sadness, fear, frustration, or some difficulty concentrating or coping, "
    "while the writer remains able to function. The writer must not appear "
    "overwhelmed, panicked, desperate, emotionally out of control, or unable "
    "to function. A single mild concern is not enough for medium distress."
)

HIGH_GENERATION_GUIDANCE = (
    "This message must express clearly HIGH distress. Distress should dominate "
    "the writer's experience. Show loss of emotional control, serious "
    "difficulty coping, or meaningful disruption to daily functioning. "
    "Possible manifestations include emotional flooding, inability to "
    "function, physical signs of fear, inability to sleep, desperation, "
    "overwhelming fear, extreme hopelessness, or urgent help-seeking. Vary "
    "how the distress appears. Do not rely on the medical diagnosis alone. "
    "Do not merely say that the situation is 'hard', 'sad', or 'scary', "
    "because that would usually be medium. Avoid a calm, joking, or upbeat "
    "overall tone."
)

DISTRESS_CONTRAST_GUIDANCE = (
    "Distinguish the three levels carefully:\n"
    "- low = noticeable but limited concern; calm and functional overall;\n"
    "- medium = sustained and significant emotional strain, but still coping;\n"
    "- high = overwhelming, uncontrollable, or seriously disruptive distress."
)

LOW_MANIFESTATIONS = [
    "one brief concern while otherwise remaining calm",
    "mild uncertainty while maintaining the normal daily routine",
    "slight worry without impaired functioning",
    "a limited negative feeling that does not dominate the message",
    "brief concern followed by a regulated, matter-of-fact tone",
    "a small moment of unease without repeated rumination",
]

MEDIUM_MANIFESTATIONS = [
    "persistent worry while still managing normal responsibilities",
    "clear emotional strain with some difficulty concentrating",
    "repeated sadness or fear while retaining self-control",
    "noticeable trouble coping, but no emotional collapse",
    "significant concern that occupies the writer's thoughts without becoming overwhelming",
    "ongoing emotional pressure while the writer remains functional",
]


# =============================================================================
# Diversity attributes
# =============================================================================

ROLES = [
    "patient",
    "spouse",
    "adult child",
    "parent of a sick child",
    "close friend",
    "nurse or carer",
    "coworker",
    "neighbor",
    "support-group member",
]

ADULT_ONLY_ROLES = {
    "spouse",
    "adult child",
    "parent of a sick child",
    "nurse or carer",
    "coworker",
}

STAGES = [
    "diagnosis",
    "treatment",
    "remission",
    "relapse",
    "palliative care",
    "bereavement",
]

CANCER_TYPES = [
    "breast",
    "lung",
    "colon",
    "prostate",
    "leukemia",
    "lymphoma",
    "pancreatic",
    "ovarian",
    "melanoma",
    "brain",
]

TONES = [
    "formal",
    "casual",
    "messy",
]

LENGTHS = [
    "short",
    "medium",
    "long",
]

CHANNELS = [
    "online forum",
    "SMS",
    "personal journal",
]

AGE_GROUPS = [
    "teen",
    "young adult",
    "middle-aged",
    "elderly",
]

ADULT_AGE_GROUPS = [
    "young adult",
    "middle-aged",
    "elderly",
]


# =============================================================================
# Leakage controls
# =============================================================================

LEAKAGE_TERMS = {
    "anxiety": [
        "anxiety",
        "anxious",
        "worried",
        "worry",
    ],
    "sadness": [
        "sadness",
        "sad",
        "depression",
        "depressed",
        "low mood",
    ],
    "acceptance": [
        "acceptance",
        "accepting",
        "accept",
    ],
    "denial": [
        "denial",
        "denying",
        "in denial",
    ],
    "anger": [
        "anger",
        "angry",
        "furious",
    ],
    "hope": [
        "hope",
        "hopeful",
        "hopeless",
    ],
    "guilt": [
        "guilt",
        "guilty",
    ],
}

# "urgent" and "emergency" are intentionally not prohibited because they may
# naturally occur in high-distress help-seeking.
DISTRESS_WORDS = [
    "distress",
    "distressed",
    "low distress",
    "medium distress",
    "high distress",
]


# =============================================================================
# Path configuration
# =============================================================================

def configure_paths(mode: str) -> None:
    """
    Assign separate output directories for every experiment mode.

    This prevents pilot, full, yield-pilot, and challenge records from being
    mixed in the same checkpoint.
    """
    global CANDIDATES_PATH
    global ACCEPTED_PATH
    global CHALLENGE_PATH

    safe_mode = mode.replace("-", "_")
    output_dir = BASE_DATA_DIR / EXPERIMENT_ID / safe_mode

    CANDIDATES_PATH = output_dir / "candidates.jsonl"
    ACCEPTED_PATH = output_dir / "accepted.jsonl"
    CHALLENGE_PATH = output_dir / "challenge_acceptance_high.jsonl"


# =============================================================================
# Attribute and prompt construction
# =============================================================================

def sample_attrs(rng: random.Random) -> dict[str, Any]:
    """Sample realistic diversity attributes using a record-specific RNG."""
    role = rng.choice(ROLES)

    if role in ADULT_ONLY_ROLES:
        age_group = rng.choice(ADULT_AGE_GROUPS)
    else:
        age_group = rng.choice(AGE_GROUPS)

    return {
        "role": role,
        "stage": rng.choice(STAGES),
        "cancer_type": rng.choice(CANCER_TYPES),
        "tone": rng.choice(TONES),
        "length": rng.choice(LENGTHS),
        "channel": rng.choice(CHANNELS),
        "age_group": age_group,
        "noisy": rng.random() < 0.25,
    }


def build_distress_generation_guidance(
    distress: str,
    rng: random.Random,
) -> str:
    """
    Build generation-only instructions.

    The randomly selected manifestation is intentionally not stored as a
    dataset label. It is only used to diversify realization of the requested
    distress level.
    """
    if distress == "low":
        manifestation = rng.choice(LOW_MANIFESTATIONS)

        return (
            f"{LOW_GENERATION_GUIDANCE}\n\n"
            f"{DISTRESS_CONTRAST_GUIDANCE}\n\n"
            f"Use this manifestation pattern: {manifestation}."
        )

    if distress == "medium":
        manifestation = rng.choice(MEDIUM_MANIFESTATIONS)

        return (
            f"{MEDIUM_GENERATION_GUIDANCE}\n\n"
            f"{DISTRESS_CONTRAST_GUIDANCE}\n\n"
            f"Use this manifestation pattern: {manifestation}."
        )

    if distress == "high":
        return (
            f"{HIGH_GENERATION_GUIDANCE}\n\n"
            f"{DISTRESS_CONTRAST_GUIDANCE}"
        )

    raise ValueError(f"Unknown distress level: {distress}")


def build_generation_prompt(
    response: str,
    distress: str,
    attrs: dict[str, Any],
    rng: random.Random,
) -> str:
    """Build one conditional synthetic-generation prompt."""
    response_definition = RESPONSE_CLASSES[response]

    distress_guidance = build_distress_generation_guidance(
        distress=distress,
        rng=rng,
    )

    forbidden = (
        set(LEAKAGE_TERMS.get(response, []))
        | set(DISTRESS_WORDS)
    )
    forbidden_list = ", ".join(sorted(forbidden))

    length_hint = {
        "short": "1 short sentence under 20 words",
        "medium": "2 to 3 sentences",
        "long": "4 to 6 sentences",
    }[attrs["length"]]

    noisy_hint = ""

    if attrs["noisy"]:
        noisy_hint = (
            "Make the message somewhat noisy and informal: include a small "
            "number of natural typos or abbreviations such as 'u', 'rn', or "
            "'thx', plus one minor irrelevant detail unrelated to the illness. "
            "The text must remain understandable. "
        )

    return f"""
You are generating ONE realistic oncology-related message for a research dataset.

Writer and context:
- age group: {attrs["age_group"]}
- role: {attrs["role"]}
- cancer stage or context: {attrs["stage"]}
- cancer type: {attrs["cancer_type"]}
- communication channel: {attrs["channel"]}
- writing tone: {attrs["tone"]}

Target psychosocial response:
**{response}**

Definition:
{response_definition}

The target response must be the SINGLE MOST SALIENT psychosocial response in
the message. Other reactions may appear naturally, but none should be equally
strong or more central than the target.

{PRIORITY_RULE}

Target expressed-distress intensity:
**{distress}**

General distress definitions:
{DISTRESS_RUBRIC}

Generation-only instructions for this example:
{distress_guidance}

Hard rules:
- Write {length_hint}.
- {noisy_hint}Show the target response and distress intensity through the
  situation, wording, and tone. Do not explicitly name the labels.
- Do not use any of these words or their close forms:
  {forbidden_list}
- Do not mention the words "emotion", "response", "distress", "level", or any
  category name.
- Do not write a clinical mental-health diagnosis.
- Do not make claims about the writer having a psychiatric disorder.
- Do not include precise measurements or exact medical test values.
- Output ONLY the message text.
- Do not add quotation marks, a preamble, a label, or an explanation.
""".strip()


def response_judge_prompt(text: str) -> str:
    """Build the blind response-classification prompt."""
    definitions = "\n".join(
        f"- {label}: {definition}"
        for label, definition in RESPONSE_CLASSES.items()
    )

    return f"""
You are an expert annotator.

Read the oncology-related message and independently select the SINGLE most
salient psychosocial response from the label set.

Choose the response that is most central to the message, whether emotional or
coping-related. Do not assume that any predefined generation label exists. If
no class is clearly dominant, mark the message as ambiguous.

Label set:
{definitions}

{PRIORITY_RULE}

Message:
\"\"\"{text}\"\"\"

Return ONLY one valid JSON object with exactly these fields:
{{
  "dominant_label": "hope|guilt|anxiety|sadness|denial|anger|acceptance",
  "secondary_labels": ["zero or more labels"],
  "confidence": 1,
  "ambiguous": false
}}

Confidence must be an integer from 1 to 5.
""".strip()


def distress_judge_prompt(text: str) -> str:
    """Build the blind distress-classification prompt."""
    return f"""
You are an expert annotator.

Read the oncology-related message and independently rate ONLY the expressed
distress intensity.

Use this rubric:
{DISTRESS_RUBRIC}

Message:
\"\"\"{text}\"\"\"

Return ONLY one valid JSON object with exactly these fields:
{{
  "distress_level": "low|medium|high",
  "distress_confidence": 1,
  "distress_ambiguous": false
}}

Confidence must be an integer from 1 to 5.
""".strip()


# =============================================================================
# Parsing and normalization
# =============================================================================

def extract_json(raw: str) -> dict[str, Any] | None:
    """Extract the first complete JSON object from a model response."""
    if not raw:
        return None

    text = raw.strip()

    if text.startswith("```"):
        text = text.strip("`").strip()

        if text.lower().startswith("json"):
            text = text[4:].strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end < start:
        return None

    try:
        parsed = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None

    return parsed if isinstance(parsed, dict) else None


def norm_label(value: Any) -> str | None:
    if not isinstance(value, str):
        return None

    normalized = value.strip().lower()

    return normalized if normalized in RESPONSES else None


def norm_level(value: Any) -> str | None:
    if not isinstance(value, str):
        return None

    normalized = value.strip().lower()

    return normalized if normalized in DISTRESS_LEVELS else None


def to_confidence(value: Any) -> int | None:
    try:
        confidence = int(value)
    except (TypeError, ValueError):
        return None

    return confidence if 1 <= confidence <= 5 else None


def to_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        normalized = value.strip().lower()

        if normalized == "true":
            return True

        if normalized == "false":
            return False

    return None


def normalize_secondary_labels(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []

    output: list[str] = []

    for item in value:
        label = norm_label(item)

        if label and label not in output:
            output.append(label)

    return output


def normalize_text(text: str) -> str:
    """Normalization used for exact duplicate detection."""
    return re.sub(r"\s+", " ", text.strip().lower())


def has_leakage(text: str, response: str) -> bool:
    """Detect explicit target-label or distress-label leakage."""
    lowered = text.lower()

    forbidden = (
        set(LEAKAGE_TERMS.get(response, []))
        | set(DISTRESS_WORDS)
    )

    for term in forbidden:
        pattern = r"\b" + re.escape(term) + r"\b"

        if re.search(pattern, lowered):
            return True

    return False


# =============================================================================
# Ollama response helpers
# =============================================================================

def get_ollama_response_text(response: Any) -> str:
    if isinstance(response, dict):
        return str(response.get("response") or "")

    return str(getattr(response, "response", "") or "")


def get_available_ollama_models() -> set[str]:
    listing = ollama.list()

    if isinstance(listing, dict):
        models = listing.get("models", [])
    else:
        models = getattr(listing, "models", [])

    names: set[str] = set()

    for model in models:
        if isinstance(model, dict):
            name = model.get("model") or model.get("name")
        else:
            name = (
                getattr(model, "model", None)
                or getattr(model, "name", None)
            )

        if name:
            names.add(str(name))

    return names


# =============================================================================
# Model callers
# =============================================================================

def call_generator(prompt: str, seed: int) -> str:
    """Call the local Gemma generator."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = ollama.generate(
                model=GEN_MODEL,
                prompt=prompt,
                options={
                    "temperature": GEN_TEMPERATURE,
                    "seed": seed,
                },
            )

            text = get_ollama_response_text(response).strip()
            text = text.strip('"').strip("'").strip()

            if text:
                return text

        except Exception as exc:  # noqa: BLE001
            tqdm.write(
                f"[generator retry {attempt}] "
                f"{type(exc).__name__}: {exc}"
            )
            time.sleep(1.5 * attempt)

    return ""


def call_judge_a(client: OpenAI, prompt: str) -> str:
    """Call the cloud judge in JSON mode."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=JUDGE_A_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=JUDGE_TEMPERATURE,
                response_format={
                    "type": "json_object",
                },
            )

            return response.choices[0].message.content or ""

        except Exception as exc:  # noqa: BLE001
            tqdm.write(
                f"[judge A retry {attempt}] "
                f"{type(exc).__name__}: {exc}"
            )
            time.sleep(max(1.0, SLEEP_BETWEEN_CALLS * attempt * 3))

    return ""


def call_judge_b(prompt: str) -> str:
    """Call the local Qwen judge in JSON mode."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = ollama.generate(
                model=JUDGE_B_MODEL,
                prompt=prompt,
                format="json",
                options={
                    "temperature": JUDGE_TEMPERATURE,
                    "seed": 0,
                },
            )

            return get_ollama_response_text(response)

        except Exception as exc:  # noqa: BLE001
            tqdm.write(
                f"[judge B retry {attempt}] "
                f"{type(exc).__name__}: {exc}"
            )
            time.sleep(1.0 * attempt)

    return ""


def judge_both(client: OpenAI, text: str) -> dict[str, dict[str, Any]]:
    """
    Run two blind judges.

    Each judge receives two separate calls:
      1. response classification
      2. distress classification
    """
    output: dict[str, dict[str, Any]] = {}

    judge_callers: list[tuple[str, Callable[[str], str]]] = [
        (
            "a",
            lambda prompt: call_judge_a(client, prompt),
        ),
        (
            "b",
            call_judge_b,
        ),
    ]

    for judge_name, caller in judge_callers:
        response_raw = caller(response_judge_prompt(text))
        response_object = extract_json(response_raw) or {}

        time.sleep(SLEEP_BETWEEN_CALLS)

        distress_raw = caller(distress_judge_prompt(text))
        distress_object = extract_json(distress_raw) or {}

        output[judge_name] = {
            "response": norm_label(
                response_object.get("dominant_label")
            ),
            "response_confidence": to_confidence(
                response_object.get("confidence")
            ),
            "response_secondary": normalize_secondary_labels(
                response_object.get("secondary_labels")
            ),
            "response_ambiguous": to_bool(
                response_object.get("ambiguous")
            ),
            "distress": norm_level(
                distress_object.get("distress_level")
            ),
            "distress_confidence": to_confidence(
                distress_object.get("distress_confidence")
            ),
            "distress_ambiguous": to_bool(
                distress_object.get("distress_ambiguous")
            ),
        }

        time.sleep(SLEEP_BETWEEN_CALLS)

    return output


# =============================================================================
# Validation
# =============================================================================

def valid_judgment(judgment: dict[str, Any]) -> bool:
    """Check that all required judgment fields were parsed correctly."""
    return (
        judgment.get("response") in RESPONSES
        and judgment.get("distress") in DISTRESS_LEVELS
        and isinstance(
            judgment.get("response_confidence"),
            int,
        )
        and isinstance(
            judgment.get("distress_confidence"),
            int,
        )
        and isinstance(
            judgment.get("response_ambiguous"),
            bool,
        )
        and isinstance(
            judgment.get("distress_ambiguous"),
            bool,
        )
    )


def strict_pass(
    intended_response: str,
    intended_distress: str,
    judge_a: dict[str, Any],
    judge_b: dict[str, Any],
) -> bool:
    """Strict dual-judge acceptance criterion."""
    if not valid_judgment(judge_a):
        return False

    if not valid_judgment(judge_b):
        return False

    return (
        judge_a["response"] == intended_response
        and judge_b["response"] == intended_response
        and judge_a["distress"] == intended_distress
        and judge_b["distress"] == intended_distress
        and judge_a["response_confidence"] >= MIN_CONFIDENCE
        and judge_b["response_confidence"] >= MIN_CONFIDENCE
        and judge_a["distress_confidence"] >= MIN_CONFIDENCE
        and judge_b["distress_confidence"] >= MIN_CONFIDENCE
        and judge_a["response_ambiguous"] is False
        and judge_b["response_ambiguous"] is False
        and judge_a["distress_ambiguous"] is False
        and judge_b["distress_ambiguous"] is False
    )


def classify_status(
    intended_response: str,
    intended_distress: str,
    judge_a: dict[str, Any],
    judge_b: dict[str, Any],
) -> str:
    """Assign one mutually exclusive validation status."""
    if not valid_judgment(judge_a) or not valid_judgment(judge_b):
        return "invalid_judgment"

    if strict_pass(
        intended_response=intended_response,
        intended_distress=intended_distress,
        judge_a=judge_a,
        judge_b=judge_b,
    ):
        return "accepted"

    any_ambiguity = (
        judge_a["response_ambiguous"]
        or judge_b["response_ambiguous"]
        or judge_a["distress_ambiguous"]
        or judge_b["distress_ambiguous"]
    )

    if any_ambiguity:
        return "ambiguous"

    judges_agree = (
        judge_a["response"] == judge_b["response"]
        and judge_a["distress"] == judge_b["distress"]
    )

    if judges_agree:
        return "judges_agree_label_mismatch"

    return "judge_disagreement"


# =============================================================================
# JSONL and checkpoint helpers
# =============================================================================

def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    if not path.exists():
        return records

    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                tqdm.write(
                    f"Skipping invalid JSON in {path}, "
                    f"line {line_number}"
                )
                continue

            if isinstance(record, dict):
                records.append(record)

    return records


def load_seen_texts() -> set[str]:
    """Load texts already written in the current experiment mode."""
    seen: set[str] = set()

    for path in (
        CANDIDATES_PATH,
        ACCEPTED_PATH,
        CHALLENGE_PATH,
    ):
        for record in read_jsonl(path):
            text = record.get("text")

            if isinstance(text, str) and text.strip():
                seen.add(normalize_text(text))

    return seen


def count_accepted_per_cell() -> dict[tuple[str, str], int]:
    """Count accepted checkpoint records for this experiment."""
    counts: dict[tuple[str, str], int] = {}

    for record in read_jsonl(ACCEPTED_PATH):
        if record.get("experiment_id") != EXPERIMENT_ID:
            continue

        if record.get("validation_status") != "accepted":
            continue

        response = record.get("intended_response")
        distress = record.get("intended_distress")

        if response not in RESPONSES:
            continue

        if distress not in DISTRESS_LEVELS:
            continue

        cell = (response, distress)
        counts[cell] = counts.get(cell, 0) + 1

    return counts


def count_records_per_cell(
    path: Path,
) -> dict[tuple[str, str], int]:
    """Count all valid records per intended cell."""
    counts: dict[tuple[str, str], int] = {}

    for record in read_jsonl(path):
        if record.get("experiment_id") != EXPERIMENT_ID:
            continue

        response = record.get("intended_response")
        distress = record.get("intended_distress")

        if response not in RESPONSES:
            continue

        if distress not in DISTRESS_LEVELS:
            continue

        cell = (response, distress)
        counts[cell] = counts.get(cell, 0) + 1

    return counts


def count_valid_audited_per_cell(
    path: Path,
) -> dict[tuple[str, str], int]:
    """Count valid audited candidates per intended cell (excludes invalid_judgment)."""
    counts: dict[tuple[str, str], int] = {}

    for record in read_jsonl(path):
        if record.get("experiment_id") != EXPERIMENT_ID:
            continue

        if record.get("validation_status") == "invalid_judgment":
            continue

        response = record.get("intended_response")
        distress = record.get("intended_distress")

        if response not in RESPONSES:
            continue

        if distress not in DISTRESS_LEVELS:
            continue

        cell = (response, distress)
        counts[cell] = counts.get(cell, 0) + 1

    return counts


# =============================================================================
# Candidate generation
# =============================================================================

def create_candidate_record(
    client: OpenAI,
    response: str,
    distress: str,
    seen_texts: set[str],
) -> dict[str, Any] | None:
    """
    Generate and judge one unique candidate.

    Returns None when generation failed, leakage was detected, or the text was
    an exact duplicate.
    """
    seed = random.randint(0, 2_147_483_647)
    rng = random.Random(seed)

    attrs = sample_attrs(rng)

    prompt = build_generation_prompt(
        response=response,
        distress=distress,
        attrs=attrs,
        rng=rng,
    )

    text = call_generator(
        prompt=prompt,
        seed=seed,
    ).strip()

    if not text:
        return None

    if has_leakage(text, response):
        return None

    normalized = normalize_text(text)

    if normalized in seen_texts:
        return None

    # Mark before judging so the same generated text is not accepted again
    # after an API or parsing failure.
    seen_texts.add(normalized)

    verdicts = judge_both(
        client=client,
        text=text,
    )

    judge_a = verdicts["a"]
    judge_b = verdicts["b"]

    status = classify_status(
        intended_response=response,
        intended_distress=distress,
        judge_a=judge_a,
        judge_b=judge_b,
    )

    return {
        "id": f"syn_{uuid.uuid4().hex[:12]}",
        "experiment_id": EXPERIMENT_ID,
        "text": text,
        "intended_response": response,
        "intended_distress": distress,
        "generator_model": GEN_MODEL,
        "generator_temperature": GEN_TEMPERATURE,
        "generation_seed": seed,
        "prompt_version": EXPERIMENT_ID,
        "attributes": attrs,
        "judge_a_model": JUDGE_A_MODEL,
        "judge_b_model": JUDGE_B_MODEL,
        "judge_a": judge_a,
        "judge_b": judge_b,
        "validation_status": status,
        "strict_pass": status == "accepted",
        "final_response": None,
        "final_distress": None,
        "human_review_status": None,
        "split": None,
        "created_at_unix": int(time.time()),
    }


# =============================================================================
# Strict accepted-target generation
# =============================================================================

def generate_cell_until_target(
    client: OpenAI,
    response: str,
    distress: str,
    target: int,
    already_accepted: int,
    seen_texts: set[str],
    progress_bar: tqdm,
    mode: str,
) -> int:
    """Generate a cell until its accepted quota or attempt cap is reached."""
    if target <= 0:
        return already_accepted

    accepted = already_accepted
    remaining = max(0, target - already_accepted)

    if remaining == 0:
        return accepted

    if mode == "pilot":
        max_attempts = max(
            PILOT_MAX_ATTEMPTS_PER_CELL,
            remaining * MAX_ATTEMPTS_MULTIPLIER,
        )
    else:
        max_attempts = remaining * MAX_ATTEMPTS_MULTIPLIER

    attempts = 0
    judged_candidates = 0

    while accepted < target and attempts < max_attempts:
        attempts += 1

        record = create_candidate_record(
            client=client,
            response=response,
            distress=distress,
            seen_texts=seen_texts,
        )

        if record is None:
            continue

        judged_candidates += 1
        append_jsonl(CANDIDATES_PATH, record)

        if record["validation_status"] == "accepted":
            append_jsonl(ACCEPTED_PATH, record)
            accepted += 1
            progress_bar.update(1)

        progress_bar.set_postfix(
            cell=f"{response[:4]}/{distress[:3]}",
            accepted=accepted,
            attempts=attempts,
            judged=judged_candidates,
            refresh=False,
        )

        time.sleep(SLEEP_BETWEEN_CALLS)

    if accepted < target:
        tqdm.write(
            f"WARNING {response}/{distress}: "
            f"only {accepted}/{target} accepted "
            f"after {attempts} attempts "
            f"({judged_candidates} unique judged candidates)"
        )

    return accepted


def run_strict_target_mode(
    client: OpenAI,
    mode: str,
    seen_texts: set[str],
) -> None:
    """Run smoke pilot or full 800 accepted-target generation."""
    targets: dict[tuple[str, str], int] = {}

    for response, distress_counts in TARGET_COUNTS.items():
        for distress, count in distress_counts.items():
            if count == 0:
                continue

            if mode == "pilot":
                targets[(response, distress)] = PILOT_PER_CELL
            else:
                targets[(response, distress)] = count

    checkpoint = count_accepted_per_cell()

    total_target = sum(targets.values())

    total_done = sum(
        min(
            checkpoint.get(cell, 0),
            target,
        )
        for cell, target in targets.items()
    )

    print(
        f"Generator: {GEN_MODEL} | "
        f"Judges: {JUDGE_A_MODEL} + {JUDGE_B_MODEL}"
    )
    print(f"Mode: {mode.upper()}")
    print(f"Experiment: {EXPERIMENT_ID}")
    print(f"Output directory: {CANDIDATES_PATH.parent}")
    print(
        f"Target accepted: {total_target} | "
        f"checkpoint accepted: {total_done}"
    )
    print(f"Seen texts: {len(seen_texts)}")
    print(
        f"Cells: {len(targets)} "
        f"(acceptance + high excluded)"
    )
    print()

    progress_bar = tqdm(
        total=total_target,
        initial=total_done,
        desc="Accepted",
        colour="green",
        unit="example",
        ncols=110,
    )

    for (response, distress), target in targets.items():
        already = min(
            checkpoint.get((response, distress), 0),
            target,
        )

        generate_cell_until_target(
            client=client,
            response=response,
            distress=distress,
            target=target,
            already_accepted=already,
            seen_texts=seen_texts,
            progress_bar=progress_bar,
            mode=mode,
        )

    progress_bar.close()

    final_counts = count_accepted_per_cell()

    final_total = sum(
        min(
            final_counts.get(cell, 0),
            target,
        )
        for cell, target in targets.items()
    )

    print()
    print(f"Finished accepted target: {final_total}/{total_target}")
    print(f"Candidates log: {CANDIDATES_PATH}")
    print(f"Accepted set:   {ACCEPTED_PATH}")

    if mode == "full":
        print()
        print(
            "Next recommended steps: run the human check, generate the "
            "challenge set, then create train/validation/test splits."
        )


# =============================================================================
# Fixed-size yield pilot
# =============================================================================

def run_yield_pilot(
    client: OpenAI,
    seen_texts: set[str],
) -> None:
    """
    Generate a fixed number of candidates per low/medium cell.

    Unlike --pilot, this mode does not stop after the first accepted example.
    It is intended for unbiased per-cell yield estimation.
    """
    cells = [
        (response, distress)
        for response in RESPONSES
        for distress in YIELD_PILOT_DISTRESS_LEVELS
    ]

    existing_counts = count_records_per_cell(CANDIDATES_PATH)

    total_target = len(cells) * YIELD_PILOT_PER_CELL
    total_done = sum(
        min(
            existing_counts.get(cell, 0),
            YIELD_PILOT_PER_CELL,
        )
        for cell in cells
    )

    print(
        f"Generator: {GEN_MODEL} | "
        f"Judges: {JUDGE_A_MODEL} + {JUDGE_B_MODEL}"
    )
    print("Mode: YIELD-PILOT")
    print(f"Experiment: {EXPERIMENT_ID}")
    print(f"Output directory: {CANDIDATES_PATH.parent}")
    print(
        f"Target judged candidates: {total_target} | "
        f"checkpoint candidates: {total_done}"
    )
    print(
        f"Cells: {len(cells)} "
        f"({len(RESPONSES)} responses × low/medium)"
    )
    print(f"Seen texts: {len(seen_texts)}")
    print()

    progress_bar = tqdm(
        total=total_target,
        initial=total_done,
        desc="Judged candidates",
        colour="cyan",
        unit="example",
        ncols=110,
    )

    for response, distress in cells:
        already = min(
            existing_counts.get((response, distress), 0),
            YIELD_PILOT_PER_CELL,
        )

        needed = YIELD_PILOT_PER_CELL - already

        if needed <= 0:
            continue

        produced = 0
        attempts = 0
        max_attempts = (
            needed
            * YIELD_PILOT_ATTEMPTS_MULTIPLIER
        )

        while produced < needed and attempts < max_attempts:
            attempts += 1

            record = create_candidate_record(
                client=client,
                response=response,
                distress=distress,
                seen_texts=seen_texts,
            )

            if record is None:
                continue

            append_jsonl(CANDIDATES_PATH, record)

            if record["validation_status"] == "accepted":
                append_jsonl(ACCEPTED_PATH, record)

            produced += 1
            progress_bar.update(1)

            progress_bar.set_postfix(
                cell=f"{response[:4]}/{distress[:3]}",
                generated=already + produced,
                attempts=attempts,
                status=record["validation_status"],
                refresh=False,
            )

        if produced < needed:
            tqdm.write(
                f"WARNING {response}/{distress}: "
                f"created only {produced}/{needed} new judged candidates "
                f"after {attempts} attempts"
            )

    progress_bar.close()

    print()
    print(f"Yield-pilot candidates: {CANDIDATES_PATH}")
    print(f"Strict passes:          {ACCEPTED_PATH}")
    print()
    print(
        "Run the analysis script on candidates.jsonl to calculate the "
        "true overall and per-cell validation yield."
    )


# =============================================================================
# Challenge-set generation
# =============================================================================

def run_challenge_mode(
    client: OpenAI,
    seen_texts: set[str],
) -> None:
    """
    Generate acceptance + high examples for human review.

    No candidate is automatically added to accepted.jsonl because this cell is
    conceptually difficult and partly conflicts with the response definition.
    """
    response = "acceptance"
    distress = "high"

    existing_records = count_records_per_cell(CHALLENGE_PATH)
    already = min(
        existing_records.get((response, distress), 0),
        CHALLENGE_ACCEPTANCE_HIGH,
    )

    needed = CHALLENGE_ACCEPTANCE_HIGH - already

    print(
        f"Generator: {GEN_MODEL} | "
        f"Judges: {JUDGE_A_MODEL} + {JUDGE_B_MODEL}"
    )
    print("Mode: CHALLENGE")
    print(f"Experiment: {EXPERIMENT_ID}")
    print(f"Output directory: {CHALLENGE_PATH.parent}")
    print(
        f"Target acceptance/high candidates: "
        f"{CHALLENGE_ACCEPTANCE_HIGH}"
    )
    print(f"Checkpoint candidates: {already}")
    print(f"Seen texts: {len(seen_texts)}")
    print()

    if needed <= 0:
        print("Challenge target is already complete.")
        print(f"Challenge set: {CHALLENGE_PATH}")
        return

    progress_bar = tqdm(
        total=CHALLENGE_ACCEPTANCE_HIGH,
        initial=already,
        desc="Challenge candidates",
        colour="yellow",
        unit="example",
        ncols=110,
    )

    produced = 0
    attempts = 0
    max_attempts = (
        needed
        * CHALLENGE_ATTEMPTS_MULTIPLIER
    )

    while produced < needed and attempts < max_attempts:
        attempts += 1

        record = create_candidate_record(
            client=client,
            response=response,
            distress=distress,
            seen_texts=seen_texts,
        )

        if record is None:
            continue

        record["challenge_cell"] = True
        record["requires_human_review"] = True

        append_jsonl(CHALLENGE_PATH, record)

        produced += 1
        progress_bar.update(1)

        progress_bar.set_postfix(
            produced=already + produced,
            attempts=attempts,
            status=record["validation_status"],
            refresh=False,
        )

    progress_bar.close()

    if produced < needed:
        print(
            f"WARNING: produced only {produced}/{needed} new challenge "
            f"candidates after {attempts} attempts."
        )

    print()
    print(f"Challenge set: {CHALLENGE_PATH}")
    print(
        "These records require blind human review before any are used as "
        "evaluation examples."
    )


# =============================================================================
# Top-up generation
# =============================================================================

def run_topup_mode(
    client: OpenAI,
    seen_texts: set[str],
) -> None:
    """
    Generate a fixed number of VALID AUDITED candidates per selected cell.

    Unlike the strict-target modes, this does not stop at strict acceptance. A candidate counts
    toward the top-up target whenever its validation_status is not invalid_judgment, because the
    2000-message corpus requirement is about audited messages, not strict acceptances. Records
    are written to this mode's own candidates.jsonl (separate topup/ directory) and never touch
    the existing full/ pool.
    """
    cells = {
        (response, distress): target
        for response, levels in TOPUP_COUNTS.items()
        for distress, target in levels.items()
        if target > 0
    }

    existing_counts = count_valid_audited_per_cell(CANDIDATES_PATH)

    total_target = sum(cells.values())
    total_done = sum(
        min(existing_counts.get(cell, 0), target)
        for cell, target in cells.items()
    )

    print(
        f"Generator: {GEN_MODEL} | "
        f"Judges: {JUDGE_A_MODEL} + {JUDGE_B_MODEL}"
    )
    print("Mode: TOPUP")
    print(f"Experiment: {EXPERIMENT_ID}")
    print(f"Output directory: {CANDIDATES_PATH.parent}")
    print(
        f"Target valid audited candidates: {total_target} | "
        f"checkpoint: {total_done}"
    )
    print(f"Seen texts: {len(seen_texts)}")
    print(f"Cells: {len(cells)} (anger, denial, acceptance top-up)")
    print()

    progress_bar = tqdm(
        total=total_target,
        initial=total_done,
        desc="Top-up audited",
        colour="magenta",
        unit="example",
        ncols=110,
    )

    for (response, distress), target in cells.items():
        already = min(
            existing_counts.get((response, distress), 0),
            target,
        )

        needed = target - already

        if needed <= 0:
            continue

        valid_created = 0
        attempts = 0
        max_attempts = max(
            needed * TOPUP_MAX_ATTEMPTS_MULTIPLIER,
            needed + 10,
        )

        while valid_created < needed and attempts < max_attempts:
            attempts += 1

            record = create_candidate_record(
                client=client,
                response=response,
                distress=distress,
                seen_texts=seen_texts,
            )

            if record is None:
                continue

            # Preserve every unique judged candidate.
            append_jsonl(CANDIDATES_PATH, record)

            # Invalid parsing/API judgments do not count toward the 2000 corpus.
            if record["validation_status"] == "invalid_judgment":
                continue

            valid_created += 1
            progress_bar.update(1)

            progress_bar.set_postfix(
                cell=f"{response[:4]}/{distress[:3]}",
                valid=already + valid_created,
                attempts=attempts,
                status=record["validation_status"],
                refresh=False,
            )

            time.sleep(SLEEP_BETWEEN_CALLS)

        if valid_created < needed:
            tqdm.write(
                f"WARNING {response}/{distress}: "
                f"only {valid_created}/{needed} new valid audited "
                f"candidates after {attempts} attempts"
            )

    progress_bar.close()

    print()
    print(f"Top-up candidates: {CANDIDATES_PATH}")
    print(
        "Next: merge this topup pool with the full pool, then re-run the tier "
        "script (p3) to rebuild final labels and check balance by final_response."
    )


# =============================================================================
# Diagnostics
# =============================================================================

def print_current_status_summary() -> None:
    """Print status counts for the current candidates file."""
    records = read_jsonl(CANDIDATES_PATH)

    if not records:
        return

    statuses = Counter(
        record.get("validation_status", "missing")
        for record in records
    )

    print()
    print("Current candidates status summary:")

    for status, count in statuses.most_common():
        print(f"  {status:32s} {count:5d}")


# =============================================================================
# CLI and startup checks
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validated oncology synthetic generation with dual blind judges."
        )
    )

    mode_group = parser.add_mutually_exclusive_group()

    mode_group.add_argument(
        "--pilot",
        action="store_true",
        help=(
            "Smoke test: obtain one accepted example per non-zero cell."
        ),
    )

    mode_group.add_argument(
        "--yield-pilot",
        action="store_true",
        help=(
            "Generate a fixed set of 70 low/medium candidates for yield analysis."
        ),
    )

    mode_group.add_argument(
        "--challenge",
        action="store_true",
        help=(
            "Generate acceptance + high candidates for human review."
        ),
    )

    mode_group.add_argument(
        "--topup",
        action="store_true",
        help=(
            "Generate a fixed number of audited candidates for selected "
            "cells (anger, denial, acceptance) to reach the 2000 corpus."
        ),
    )

    return parser.parse_args()


def resolve_mode(args: argparse.Namespace) -> str:
    if args.pilot:
        return "pilot"

    if args.yield_pilot:
        return "yield_pilot"

    if args.challenge:
        return "challenge"

    if args.topup:
        return "topup"

    return "full"


def initialize_openai_client() -> OpenAI:
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY was not found. Add it to the .env file."
        )

    return OpenAI(
        api_key=api_key,
        timeout=90.0,
    )


def verify_ollama_models() -> None:
    try:
        available = get_available_ollama_models()
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"Ollama is not reachable: "
            f"{type(exc).__name__}: {exc}. "
            f"Make sure Ollama is running."
        ) from exc

    missing = [
        model
        for model in (GEN_MODEL, JUDGE_B_MODEL)
        if model not in available
    ]

    if missing:
        commands = "\n".join(
            f"  ollama pull {model}"
            for model in missing
        )

        raise SystemExit(
            "Missing Ollama models:\n"
            + "\n".join(f"  - {model}" for model in missing)
            + "\nInstall them with:\n"
            + commands
        )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()
    mode = resolve_mode(args)

    configure_paths(mode)

    client = initialize_openai_client()
    verify_ollama_models()

    seen_texts = load_seen_texts()

    try:
        if mode in {"pilot", "full"}:
            run_strict_target_mode(
                client=client,
                mode=mode,
                seen_texts=seen_texts,
            )

        elif mode == "yield_pilot":
            run_yield_pilot(
                client=client,
                seen_texts=seen_texts,
            )

        elif mode == "challenge":
            run_challenge_mode(
                client=client,
                seen_texts=seen_texts,
            )

        elif mode == "topup":
            run_topup_mode(
                client=client,
                seen_texts=seen_texts,
            )

        else:
            raise RuntimeError(f"Unsupported mode: {mode}")

    except KeyboardInterrupt:
        print()
        print("Interrupted by user.")
        print("All completed records were checkpointed to JSONL files.")
        raise SystemExit(130)

    print_current_status_summary()


if __name__ == "__main__":
    main()