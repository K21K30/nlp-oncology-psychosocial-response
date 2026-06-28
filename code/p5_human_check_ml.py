"""
p5_human_check.py

Compatible with Python 3.10.10.

Blind human annotation and scoring for the frozen strict test set.

ANNOTATION
----------

python p5_human_check.py annotate ^
    --input "data\\gen_v6_low_medium\\splits\\test.jsonl" ^
    --annotator kt

SCORING
-------

python p5_human_check.py score ^
    --input "data\\gen_v6_low_medium\\splits\\test.jsonl" ^
    --annotations "data\\gen_v6_low_medium\\splits\\human_annotations_kt.jsonl" ^
    --merged-output "data\\gen_v6_low_medium\\splits\\test_with_human_annotations.jsonl" ^
    --report-output "data\\gen_v6_low_medium\\splits\\human_check_report.json" ^
    --subsets-dir "data\\gen_v6_low_medium\\splits\\human_confirmed"

The annotation mode does not display:
    - intended labels
    - final labels
    - LLM judge labels
    - model predictions
    - agreement feedback
"""

import argparse
import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# =============================================================================
# Configuration
# =============================================================================

RESPONSE_LABELS = [
    "anxiety",
    "sadness",
    "anger",
    "hope",
    "guilt",
    "denial",
    "acceptance",
]

DISTRESS_LABELS = [
    "low",
    "medium",
    "high",
]

DISTRESS_TO_INT = {
    "low": 0,
    "medium": 1,
    "high": 2,
}

DEFAULT_SEED = 20260628
DEFAULT_RUBRIC_VERSION = "human_check_v1"


# =============================================================================
# Frozen rubric
# =============================================================================

RESPONSE_DEFINITIONS = {
    "anxiety": (
        "Fear, worry, uncertainty, or dread about what is happening "
        "or may happen."
    ),
    "sadness": (
        "Sorrow, grief, loss, helplessness, or emotional pain. "
        "This is an expressed emotional state, not a clinical diagnosis."
    ),
    "anger": (
        "Frustration, irritation, blame, protest, resentment, or hostility "
        "directed toward a person, institution, illness, or situation. "
        "Unfairness or 'why me' language alone is not sufficient when grief "
        "or helplessness dominates."
    ),
    "hope": (
        "The speaker expects, wishes for, or actively strives toward a better "
        "future outcome. Future-oriented optimism, recovery, improvement, "
        "successful treatment, fighting, or overcoming the illness is central."
    ),
    "guilt": (
        "Self-blame, remorse, or feeling responsible for something bad."
    ),
    "denial": (
        "Minimizing, rejecting, avoiding, or not engaging with the reality "
        "of the situation."
    ),
    "acceptance": (
        "Acknowledging the current reality and emotionally coming to terms "
        "with it. The dominant tone is calm, reflective, grounded, or quietly "
        "resigned. Acceptance does not depend on expecting improvement."
    ),
}

PRIORITY_RULES = [
    (
        "Improvement, winning, fighting, or belief in a positive future "
        "outcome usually indicates hope."
    ),
    (
        "Acknowledging reality and calmly adjusting to it usually indicates "
        "acceptance."
    ),
    (
        "If loss, helplessness, unfairness, or pain remains dominant, choose "
        "sadness even if the text includes a phrase such as 'keep going'."
    ),
    (
        "Anger requires observable irritation, blame, protest, resentment, "
        "or hostility, not only 'why me' or unfairness."
    ),
]

DISTRESS_DEFINITIONS = {
    "low": (
        "Little or mild expressed distress. The speaker is mostly calm, "
        "regulated, matter-of-fact, or only slightly worried. No strong "
        "emotional or functional disruption is expressed."
    ),
    "medium": (
        "Clear and sustained worry, sadness, fear, frustration, or emotional "
        "strain, but the speaker remains relatively coherent and able to "
        "cope. Distress is significant but not overwhelming."
    ),
    "high": (
        "Distress is overwhelming, acute, uncontrollable, or seriously "
        "disruptive. Evidence may include panic, desperation, inability to "
        "cope or function, emotional flooding, extreme hopelessness, or "
        "urgent help-seeking."
    ),
}

DISTRESS_NOTES = [
    (
        "Judge only distress explicitly expressed in the text. Do not infer "
        "distress from the seriousness of the illness or prognosis."
    ),
    (
        "A serious diagnosis described calmly is not automatically high."
    ),
    (
        "One strong emotional word without further evidence is not "
        "automatically high."
    ),
]

CONFIDENCE_DEFINITIONS = {
    5: "Practically unambiguous.",
    4: "Confident, with only a minor plausible alternative.",
    3: "Most likely, but a reasonable alternative exists.",
    2: "Strong uncertainty.",
    1: "Cannot classify reliably.",
}


# =============================================================================
# Russian translations of the rubric (shown under the English text).
# Reference labels stay English; these are display aids only.
# =============================================================================

RESPONSE_DEFINITIONS_RU = {
    "anxiety": (
        "\u0422\u0440\u0435\u0432\u043e\u0433\u0430: \u0441\u0442\u0440\u0430\u0445, "
        "\u0431\u0435\u0441\u043f\u043e\u043a\u043e\u0439\u0441\u0442\u0432\u043e, "
        "\u043d\u0435\u043e\u043f\u0440\u0435\u0434\u0435\u043b\u0451\u043d\u043d\u043e\u0441\u0442\u044c "
        "\u0438\u043b\u0438 \u0443\u0436\u0430\u0441 \u043f\u043e \u043f\u043e\u0432\u043e\u0434\u0443 "
        "\u0442\u043e\u0433\u043e, \u0447\u0442\u043e \u043f\u0440\u043e\u0438\u0441\u0445\u043e\u0434\u0438\u0442 "
        "\u0438\u043b\u0438 \u043c\u043e\u0436\u0435\u0442 \u043f\u0440\u043e\u0438\u0437\u043e\u0439\u0442\u0438."
    ),
    "sadness": (
        "\u0413\u0440\u0443\u0441\u0442\u044c: \u043f\u0435\u0447\u0430\u043b\u044c, \u0433\u043e\u0440\u0435, "
        "\u0443\u0442\u0440\u0430\u0442\u0430, \u0431\u0435\u0441\u043f\u043e\u043c\u043e\u0449\u043d\u043e\u0441\u0442\u044c "
        "\u0438\u043b\u0438 \u0434\u0443\u0448\u0435\u0432\u043d\u0430\u044f \u0431\u043e\u043b\u044c. "
        "\u042d\u0442\u043e \u0432\u044b\u0440\u0430\u0436\u0435\u043d\u043d\u043e\u0435 "
        "\u044d\u043c\u043e\u0446\u0438\u043e\u043d\u0430\u043b\u044c\u043d\u043e\u0435 "
        "\u0441\u043e\u0441\u0442\u043e\u044f\u043d\u0438\u0435, \u0430 \u043d\u0435 "
        "\u043a\u043b\u0438\u043d\u0438\u0447\u0435\u0441\u043a\u0438\u0439 \u0434\u0438\u0430\u0433\u043d\u043e\u0437."
    ),
    "anger": (
        "\u0413\u043d\u0435\u0432: \u0444\u0440\u0443\u0441\u0442\u0440\u0430\u0446\u0438\u044f, "
        "\u0440\u0430\u0437\u0434\u0440\u0430\u0436\u0435\u043d\u0438\u0435, "
        "\u043e\u0431\u0432\u0438\u043d\u0435\u043d\u0438\u0435, \u043f\u0440\u043e\u0442\u0435\u0441\u0442, "
        "\u043e\u0431\u0438\u0434\u0430 \u0438\u043b\u0438 \u0432\u0440\u0430\u0436\u0434\u0435\u0431\u043d\u043e\u0441\u0442\u044c "
        "\u043a \u0447\u0435\u043b\u043e\u0432\u0435\u043a\u0443, "
        "\u0443\u0447\u0440\u0435\u0436\u0434\u0435\u043d\u0438\u044e, \u0431\u043e\u043b\u0435\u0437\u043d\u0438 "
        "\u0438\u043b\u0438 \u0441\u0438\u0442\u0443\u0430\u0446\u0438\u0438. \u0422\u043e\u043b\u044c\u043a\u043e "
        "\u043d\u0435\u0441\u043f\u0440\u0430\u0432\u0435\u0434\u043b\u0438\u0432\u043e\u0441\u0442\u044c "
        "\u0438\u043b\u0438 \u00ab\u043f\u043e\u0447\u0435\u043c\u0443 \u044f\u00bb "
        "\u043d\u0435\u0434\u043e\u0441\u0442\u0430\u0442\u043e\u0447\u043d\u043e, \u0435\u0441\u043b\u0438 "
        "\u043f\u0440\u0435\u043e\u0431\u043b\u0430\u0434\u0430\u0435\u0442 \u0433\u043e\u0440\u0435 "
        "\u0438\u043b\u0438 \u0431\u0435\u0441\u043f\u043e\u043c\u043e\u0449\u043d\u043e\u0441\u0442\u044c."
    ),
    "hope": (
        "\u041d\u0430\u0434\u0435\u0436\u0434\u0430: \u0433\u043e\u0432\u043e\u0440\u044f\u0449\u0438\u0439 "
        "\u043e\u0436\u0438\u0434\u0430\u0435\u0442, \u0436\u0435\u043b\u0430\u0435\u0442 "
        "\u0438\u043b\u0438 \u0430\u043a\u0442\u0438\u0432\u043d\u043e "
        "\u0441\u0442\u0440\u0435\u043c\u0438\u0442\u0441\u044f \u043a "
        "\u043b\u0443\u0447\u0448\u0435\u043c\u0443 \u0438\u0441\u0445\u043e\u0434\u0443. "
        "\u041e\u043f\u0442\u0438\u043c\u0438\u0437\u043c, "
        "\u0432\u044b\u0437\u0434\u043e\u0440\u043e\u0432\u043b\u0435\u043d\u0438\u0435, "
        "\u0443\u043b\u0443\u0447\u0448\u0435\u043d\u0438\u0435, "
        "\u0443\u0441\u043f\u0435\u0448\u043d\u043e\u0435 \u043b\u0435\u0447\u0435\u043d\u0438\u0435, "
        "\u0431\u043e\u0440\u044c\u0431\u0430 \u0441 \u0431\u043e\u043b\u0435\u0437\u043d\u044c\u044e."
    ),
    "guilt": (
        "\u0412\u0438\u043d\u0430: \u0441\u0430\u043c\u043e\u043e\u0431\u0432\u0438\u043d\u0435\u043d\u0438\u0435, "
        "\u0440\u0430\u0441\u043a\u0430\u044f\u043d\u0438\u0435 \u0438\u043b\u0438 "
        "\u0447\u0443\u0432\u0441\u0442\u0432\u043e "
        "\u043e\u0442\u0432\u0435\u0442\u0441\u0442\u0432\u0435\u043d\u043d\u043e\u0441\u0442\u0438 "
        "\u0437\u0430 \u0447\u0442\u043e-\u0442\u043e \u043f\u043b\u043e\u0445\u043e\u0435."
    ),
    "denial": (
        "\u041e\u0442\u0440\u0438\u0446\u0430\u043d\u0438\u0435: "
        "\u043f\u0440\u0435\u0443\u043c\u0435\u043d\u044c\u0448\u0435\u043d\u0438\u0435, "
        "\u043e\u0442\u043a\u0430\u0437, \u0438\u0437\u0431\u0435\u0433\u0430\u043d\u0438\u0435 "
        "\u0438\u043b\u0438 \u043d\u0435\u0436\u0435\u043b\u0430\u043d\u0438\u0435 "
        "\u043f\u0440\u0438\u0437\u043d\u0430\u0432\u0430\u0442\u044c "
        "\u0440\u0435\u0430\u043b\u044c\u043d\u043e\u0441\u0442\u044c "
        "\u0441\u0438\u0442\u0443\u0430\u0446\u0438\u0438."
    ),
    "acceptance": (
        "\u041f\u0440\u0438\u043d\u044f\u0442\u0438\u0435: "
        "\u043f\u0440\u0438\u0437\u043d\u0430\u043d\u0438\u0435 "
        "\u0442\u0435\u043a\u0443\u0449\u0435\u0439 "
        "\u0440\u0435\u0430\u043b\u044c\u043d\u043e\u0441\u0442\u0438 \u0438 "
        "\u044d\u043c\u043e\u0446\u0438\u043e\u043d\u0430\u043b\u044c\u043d\u043e\u0435 "
        "\u043f\u0440\u0438\u043c\u0438\u0440\u0435\u043d\u0438\u0435 \u0441 \u043d\u0435\u0439. "
        "\u0422\u043e\u043d \u0441\u043f\u043e\u043a\u043e\u0439\u043d\u044b\u0439 \u0438\u043b\u0438 "
        "\u0442\u0438\u0445\u043e \u0441\u043c\u0438\u0440\u0451\u043d\u043d\u044b\u0439. "
        "\u041d\u0435 \u0437\u0430\u0432\u0438\u0441\u0438\u0442 \u043e\u0442 "
        "\u043e\u0436\u0438\u0434\u0430\u043d\u0438\u044f "
        "\u0443\u043b\u0443\u0447\u0448\u0435\u043d\u0438\u044f."
    ),
}

PRIORITY_RULES_RU = [
    (
        "\u0423\u043b\u0443\u0447\u0448\u0435\u043d\u0438\u0435, \u043f\u043e\u0431\u0435\u0434\u0430, "
        "\u0431\u043e\u0440\u044c\u0431\u0430 \u0438\u043b\u0438 \u0432\u0435\u0440\u0430 \u0432 "
        "\u043f\u043e\u0437\u0438\u0442\u0438\u0432\u043d\u044b\u0439 \u0438\u0441\u0445\u043e\u0434 "
        "\u043e\u0431\u044b\u0447\u043d\u043e \u0443\u043a\u0430\u0437\u044b\u0432\u0430\u044e\u0442 \u043d\u0430 "
        "hope (\u043d\u0430\u0434\u0435\u0436\u0434\u0443)."
    ),
    (
        "\u041f\u0440\u0438\u0437\u043d\u0430\u043d\u0438\u0435 "
        "\u0440\u0435\u0430\u043b\u044c\u043d\u043e\u0441\u0442\u0438 \u0438 "
        "\u0441\u043f\u043e\u043a\u043e\u0439\u043d\u043e\u0435 "
        "\u043f\u0440\u0438\u0441\u043f\u043e\u0441\u043e\u0431\u043b\u0435\u043d\u0438\u0435 \u043a \u043d\u0435\u0439 "
        "\u043e\u0431\u044b\u0447\u043d\u043e \u0443\u043a\u0430\u0437\u044b\u0432\u0430\u044e\u0442 \u043d\u0430 "
        "acceptance (\u043f\u0440\u0438\u043d\u044f\u0442\u0438\u0435)."
    ),
    (
        "\u0415\u0441\u043b\u0438 \u043f\u0440\u0435\u043e\u0431\u043b\u0430\u0434\u0430\u044e\u0442 "
        "\u0443\u0442\u0440\u0430\u0442\u0430, \u0431\u0435\u0441\u043f\u043e\u043c\u043e\u0449\u043d\u043e\u0441\u0442\u044c, "
        "\u043d\u0435\u0441\u043f\u0440\u0430\u0432\u0435\u0434\u043b\u0438\u0432\u043e\u0441\u0442\u044c "
        "\u0438\u043b\u0438 \u0431\u043e\u043b\u044c \u2014 \u0432\u044b\u0431\u0438\u0440\u0430\u0439 "
        "sadness (\u0433\u0440\u0443\u0441\u0442\u044c), \u0434\u0430\u0436\u0435 \u0435\u0441\u043b\u0438 "
        "\u0435\u0441\u0442\u044c \u0444\u0440\u0430\u0437\u0430 \u0432\u0440\u043e\u0434\u0435 "
        "\u00ab\u043f\u0440\u043e\u0434\u043e\u043b\u0436\u0430\u044e \u0431\u043e\u0440\u043e\u0442\u044c\u0441\u044f\u00bb."
    ),
    (
        "Anger (\u0433\u043d\u0435\u0432) \u0442\u0440\u0435\u0431\u0443\u0435\u0442 "
        "\u0437\u0430\u043c\u0435\u0442\u043d\u043e\u0433\u043e "
        "\u0440\u0430\u0437\u0434\u0440\u0430\u0436\u0435\u043d\u0438\u044f, "
        "\u043e\u0431\u0432\u0438\u043d\u0435\u043d\u0438\u044f, \u043f\u0440\u043e\u0442\u0435\u0441\u0442\u0430 "
        "\u0438\u043b\u0438 \u0432\u0440\u0430\u0436\u0434\u0435\u0431\u043d\u043e\u0441\u0442\u0438, \u0430 "
        "\u043d\u0435 \u0442\u043e\u043b\u044c\u043a\u043e \u00ab\u043f\u043e\u0447\u0435\u043c\u0443 \u044f\u00bb "
        "\u0438\u043b\u0438 \u043d\u0435\u0441\u043f\u0440\u0430\u0432\u0435\u0434\u043b\u0438\u0432\u043e\u0441\u0442\u0438."
    ),
]

DISTRESS_DEFINITIONS_RU = {
    "low": (
        "\u041d\u0438\u0437\u043a\u0438\u0439: \u0441\u043b\u0430\u0431\u043e "
        "\u0432\u044b\u0440\u0430\u0436\u0435\u043d\u043d\u044b\u0439 \u0434\u0438\u0441\u0442\u0440\u0435\u0441\u0441. "
        "\u0413\u043e\u0432\u043e\u0440\u044f\u0449\u0438\u0439 \u0432 \u043e\u0441\u043d\u043e\u0432\u043d\u043e\u043c "
        "\u0441\u043f\u043e\u043a\u043e\u0435\u043d, \u0441\u043e\u0431\u0440\u0430\u043d, "
        "\u0434\u0435\u043b\u043e\u0432\u0438\u0442 \u0438\u043b\u0438 \u043b\u0438\u0448\u044c "
        "\u0441\u043b\u0435\u0433\u043a\u0430 \u0432\u0441\u0442\u0440\u0435\u0432\u043e\u0436\u0435\u043d."
    ),
    "medium": (
        "\u0421\u0440\u0435\u0434\u043d\u0438\u0439: \u044f\u0432\u043d\u044b\u0435 \u0438 "
        "\u0443\u0441\u0442\u043e\u0439\u0447\u0438\u0432\u044b\u0435 "
        "\u0431\u0435\u0441\u043f\u043e\u043a\u043e\u0439\u0441\u0442\u0432\u043e, "
        "\u0433\u0440\u0443\u0441\u0442\u044c, \u0441\u0442\u0440\u0430\u0445 \u0438\u043b\u0438 "
        "\u043d\u0430\u043f\u0440\u044f\u0436\u0435\u043d\u0438\u0435, \u043d\u043e "
        "\u0447\u0435\u043b\u043e\u0432\u0435\u043a \u0432\u0441\u0451 \u0435\u0449\u0451 "
        "\u0441\u043f\u0440\u0430\u0432\u043b\u044f\u0435\u0442\u0441\u044f. "
        "\u0414\u0438\u0441\u0442\u0440\u0435\u0441\u0441 \u0437\u043d\u0430\u0447\u0438\u043c\u044b\u0439, "
        "\u043d\u043e \u043d\u0435 "
        "\u043f\u043e\u0434\u0430\u0432\u043b\u044f\u044e\u0449\u0438\u0439."
    ),
    "high": (
        "\u0412\u044b\u0441\u043e\u043a\u0438\u0439: \u0434\u0438\u0441\u0442\u0440\u0435\u0441\u0441 "
        "\u043f\u043e\u0434\u0430\u0432\u043b\u044f\u044e\u0449\u0438\u0439, \u043e\u0441\u0442\u0440\u044b\u0439, "
        "\u043d\u0435\u043a\u043e\u043d\u0442\u0440\u043e\u043b\u0438\u0440\u0443\u0435\u043c\u044b\u0439. "
        "\u041f\u0440\u0438\u0437\u043d\u0430\u043a\u0438: \u043f\u0430\u043d\u0438\u043a\u0430, "
        "\u043e\u0442\u0447\u0430\u044f\u043d\u0438\u0435, "
        "\u043d\u0435\u0441\u043f\u043e\u0441\u043e\u0431\u043d\u043e\u0441\u0442\u044c "
        "\u0444\u0443\u043d\u043a\u0446\u0438\u043e\u043d\u0438\u0440\u043e\u0432\u0430\u0442\u044c, "
        "\u043a\u0440\u0430\u0439\u043d\u044f\u044f "
        "\u0431\u0435\u0437\u043d\u0430\u0434\u0451\u0436\u043d\u043e\u0441\u0442\u044c \u0438\u043b\u0438 "
        "\u0441\u0440\u043e\u0447\u043d\u044b\u0439 \u043f\u043e\u0438\u0441\u043a "
        "\u043f\u043e\u043c\u043e\u0449\u0438."
    ),
}

DISTRESS_NOTES_RU = [
    (
        "\u041e\u0446\u0435\u043d\u0438\u0432\u0430\u0439 \u0442\u043e\u043b\u044c\u043a\u043e "
        "\u0434\u0438\u0441\u0442\u0440\u0435\u0441\u0441, \u044f\u0432\u043d\u043e "
        "\u0432\u044b\u0440\u0430\u0436\u0435\u043d\u043d\u044b\u0439 \u0432 \u0442\u0435\u043a\u0441\u0442\u0435. "
        "\u041d\u0435 \u0432\u044b\u0432\u043e\u0434\u0438 \u0435\u0433\u043e \u0438\u0437 "
        "\u0442\u044f\u0436\u0435\u0441\u0442\u0438 \u0431\u043e\u043b\u0435\u0437\u043d\u0438 \u0438\u043b\u0438 "
        "\u043f\u0440\u043e\u0433\u043d\u043e\u0437\u0430."
    ),
    (
        "\u0421\u0435\u0440\u044c\u0451\u0437\u043d\u044b\u0439 \u0434\u0438\u0430\u0433\u043d\u043e\u0437, "
        "\u043e\u043f\u0438\u0441\u0430\u043d\u043d\u044b\u0439 \u0441\u043f\u043e\u043a\u043e\u0439\u043d\u043e, "
        "\u043d\u0435 \u044f\u0432\u043b\u044f\u0435\u0442\u0441\u044f "
        "\u0430\u0432\u0442\u043e\u043c\u0430\u0442\u0438\u0447\u0435\u0441\u043a\u0438 "
        "\u0432\u044b\u0441\u043e\u043a\u0438\u043c."
    ),
    (
        "\u041e\u0434\u043d\u043e \u0441\u0438\u043b\u044c\u043d\u043e\u0435 "
        "\u044d\u043c\u043e\u0446\u0438\u043e\u043d\u0430\u043b\u044c\u043d\u043e\u0435 \u0441\u043b\u043e\u0432\u043e "
        "\u0431\u0435\u0437 \u0434\u0440\u0443\u0433\u0438\u0445 "
        "\u043f\u0440\u0438\u0437\u043d\u0430\u043a\u043e\u0432 \u043d\u0435 "
        "\u0434\u0435\u043b\u0430\u0435\u0442 \u0434\u0438\u0441\u0442\u0440\u0435\u0441\u0441 "
        "\u0430\u0432\u0442\u043e\u043c\u0430\u0442\u0438\u0447\u0435\u0441\u043a\u0438 "
        "\u0432\u044b\u0441\u043e\u043a\u0438\u043c."
    ),
]

CONFIDENCE_DEFINITIONS_RU = {
    5: "\u041f\u0440\u0430\u043a\u0442\u0438\u0447\u0435\u0441\u043a\u0438 \u043e\u0434\u043d\u043e\u0437\u043d\u0430\u0447\u043d\u043e.",
    4: "\u0423\u0432\u0435\u0440\u0435\u043d\u043d\u043e, \u0441 \u043d\u0435\u0431\u043e\u043b\u044c\u0448\u043e\u0439 "
       "\u0432\u043e\u0437\u043c\u043e\u0436\u043d\u043e\u0439 "
       "\u0430\u043b\u044c\u0442\u0435\u0440\u043d\u0430\u0442\u0438\u0432\u043e\u0439.",
    3: "\u0421\u043a\u043e\u0440\u0435\u0435 \u0432\u0441\u0435\u0433\u043e, \u043d\u043e \u0435\u0441\u0442\u044c "
       "\u0440\u0430\u0437\u0443\u043c\u043d\u0430\u044f "
       "\u0430\u043b\u044c\u0442\u0435\u0440\u043d\u0430\u0442\u0438\u0432\u0430.",
    2: "\u0421\u0438\u043b\u044c\u043d\u0430\u044f "
       "\u043d\u0435\u043e\u043f\u0440\u0435\u0434\u0435\u043b\u0451\u043d\u043d\u043e\u0441\u0442\u044c.",
    1: "\u041d\u0435\u0432\u043e\u0437\u043c\u043e\u0436\u043d\u043e \u043d\u0430\u0434\u0451\u0436\u043d\u043e "
       "\u043a\u043b\u0430\u0441\u0441\u0438\u0444\u0438\u0446\u0438\u0440\u043e\u0432\u0430\u0442\u044c.",
}


# =============================================================================
# Exceptions
# =============================================================================

class QuitAnnotation(Exception):
    """Raised when the annotator requests a safe exit."""


# =============================================================================
# File helpers
# =============================================================================

def load_jsonl(path):
    # type: (Path) -> List[Dict[str, Any]]
    records = []

    if not path.exists():
        raise FileNotFoundError("File not found: {}".format(path))

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    "Invalid JSON in {}, line {}: {}".format(
                        path,
                        line_number,
                        error,
                    )
                )

            if not isinstance(record, dict):
                raise ValueError(
                    "Expected JSON object in {}, line {}".format(
                        path,
                        line_number,
                    )
                )

            records.append(record)

    return records


def append_jsonl(path, record):
    # type: (Path, Dict[str, Any]) -> None
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


def write_jsonl(path, records):
    # type: (Path, List[Dict[str, Any]]) -> None
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def write_json(path, value):
    # type: (Path, Dict[str, Any]) -> None
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            value,
            file,
            ensure_ascii=False,
            indent=2,
        )


def sha256_file(path):
    # type: (Path) -> str
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def utc_now():
    # type: () -> str
    return datetime.now(timezone.utc).isoformat()


# =============================================================================
# Test record helpers
# =============================================================================

def get_item_id(record, index):
    # type: (Dict[str, Any], int) -> str
    value = record.get("id")

    if isinstance(value, str) and value.strip():
        return value.strip()

    value = record.get("item_id")

    if isinstance(value, str) and value.strip():
        return value.strip()

    return "test_item_{:04d}".format(index)


def get_text(record):
    # type: (Dict[str, Any]) -> str
    for field in ("text", "message", "content"):
        value = record.get(field)

        if isinstance(value, str) and value.strip():
            return value.strip()

    raise ValueError(
        "A test record does not contain a non-empty "
        "'text', 'message', or 'content' field."
    )


def get_strict_response(record):
    # type: (Dict[str, Any]) -> str
    for field in (
        "final_response",
        "intended_response",
        "response",
    ):
        value = record.get(field)

        if isinstance(value, str):
            normalized = value.strip().lower()

            if normalized in RESPONSE_LABELS:
                return normalized

    raise ValueError(
        "Could not find a valid strict response label in item {}".format(
            record.get("id", "<unknown>")
        )
    )


def get_strict_distress(record):
    # type: (Dict[str, Any]) -> str
    for field in (
        "final_distress",
        "intended_distress",
        "distress",
    ):
        value = record.get(field)

        if isinstance(value, str):
            normalized = value.strip().lower()

            if normalized in DISTRESS_LABELS:
                return normalized

    raise ValueError(
        "Could not find a valid strict distress label in item {}".format(
            record.get("id", "<unknown>")
        )
    )


def build_test_index(records):
    # type: (List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]
    result = {}

    for position, record in enumerate(records, start=1):
        item_id = get_item_id(record, position)

        if item_id in result:
            raise ValueError(
                "Duplicate test item ID: {}".format(item_id)
            )

        result[item_id] = record

    return result


# =============================================================================
# Rubric display
# =============================================================================

def print_separator():
    # type: () -> None
    print("=" * 78)


def print_rubric():
    # type: () -> None
    print_separator()
    print("FROZEN HUMAN ANNOTATION RUBRIC")
    print_separator()

    print("\nRESPONSE - choose exactly one dominant response:\n")

    for label in RESPONSE_LABELS:
        print(
            "{:<12s} {}".format(
                label.upper(),
                RESPONSE_DEFINITIONS[label],
            )
        )
        print(
            "             RU: {}".format(
                RESPONSE_DEFINITIONS_RU[label]
            )
        )

    print("\nPriority rules:")

    for index, rule in enumerate(PRIORITY_RULES):
        print("  - {}".format(rule))
        print("    RU: {}".format(PRIORITY_RULES_RU[index]))

    print("\nDISTRESS - judge only explicitly expressed distress:\n")

    for label in DISTRESS_LABELS:
        print(
            "{:<8s} {}".format(
                label.upper(),
                DISTRESS_DEFINITIONS[label],
            )
        )
        print(
            "         RU: {}".format(
                DISTRESS_DEFINITIONS_RU[label]
            )
        )

    print("\nImportant distress rules:")

    for index, note in enumerate(DISTRESS_NOTES):
        print("  - {}".format(note))
        print("    RU: {}".format(DISTRESS_NOTES_RU[index]))

    print("\nCONFIDENCE:\n")

    for value in range(5, 0, -1):
        print(
            "  {}: {}".format(
                value,
                CONFIDENCE_DEFINITIONS[value],
            )
        )
        print(
            "     RU: {}".format(
                CONFIDENCE_DEFINITIONS_RU[value]
            )
        )

    print("\nCommands:")
    print("  /r  show this rubric again")
    print("  /q  save progress and quit")
    print_separator()


# =============================================================================
# Console input helpers
# =============================================================================

def read_commandable_input(prompt):
    # type: (str) -> str
    value = input(prompt).strip()

    if value.lower() == "/q":
        raise QuitAnnotation()

    if value.lower() == "/r":
        print_rubric()
        return read_commandable_input(prompt)

    return value


def prompt_choice(prompt, choices):
    # type: (str, List[str]) -> str
    allowed = set(choices)

    while True:
        value = read_commandable_input(prompt).lower()

        if value in allowed:
            return value

        print(
            "Invalid value. Choose one of: {}".format(
                ", ".join(choices)
            )
        )


def prompt_numeric_choice(prompt, choices):
    # type: (str, List[str]) -> str
    """
    Select a label by typing its NUMBER (1-based).
    Returns the chosen label as a word (for the JSONL file).
    """
    while True:
        value = read_commandable_input(prompt)

        try:
            number = int(value)
        except ValueError:
            print(
                "Enter a number from 1 to {}.".format(
                    len(choices)
                )
            )
            continue

        if 1 <= number <= len(choices):
            return choices[number - 1]

        print(
            "Enter a number from 1 to {}.".format(
                len(choices)
            )
        )


def prompt_confidence(prompt):
    # type: (str) -> int
    while True:
        value = read_commandable_input(prompt)

        try:
            confidence = int(value)
        except ValueError:
            print("Enter an integer from 1 to 5.")
            continue

        if 1 <= confidence <= 5:
            return confidence

        print("Enter an integer from 1 to 5.")


def prompt_boolean(prompt):
    # type: (str) -> bool
    while True:
        value = read_commandable_input(prompt).lower()

        if value in ("1", "y", "yes", "true"):
            return True

        if value in ("2", "n", "no", "false"):
            return False

        print("Enter 1 for yes or 2 for no.")


def prompt_optional_secondary(primary):
    # type: (str) -> Optional[str]
    while True:
        value = read_commandable_input(
            "Optional secondary response "
            "(\u0432\u0442\u043e\u0440\u0438\u0447\u043d\u0430\u044f "
            "\u0440\u0435\u0430\u043a\u0446\u0438\u044f / "
            "\u05ea\u05d2\u05d5\u05d1\u05d4 \u05de\u05e9\u05e0\u05d9\u05ea - "
            "0=none/\u043d\u0435\u0442) "
            "[1=anxiety 2=sadness 3=anger 4=hope "
            "5=guilt 6=denial 7=acceptance]: "
        )

        if not value or value in ("0", "none", "-", "na", "n/a"):
            return None

        try:
            number = int(value)
        except ValueError:
            print(
                "Enter a number 1-7, or 0 for none."
            )
            continue

        if not (1 <= number <= len(RESPONSE_LABELS)):
            print(
                "Enter a number 1-7, or 0 for none."
            )
            continue

        secondary = RESPONSE_LABELS[number - 1]

        if secondary == primary:
            print(
                "Secondary response must differ "
                "from the primary response."
            )
            continue

        return secondary


def prompt_optional_note():
    # type: () -> Optional[str]
    value = read_commandable_input(
        "Optional short note explaining uncertainty "
        "(\u043a\u043e\u0440\u043e\u0442\u043a\u0430\u044f \u0437\u0430\u043c\u0435\u0442\u043a\u0430 / "
        "\u05d4\u05e2\u05e8\u05d4 \u05e7\u05e6\u05e8\u05d4 - "
        "blank for none): "
    )

    if value:
        return value

    return None


# =============================================================================
# Annotation checkpoint helpers
# =============================================================================

def load_latest_annotations(path):
    # type: (Path) -> Dict[str, Dict[str, Any]]
    """
    Load annotations using last-write-wins.

    A corrected annotation may be appended later without editing the original
    JSONL history.
    """
    latest = {}

    if not path.exists():
        return latest

    for record in load_jsonl(path):
        item_id = record.get("item_id")

        if isinstance(item_id, str) and item_id:
            latest[item_id] = record

    return latest


def validate_annotation_file_hash(annotations, expected_hash):
    # type: (Dict[str, Dict[str, Any]], str) -> None
    observed_hashes = set()

    for record in annotations.values():
        value = record.get("test_file_sha256")

        if value:
            observed_hashes.add(value)

    if not observed_hashes:
        return

    if observed_hashes != {expected_hash}:
        raise ValueError(
            "The annotation file belongs to a different test file. "
            "Expected SHA256: {}; observed: {}".format(
                expected_hash,
                sorted(observed_hashes),
            )
        )


# =============================================================================
# Annotation mode
# =============================================================================

def annotate_one_item(
    item_id,
    text,
    annotator_id,
    item_position,
    total_items,
    test_hash,
    rubric_version,
    seed,
):
    # type: (
    #     str,
    #     str,
    #     str,
    #     int,
    #     int,
    #     str,
    #     str,
    #     int,
    # ) -> Dict[str, Any]

    while True:
        print()
        print_separator()
        print(
            "ITEM {}/{}".format(
                item_position,
                total_items,
            )
        )
        print("Anonymous ID: {}".format(item_id))
        print_separator()
        print(text)
        print_separator()

        started = time.monotonic()

        human_response = prompt_numeric_choice(
            "Dominant response "
            "(\u0434\u043e\u043c\u0438\u043d\u0438\u0440\u0443\u044e\u0449\u0430\u044f "
            "\u0440\u0435\u0430\u043a\u0446\u0438\u044f / "
            "\u05ea\u05d2\u05d5\u05d1\u05d4 \u05e2\u05d9\u05e7\u05e8\u05d9\u05ea)\n"
            "  [1=anxiety 2=sadness 3=anger 4=hope "
            "5=guilt 6=denial 7=acceptance]: ",
            RESPONSE_LABELS,
        )

        human_response_confidence = prompt_confidence(
            "Response confidence "
            "(\u0443\u0432\u0435\u0440\u0435\u043d\u043d\u043e\u0441\u0442\u044c "
            "\u0432 \u0440\u0435\u0430\u043a\u0446\u0438\u0438 / "
            "\u05e8\u05de\u05ea \u05d1\u05d9\u05d8\u05d7\u05d5\u05df)\n"
            "  [5=\u0442\u043e\u0447\u043d\u043e 4=\u0443\u0432\u0435\u0440\u0435\u043d "
            "3=\u0441\u043a\u043e\u0440\u0435\u0435 2=\u0441\u043e\u043c\u043d\u0435\u0432\u0430\u044e\u0441\u044c "
            "1=\u043d\u0430\u0443\u0433\u0430\u0434]: "
        )

        human_response_ambiguous = prompt_boolean(
            "Is the response genuinely ambiguous? "
            "(\u0440\u0435\u0430\u043a\u0446\u0438\u044f "
            "\u043d\u0435\u043e\u0434\u043d\u043e\u0437\u043d\u0430\u0447\u043d\u0430? / "
            "\u05d4\u05d0\u05dd \u05d4\u05ea\u05d2\u05d5\u05d1\u05d4 "
            "\u05e2\u05de\u05d5\u05de\u05d4?) "
            "[1=\u0434\u0430/yes 2=\u043d\u0435\u0442/no]: "
        )

        human_secondary_response = prompt_optional_secondary(
            human_response
        )

        human_distress = prompt_numeric_choice(
            "Expressed distress "
            "(\u0443\u0440\u043e\u0432\u0435\u043d\u044c "
            "\u0434\u0438\u0441\u0442\u0440\u0435\u0441\u0441\u0430 / "
            "\u05e8\u05de\u05ea \u05de\u05e6\u05d5\u05e7\u05d4)\n"
            "  [1=low/\u043d\u0438\u0437\u043a\u0438\u0439 "
            "2=medium/\u0441\u0440\u0435\u0434\u043d\u0438\u0439 "
            "3=high/\u0432\u044b\u0441\u043e\u043a\u0438\u0439]: ",
            DISTRESS_LABELS,
        )

        human_distress_confidence = prompt_confidence(
            "Distress confidence "
            "(\u0443\u0432\u0435\u0440\u0435\u043d\u043d\u043e\u0441\u0442\u044c "
            "\u0432 \u0434\u0438\u0441\u0442\u0440\u0435\u0441\u0441\u0435 / "
            "\u05e8\u05de\u05ea \u05d1\u05d9\u05d8\u05d7\u05d5\u05df)\n"
            "  [5=\u0442\u043e\u0447\u043d\u043e 4=\u0443\u0432\u0435\u0440\u0435\u043d "
            "3=\u0441\u043a\u043e\u0440\u0435\u0435 2=\u0441\u043e\u043c\u043d\u0435\u0432\u0430\u044e\u0441\u044c "
            "1=\u043d\u0430\u0443\u0433\u0430\u0434]: "
        )

        human_distress_ambiguous = prompt_boolean(
            "Is the distress level genuinely ambiguous? "
            "(\u0443\u0440\u043e\u0432\u0435\u043d\u044c "
            "\u0434\u0438\u0441\u0442\u0440\u0435\u0441\u0441\u0430 "
            "\u043d\u0435\u043e\u0434\u043d\u043e\u0437\u043d\u0430\u0447\u0435\u043d? / "
            "\u05d4\u05d0\u05dd \u05e8\u05de\u05ea \u05d4\u05de\u05e6\u05d5\u05e7 "
            "\u05e2\u05de\u05d5\u05dd?) "
            "[1=\u0434\u0430/yes 2=\u043d\u0435\u0442/no]: "
        )

        human_note = prompt_optional_note()

        elapsed_seconds = round(
            time.monotonic() - started,
            2,
        )

        print()
        print("Your annotation:")
        print(
            "  response:             {}".format(
                human_response
            )
        )
        print(
            "  response confidence:  {}".format(
                human_response_confidence
            )
        )
        print(
            "  response ambiguous:   {}".format(
                human_response_ambiguous
            )
        )
        print(
            "  secondary response:   {}".format(
                human_secondary_response
            )
        )
        print(
            "  distress:             {}".format(
                human_distress
            )
        )
        print(
            "  distress confidence:  {}".format(
                human_distress_confidence
            )
        )
        print(
            "  distress ambiguous:   {}".format(
                human_distress_ambiguous
            )
        )
        print(
            "  note:                 {}".format(
                human_note
            )
        )

        confirmed = prompt_boolean(
            "Save this annotation? "
            "(\u0441\u043e\u0445\u0440\u0430\u043d\u0438\u0442\u044c? / "
            "\u05dc\u05e9\u05de\u05d5\u05e8?) "
            "[1=\u0434\u0430/yes 2=\u043d\u0435\u0442/no]: "
        )

        if not confirmed:
            print(
                "Annotation discarded. Re-entering the item."
            )
            continue

        return {
            "item_id": item_id,
            "annotator_id": annotator_id,
            "rubric_version": rubric_version,
            "annotation_seed": seed,
            "annotation_position": item_position,
            "human_response": human_response,
            "human_response_confidence": (
                human_response_confidence
            ),
            "human_response_ambiguous": (
                human_response_ambiguous
            ),
            "human_secondary_response": (
                human_secondary_response
            ),
            "human_distress": human_distress,
            "human_distress_confidence": (
                human_distress_confidence
            ),
            "human_distress_ambiguous": (
                human_distress_ambiguous
            ),
            "human_note": human_note,
            "annotation_seconds": elapsed_seconds,
            "annotated_at_utc": utc_now(),
            "test_file_sha256": test_hash,
        }


def safe_filename(value):
    # type: (str) -> str
    output = []

    for character in value:
        if character.isalnum() or character in ("-", "_"):
            output.append(character)
        else:
            output.append("_")

    result = "".join(output).strip("_")

    if result:
        return result

    return "annotator"


def run_annotation(args):
    # type: (argparse.Namespace) -> None
    input_path = Path(args.input)
    annotator_id = args.annotator.strip()

    if not annotator_id:
        raise SystemExit("--annotator cannot be empty.")

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_name(
            "human_annotations_{}.jsonl".format(
                safe_filename(annotator_id)
            )
        )

    records = load_jsonl(input_path)
    test_index = build_test_index(records)
    test_hash = sha256_file(input_path)

    ordered_items = sorted(
        test_index.items(),
        key=lambda pair: pair[0],
    )

    rng = random.Random(args.seed)
    rng.shuffle(ordered_items)

    item_positions = {}

    for position, pair in enumerate(
        ordered_items,
        start=1,
    ):
        item_positions[pair[0]] = position

    existing = load_latest_annotations(output_path)

    validate_annotation_file_hash(
        existing,
        test_hash,
    )

    unknown_ids = set(existing.keys()) - set(test_index.keys())

    if unknown_ids:
        raise SystemExit(
            "The annotation file contains IDs not found in "
            "the current test set: {}".format(
                sorted(unknown_ids)[:5]
            )
        )

    completed = set(existing.keys())

    remaining = []

    for pair in ordered_items:
        if pair[0] not in completed:
            remaining.append(pair)

    print_separator()
    print("BLIND HUMAN TEST ANNOTATION")
    print_separator()
    print("Input:          {}".format(input_path))
    print("Output:         {}".format(output_path))
    print("Annotator:      {}".format(annotator_id))
    print(
        "Rubric version: {}".format(
            args.rubric_version
        )
    )
    print("Random seed:    {}".format(args.seed))
    print("Test SHA256:    {}".format(test_hash))
    print("Total items:    {}".format(len(records)))
    print("Completed:      {}".format(len(completed)))
    print("Remaining:      {}".format(len(remaining)))
    print()
    print(
        "Strict labels and judge labels will not be displayed. "
        "No agreement feedback will be provided."
    )

    print_rubric()

    if not remaining:
        print("All items are already annotated.")
        print(
            "You may now run the separate score command."
        )
        return

    try:
        for item_id, record in remaining:
            annotation = annotate_one_item(
                item_id=item_id,
                text=get_text(record),
                annotator_id=annotator_id,
                item_position=item_positions[item_id],
                total_items=len(ordered_items),
                test_hash=test_hash,
                rubric_version=args.rubric_version,
                seed=args.seed,
            )

            append_jsonl(
                output_path,
                annotation,
            )

            completed.add(item_id)

            print(
                "Saved. Progress: {}/{}".format(
                    len(completed),
                    len(records),
                )
            )

    except QuitAnnotation:
        print()
        print("Annotation stopped safely.")
        print(
            "Saved progress: {}/{} items.".format(
                len(completed),
                len(records),
            )
        )
        print(
            "Resume by running the same annotate command."
        )
        return

    print()
    print_separator()
    print("ANNOTATION COMPLETE")
    print_separator()
    print(
        "Annotated items: {}/{}".format(
            len(completed),
            len(records),
        )
    )
    print("Annotations:     {}".format(output_path))
    print()
    print(
        "Run the separate score command when ready."
    )


# =============================================================================
# Metric helpers
# =============================================================================

def safe_divide(numerator, denominator):
    # type: (float, float) -> float
    if denominator:
        return numerator / denominator

    return 0.0


def calculate_mean(values):
    # type: (List[float]) -> Optional[float]
    if not values:
        return None

    return sum(values) / len(values)


def require_sklearn():
    try:
        from sklearn.metrics import (
            cohen_kappa_score,
            confusion_matrix,
        )
    except ImportError as error:
        raise SystemExit(
            "Scoring requires scikit-learn.\n"
            "Install it with:\n"
            "python -m pip install scikit-learn"
        ) from error

    return cohen_kappa_score, confusion_matrix


def safe_nominal_kappa(labels_a, labels_b):
    # type: (List[str], List[str]) -> Optional[float]
    cohen_kappa_score, _ = require_sklearn()

    if not labels_a:
        return None

    if len(labels_a) != len(labels_b):
        return None

    try:
        value = cohen_kappa_score(
            labels_a,
            labels_b,
        )
    except (ValueError, ZeroDivisionError):
        return None

    if value != value:
        return None

    return float(value)


def safe_distress_kappa(
    labels_a,
    labels_b,
    weights,
):
    # type: (List[str], List[str], str) -> Optional[float]
    """
    Weighted distress kappa using the correct ordinal order:
    low=0, medium=1, high=2.
    """
    cohen_kappa_score, _ = require_sklearn()

    if not labels_a:
        return None

    if len(labels_a) != len(labels_b):
        return None

    numeric_a = [
        DISTRESS_TO_INT[label]
        for label in labels_a
    ]

    numeric_b = [
        DISTRESS_TO_INT[label]
        for label in labels_b
    ]

    try:
        value = cohen_kappa_score(
            numeric_a,
            numeric_b,
            labels=[0, 1, 2],
            weights=weights,
        )
    except (ValueError, ZeroDivisionError):
        return None

    if value != value:
        return None

    return float(value)


def calculate_confusion(
    true_labels,
    predicted_labels,
    label_order,
):
    # type: (List[str], List[str], List[str]) -> List[List[int]]
    _, confusion_matrix = require_sklearn()

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=label_order,
    )

    return matrix.astype(int).tolist()


def calculate_per_class_agreement(
    strict_labels,
    human_labels,
    label_order,
):
    # type: (
    #     List[str],
    #     List[str],
    #     List[str],
    # ) -> Dict[str, Dict[str, Any]]

    result = {}

    for label in label_order:
        positions = []

        for index, strict_label in enumerate(strict_labels):
            if strict_label == label:
                positions.append(index)

        agreements = 0

        for index in positions:
            if human_labels[index] == label:
                agreements += 1

        support = len(positions)

        result[label] = {
            "support": support,
            "agreements": agreements,
            "agreement_rate": safe_divide(
                agreements,
                support,
            ),
        }

    return result


# =============================================================================
# Human versus strict report
# =============================================================================

def calculate_human_strict_report(
    test_records,
    annotations,
):
    # type: (
    #     List[Dict[str, Any]],
    #     Dict[str, Dict[str, Any]],
    # ) -> Dict[str, Any]

    response_strict = []
    response_human = []

    distress_strict = []
    distress_human = []

    response_confidences = []
    distress_confidences = []

    response_ambiguous_count = 0
    distress_ambiguous_count = 0

    response_confirmed_ids = []
    distress_confirmed_ids = []
    joint_confirmed_ids = []

    response_only_agreement = 0
    distress_only_agreement = 0
    neither_agreement = 0

    annotation_times = []

    for index, record in enumerate(
        test_records,
        start=1,
    ):
        item_id = get_item_id(record, index)
        annotation = annotations[item_id]

        strict_response = get_strict_response(record)
        strict_distress = get_strict_distress(record)

        human_response = annotation["human_response"]
        human_distress = annotation["human_distress"]

        response_strict.append(strict_response)
        response_human.append(human_response)

        distress_strict.append(strict_distress)
        distress_human.append(human_distress)

        response_confidences.append(
            float(
                annotation[
                    "human_response_confidence"
                ]
            )
        )

        distress_confidences.append(
            float(
                annotation[
                    "human_distress_confidence"
                ]
            )
        )

        if annotation["human_response_ambiguous"]:
            response_ambiguous_count += 1

        if annotation["human_distress_ambiguous"]:
            distress_ambiguous_count += 1

        response_agrees = (
            human_response == strict_response
        )

        distress_agrees = (
            human_distress == strict_distress
        )

        if response_agrees:
            response_confirmed_ids.append(item_id)

        if distress_agrees:
            distress_confirmed_ids.append(item_id)

        if response_agrees and distress_agrees:
            joint_confirmed_ids.append(item_id)
        elif response_agrees:
            response_only_agreement += 1
        elif distress_agrees:
            distress_only_agreement += 1
        else:
            neither_agreement += 1

        seconds = annotation.get("annotation_seconds")

        if isinstance(seconds, (int, float)):
            annotation_times.append(float(seconds))

    total = len(test_records)

    response_exact = 0

    for strict_label, human_label in zip(
        response_strict,
        response_human,
    ):
        if strict_label == human_label:
            response_exact += 1

    distress_exact = 0

    for strict_label, human_label in zip(
        distress_strict,
        distress_human,
    ):
        if strict_label == human_label:
            distress_exact += 1

    distress_distances = []

    for strict_label, human_label in zip(
        distress_strict,
        distress_human,
    ):
        distance = abs(
            DISTRESS_TO_INT[strict_label]
            - DISTRESS_TO_INT[human_label]
        )

        distress_distances.append(distance)

    distress_within_one = 0

    for distance in distress_distances:
        if distance <= 1:
            distress_within_one += 1

    return {
        "n": total,
        "response": {
            "exact_agreement_count": response_exact,
            "exact_agreement": safe_divide(
                response_exact,
                total,
            ),
            "cohen_kappa": safe_nominal_kappa(
                response_strict,
                response_human,
            ),
            "ambiguous_count": response_ambiguous_count,
            "ambiguous_rate": safe_divide(
                response_ambiguous_count,
                total,
            ),
            "mean_human_confidence": calculate_mean(
                response_confidences
            ),
            "per_class_agreement": (
                calculate_per_class_agreement(
                    response_strict,
                    response_human,
                    RESPONSE_LABELS,
                )
            ),
            "confusion_matrix": {
                "rows_strict": RESPONSE_LABELS,
                "columns_human": RESPONSE_LABELS,
                "values": calculate_confusion(
                    response_strict,
                    response_human,
                    RESPONSE_LABELS,
                ),
            },
            "human_confirmed_item_ids": (
                response_confirmed_ids
            ),
        },
        "distress": {
            "exact_agreement_count": distress_exact,
            "exact_agreement": safe_divide(
                distress_exact,
                total,
            ),
            "weighted_kappa_linear": (
                safe_distress_kappa(
                    distress_strict,
                    distress_human,
                    "linear",
                )
            ),
            "weighted_kappa_quadratic": (
                safe_distress_kappa(
                    distress_strict,
                    distress_human,
                    "quadratic",
                )
            ),
            "mean_absolute_ordinal_difference": (
                calculate_mean(
                    [
                        float(value)
                        for value in distress_distances
                    ]
                )
            ),
            "within_one_level_count": (
                distress_within_one
            ),
            "within_one_level_agreement": safe_divide(
                distress_within_one,
                total,
            ),
            "ambiguous_count": (
                distress_ambiguous_count
            ),
            "ambiguous_rate": safe_divide(
                distress_ambiguous_count,
                total,
            ),
            "mean_human_confidence": calculate_mean(
                distress_confidences
            ),
            "per_class_agreement": (
                calculate_per_class_agreement(
                    distress_strict,
                    distress_human,
                    DISTRESS_LABELS,
                )
            ),
            "confusion_matrix": {
                "rows_strict": DISTRESS_LABELS,
                "columns_human": DISTRESS_LABELS,
                "values": calculate_confusion(
                    distress_strict,
                    distress_human,
                    DISTRESS_LABELS,
                ),
            },
            "human_confirmed_item_ids": (
                distress_confirmed_ids
            ),
        },
        "joint": {
            "joint_exact_count": len(
                joint_confirmed_ids
            ),
            "joint_exact_agreement": safe_divide(
                len(joint_confirmed_ids),
                total,
            ),
            "response_only_agreement_count": (
                response_only_agreement
            ),
            "distress_only_agreement_count": (
                distress_only_agreement
            ),
            "neither_agreement_count": (
                neither_agreement
            ),
            "human_confirmed_item_ids": (
                joint_confirmed_ids
            ),
        },
        "annotation_process": {
            "mean_seconds_per_item": calculate_mean(
                annotation_times
            ),
            "total_recorded_annotation_seconds": sum(
                annotation_times
            ),
        },
    }


# =============================================================================
# Human versus human report
# =============================================================================

def calculate_human_human_report(
    annotations_a,
    annotations_b,
):
    # type: (
    #     Dict[str, Dict[str, Any]],
    #     Dict[str, Dict[str, Any]],
    # ) -> Dict[str, Any]

    overlap_ids = sorted(
        set(annotations_a.keys())
        & set(annotations_b.keys())
    )

    response_a = []
    response_b = []

    distress_a = []
    distress_b = []

    for item_id in overlap_ids:
        response_a.append(
            annotations_a[item_id]["human_response"]
        )
        response_b.append(
            annotations_b[item_id]["human_response"]
        )
        distress_a.append(
            annotations_a[item_id]["human_distress"]
        )
        distress_b.append(
            annotations_b[item_id]["human_distress"]
        )

    response_agreements = 0
    distress_agreements = 0
    joint_agreements = 0

    for index in range(len(overlap_ids)):
        response_agrees = (
            response_a[index] == response_b[index]
        )

        distress_agrees = (
            distress_a[index] == distress_b[index]
        )

        if response_agrees:
            response_agreements += 1

        if distress_agrees:
            distress_agreements += 1

        if response_agrees and distress_agrees:
            joint_agreements += 1

    total = len(overlap_ids)

    return {
        "overlap_n": total,
        "overlap_item_ids": overlap_ids,
        "response": {
            "exact_agreement_count": (
                response_agreements
            ),
            "exact_agreement": safe_divide(
                response_agreements,
                total,
            ),
            "cohen_kappa": safe_nominal_kappa(
                response_a,
                response_b,
            ),
        },
        "distress": {
            "exact_agreement_count": (
                distress_agreements
            ),
            "exact_agreement": safe_divide(
                distress_agreements,
                total,
            ),
            "weighted_kappa_linear": (
                safe_distress_kappa(
                    distress_a,
                    distress_b,
                    "linear",
                )
            ),
            "weighted_kappa_quadratic": (
                safe_distress_kappa(
                    distress_a,
                    distress_b,
                    "quadratic",
                )
            ),
        },
        "joint": {
            "exact_agreement_count": joint_agreements,
            "exact_agreement": safe_divide(
                joint_agreements,
                total,
            ),
        },
    }


# =============================================================================
# Merge annotations into test records
# =============================================================================

def merge_annotations(test_records, annotations):
    # type: (
    #     List[Dict[str, Any]],
    #     Dict[str, Dict[str, Any]],
    # ) -> List[Dict[str, Any]]

    merged = []

    for index, original in enumerate(
        test_records,
        start=1,
    ):
        item_id = get_item_id(original, index)
        record = dict(original)
        annotation = annotations.get(item_id)

        if annotation is None:
            record[
                "human_annotation_status"
            ] = "missing"
            merged.append(record)
            continue

        strict_response = get_strict_response(original)
        strict_distress = get_strict_distress(original)

        response_agrees = (
            annotation["human_response"]
            == strict_response
        )

        distress_agrees = (
            annotation["human_distress"]
            == strict_distress
        )

        record.update(
            {
                "human_annotation_status": "complete",
                "human_annotator_id": annotation[
                    "annotator_id"
                ],
                "human_rubric_version": annotation[
                    "rubric_version"
                ],
                "human_response": annotation[
                    "human_response"
                ],
                "human_response_confidence": annotation[
                    "human_response_confidence"
                ],
                "human_response_ambiguous": annotation[
                    "human_response_ambiguous"
                ],
                "human_secondary_response": (
                    annotation.get(
                        "human_secondary_response"
                    )
                ),
                "human_distress": annotation[
                    "human_distress"
                ],
                "human_distress_confidence": annotation[
                    "human_distress_confidence"
                ],
                "human_distress_ambiguous": annotation[
                    "human_distress_ambiguous"
                ],
                "human_note": annotation.get(
                    "human_note"
                ),
                "human_response_agrees": (
                    response_agrees
                ),
                "human_distress_agrees": (
                    distress_agrees
                ),
                "human_joint_agrees": (
                    response_agrees
                    and distress_agrees
                ),
            }
        )

        merged.append(record)

    return merged


def write_confirmed_subsets(
    output_dir,
    merged_records,
):
    # type: (Path, List[Dict[str, Any]]) -> None

    response_confirmed = []
    distress_confirmed = []
    joint_confirmed = []

    for record in merged_records:
        if record.get("human_response_agrees") is True:
            response_confirmed.append(record)

        if record.get("human_distress_agrees") is True:
            distress_confirmed.append(record)

        if record.get("human_joint_agrees") is True:
            joint_confirmed.append(record)

    write_jsonl(
        output_dir
        / "test_human_confirmed_response.jsonl",
        response_confirmed,
    )

    write_jsonl(
        output_dir
        / "test_human_confirmed_distress.jsonl",
        distress_confirmed,
    )

    write_jsonl(
        output_dir
        / "test_human_confirmed_joint.jsonl",
        joint_confirmed,
    )


# =============================================================================
# Console report
# =============================================================================

def format_float(value):
    # type: (Any) -> str
    if value is None:
        return "N/A"

    if isinstance(value, float):
        return "{:.4f}".format(value)

    return str(value)


def print_matrix(title, labels, values):
    # type: (str, List[str], List[List[int]]) -> None
    print()
    print(title)

    width = max(
        max(len(label) for label in labels),
        8,
    )

    header = " " * (width + 2)

    for label in labels:
        header += "{:>9s}".format(label[:8])

    print(header)

    for row_label, row in zip(labels, values):
        line = "{:<{width}s}  ".format(
            row_label,
            width=width,
        )

        for value in row:
            line += "{:9d}".format(value)

        print(line)


def print_scoring_summary(report):
    # type: (Dict[str, Any]) -> None
    human_vs_strict = report["human_vs_strict"]

    response = human_vs_strict["response"]
    distress = human_vs_strict["distress"]
    joint = human_vs_strict["joint"]
    total = human_vs_strict["n"]

    print_separator()
    print("HUMAN CHECK RESULTS")
    print_separator()
    print("N: {}".format(total))

    print("\nResponse:")
    print(
        "  exact agreement: {}/{} ({:.1%})".format(
            response["exact_agreement_count"],
            total,
            response["exact_agreement"],
        )
    )
    print(
        "  Cohen's kappa:   {}".format(
            format_float(response["cohen_kappa"])
        )
    )
    print(
        "  ambiguous:       {} ({:.1%})".format(
            response["ambiguous_count"],
            response["ambiguous_rate"],
        )
    )
    print(
        "  mean confidence: {}".format(
            format_float(
                response["mean_human_confidence"]
            )
        )
    )

    print("\nResponse agreement by strict class:")

    for label in RESPONSE_LABELS:
        values = response[
            "per_class_agreement"
        ][label]

        print(
            "  {:11s} {:2d}/{:2d} ({:.1%})".format(
                label,
                values["agreements"],
                values["support"],
                values["agreement_rate"],
            )
        )

    print("\nDistress:")
    print(
        "  exact agreement: {}/{} ({:.1%})".format(
            distress["exact_agreement_count"],
            total,
            distress["exact_agreement"],
        )
    )
    print(
        "  weighted kappa, linear:    {}".format(
            format_float(
                distress["weighted_kappa_linear"]
            )
        )
    )
    print(
        "  weighted kappa, quadratic: {}".format(
            format_float(
                distress["weighted_kappa_quadratic"]
            )
        )
    )
    print(
        "  mean absolute difference:  {}".format(
            format_float(
                distress[
                    "mean_absolute_ordinal_difference"
                ]
            )
        )
    )
    print(
        "  ambiguous:                 {} ({:.1%})".format(
            distress["ambiguous_count"],
            distress["ambiguous_rate"],
        )
    )
    print(
        "  mean confidence:           {}".format(
            format_float(
                distress["mean_human_confidence"]
            )
        )
    )

    print("\nJoint response + distress:")
    print(
        "  joint exact agreement: {}/{} ({:.1%})".format(
            joint["joint_exact_count"],
            total,
            joint["joint_exact_agreement"],
        )
    )
    print(
        "  response only agrees:  {}".format(
            joint["response_only_agreement_count"]
        )
    )
    print(
        "  distress only agrees:  {}".format(
            joint["distress_only_agreement_count"]
        )
    )
    print(
        "  neither agrees:        {}".format(
            joint["neither_agreement_count"]
        )
    )

    response_matrix = response["confusion_matrix"]

    print_matrix(
        "Response confusion matrix "
        "(rows=strict, columns=human)",
        response_matrix["rows_strict"],
        response_matrix["values"],
    )

    distress_matrix = distress["confusion_matrix"]

    print_matrix(
        "Distress confusion matrix "
        "(rows=strict, columns=human)",
        distress_matrix["rows_strict"],
        distress_matrix["values"],
    )

    human_vs_human = report.get("human_vs_human")

    if human_vs_human:
        print()
        print("Human-human overlap:")
        print(
            "  overlap N: {}".format(
                human_vs_human["overlap_n"]
            )
        )
        print(
            "  response exact: {:.1%}".format(
                human_vs_human[
                    "response"
                ]["exact_agreement"]
            )
        )
        print(
            "  response kappa: {}".format(
                format_float(
                    human_vs_human[
                        "response"
                    ]["cohen_kappa"]
                )
            )
        )
        print(
            "  distress exact: {:.1%}".format(
                human_vs_human[
                    "distress"
                ]["exact_agreement"]
            )
        )
        print(
            "  distress weighted kappa: {}".format(
                format_float(
                    human_vs_human[
                        "distress"
                    ]["weighted_kappa_linear"]
                )
            )
        )


# =============================================================================
# Scoring mode
# =============================================================================

def run_scoring(args):
    # type: (argparse.Namespace) -> None
    input_path = Path(args.input)
    annotations_path = Path(args.annotations)

    test_records = load_jsonl(input_path)
    test_index = build_test_index(test_records)
    test_hash = sha256_file(input_path)

    annotations = load_latest_annotations(
        annotations_path
    )

    validate_annotation_file_hash(
        annotations,
        test_hash,
    )

    test_ids = set(test_index.keys())
    annotation_ids = set(annotations.keys())

    missing_ids = sorted(
        test_ids - annotation_ids
    )

    unknown_ids = sorted(
        annotation_ids - test_ids
    )

    if unknown_ids:
        raise SystemExit(
            "Annotations contain IDs not present in "
            "the test set: {}".format(
                unknown_ids[:10]
            )
        )

    if missing_ids and not args.allow_partial:
        raise SystemExit(
            "Human annotation is incomplete: {} of {} items "
            "are missing.\n"
            "Finish annotation first, or use --allow-partial "
            "only for a temporary diagnostic.".format(
                len(missing_ids),
                len(test_ids),
            )
        )

    if missing_ids:
        scoring_records = []

        for index, record in enumerate(
            test_records,
            start=1,
        ):
            item_id = get_item_id(record, index)

            if item_id in annotations:
                scoring_records.append(record)
    else:
        scoring_records = test_records

    merged_records = merge_annotations(
        test_records,
        annotations,
    )

    report = {
        "created_at_utc": utc_now(),
        "test_file": str(input_path),
        "test_file_sha256": test_hash,
        "annotations_file": str(
            annotations_path
        ),
        "coverage": {
            "test_items": len(test_ids),
            "annotated_items": len(
                test_ids & annotation_ids
            ),
            "missing_items": len(missing_ids),
            "missing_item_ids": missing_ids,
        },
        "human_vs_strict": (
            calculate_human_strict_report(
                scoring_records,
                annotations,
            )
        ),
    }

    if args.annotations_b:
        annotations_b_path = Path(
            args.annotations_b
        )

        annotations_b = load_latest_annotations(
            annotations_b_path
        )

        validate_annotation_file_hash(
            annotations_b,
            test_hash,
        )

        report["second_annotations_file"] = str(
            annotations_b_path
        )

        report["human_vs_human"] = (
            calculate_human_human_report(
                annotations,
                annotations_b,
            )
        )

    if args.merged_output:
        merged_output = Path(
            args.merged_output
        )
    else:
        merged_output = input_path.with_name(
            "test_with_human_annotations.jsonl"
        )

    if args.report_output:
        report_output = Path(
            args.report_output
        )
    else:
        report_output = input_path.with_name(
            "human_check_report.json"
        )

    write_jsonl(
        merged_output,
        merged_records,
    )

    write_json(
        report_output,
        report,
    )

    if args.subsets_dir:
        write_confirmed_subsets(
            Path(args.subsets_dir),
            merged_records,
        )

    print_scoring_summary(report)

    print()
    print_separator()
    print("FILES WRITTEN")
    print_separator()
    print("Merged test: {}".format(merged_output))
    print("Report:      {}".format(report_output))

    if args.subsets_dir:
        print(
            "Subsets:     {}".format(
                args.subsets_dir
            )
        )


# =============================================================================
# CLI
# =============================================================================

def build_parser():
    # type: () -> argparse.ArgumentParser
    parser = argparse.ArgumentParser(
        description=(
            "Blind human annotation and scoring "
            "for the frozen strict test set."
        )
    )

    subparsers = parser.add_subparsers(
        dest="command",
    )

    annotate_parser = subparsers.add_parser(
        "annotate",
        help="Run blind console annotation.",
    )

    annotate_parser.add_argument(
        "--input",
        required=True,
        help="Frozen strict test JSONL.",
    )

    annotate_parser.add_argument(
        "--annotator",
        required=True,
        help="Annotator identifier, for example: kt.",
    )

    annotate_parser.add_argument(
        "--output",
        default=None,
        help=(
            "Annotation JSONL. By default it is "
            "created beside the test file."
        ),
    )

    annotate_parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=(
            "Deterministic item-order seed. "
            "Use the same seed when resuming."
        ),
    )

    annotate_parser.add_argument(
        "--rubric-version",
        default=DEFAULT_RUBRIC_VERSION,
        help=(
            "Rubric version stored with every annotation."
        ),
    )

    score_parser = subparsers.add_parser(
        "score",
        help=(
            "Score completed annotations "
            "against strict labels."
        ),
    )

    score_parser.add_argument(
        "--input",
        required=True,
        help="Frozen strict test JSONL.",
    )

    score_parser.add_argument(
        "--annotations",
        required=True,
        help="Primary human annotations JSONL.",
    )

    score_parser.add_argument(
        "--annotations-b",
        default=None,
        help=(
            "Optional second annotator JSONL "
            "for human-human agreement."
        ),
    )

    score_parser.add_argument(
        "--merged-output",
        default=None,
        help=(
            "Output JSONL containing test records "
            "plus human fields."
        ),
    )

    score_parser.add_argument(
        "--report-output",
        default=None,
        help="Output JSON report.",
    )

    score_parser.add_argument(
        "--subsets-dir",
        default=None,
        help=(
            "Optional directory for human-confirmed "
            "response, distress, and joint subsets."
        ),
    )

    score_parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Allow temporary scoring before all "
            "test items are annotated."
        ),
    )

    return parser


def main():
    # type: () -> None
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        raise SystemExit(2)

    try:
        if args.command == "annotate":
            run_annotation(args)

        elif args.command == "score":
            run_scoring(args)

        else:
            parser.error(
                "Unknown command: {}".format(
                    args.command
                )
            )

    except FileNotFoundError as error:
        raise SystemExit(str(error))

    except ValueError as error:
        raise SystemExit(
            "Data error: {}".format(error)
        )


if __name__ == "__main__":
    main()
