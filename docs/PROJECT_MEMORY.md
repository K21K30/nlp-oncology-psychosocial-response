# PROJECT_MEMORY.md

Stable project configuration. This file changes rarely; the running, dated record of decisions
and results lives in `EXECUTION_LOG.md`.

## 1. Identity

- **Title:** Dominant Psychosocial Response and Distress Classification in Oncology-Related Messages
- **Subtitle:** A Synthetic, Label-Leakage-Controlled NLP Study
- **Course:** LLM / Generative AI (NLP), Spring 2026
- **Institution:** Holon Institute of Technology (HIT), Faculty of Digital Medical Technologies
- **Student:** K.T. - solo project
- **Advisor:** A.A.
- **Repository:** `github.com/K21K30/nlp-oncology-psychosocial-response`

## 2. Goal and task

From a single English oncology-related message, predict two labels with **separate** models:

- **Response** - the dominant psychosocial response, 7 classes:
  `anxiety, sadness, anger, hope, guilt, denial, acceptance`.
- **Distress** - the intensity of distress, **ordinal** 3 levels: `low < medium < high`
  (the error low->high is treated as worse than low->medium).

`f(text) -> (response in R, distress in {low, medium, high})`.

## 3. Data (fully synthetic, no real patient data)

- **Corpus funnel:** 2,033 generated candidates -> 2,025 valid audited -> **1,273 model-ready**
  (used in all supervised experiments); 752 review-only; 8 needs-rejudging.
- **Quality tiers** (within model-ready): strict 471, silver 75, consensus 727.
- **Splits (seed 42):** test 67, validation 45; nested training tiers train_A 359 (strict only),
  train_B 434 (+silver), train_C 1161 (+consensus).
- **Class balance (model-ready):** response - anxiety 458, guilt 234, hope 201, sadness 177,
  anger 90, denial 57, acceptance 56; distress - low 333, medium 694, high 246.
- **Message length:** median 31 words (range 4-107).
- Distributions reflect the generation/auditing pipeline, **not** real-world prevalence.

## 4. Attribute space (generation control)

role (patient / caregiver / ...), cancer stage, cancer type, tone, channel, age group, length,
noise flag - sampled to force diversity and coverage.

## 5. Generation and auditing

- **Generator:** `gemma2:27b` via Ollama, conditioned on sampled attributes + an intended
  (response, distress) label.
- **Label-leakage ban:** the generated text never explicitly names the target response label
  (reduces the risk that a classifier succeeds by detecting explicit class-name tokens).
- **Judges (blind to intended label):** `gpt-4o-mini` (OpenAI API) + `qwen2.5:32b` (Ollama).
- **Tier rule:** strict = both judges confirm both intended labels with confidence >= 4 and no
  ambiguity; silver = both judges confirm both intended labels with confidence >= 3 (below strict);
  consensus = both judges confidently agree with each other on labels that differ from the intended
  ones (final labels set to the judge consensus). Malformed/constraint-violating outputs rejected;
  label drift audited separately.

## 6. Models compared (both tasks)

- **Majority class** - trivial sanity floor.
- **TF-IDF + logistic regression** (class-weighted). Distress has two variants: nominal (direct
  3-class) and ordinal (two-threshold `P(y>low)`, `P(y>medium)`, validation-tuned thresholds,
  monotonic correction).
- **DistilBERT** (`distilbert-base-uncased`) fine-tuned with a classification head, separate models
  for response and distress, class-weighted cross-entropy. Reported as mean +/- std over 5 seeds.
- **Zero-shot BART-MNLI** (`facebook/bart-large-mnli`), two prompt templates averaged for distress.

Only DistilBERT uses multiple random seeds; TF-IDF and zero-shot are deterministic.

## 7. Metrics

- **Primary:** macro-F1 (equal weight to every class; appropriate for imbalanced multiclass data).
- **Response:** micro/macro precision, recall, F1 + minority macro-F1 (anger, denial, acceptance).
- **Distress (ordinal):** linear and quadratic weighted Cohen's kappa, MAE, exact accuracy,
  severe-error rate (low<->high), adjacent-error rate, per-level scores.
- **Significance:** 5000-resample stratified paired bootstrap; report per-system 95% CIs and paired
  difference CIs; call a difference reliable only if its CI excludes zero.
- **Human validation:** Cohen's kappa (response) / weighted kappa (distress) vs. a single blind
  annotator on the 67-item test set.

## 8. Environment (NOT Docker)

- Windows 11, **Python 3.10.10**, local virtual environment.
- NVIDIA **RTX 5090**, **CUDA 12.8**.
- `torch==2.11.0+cu128`, `transformers==5.12.1`, scikit-learn, numpy, scipy, pandas, matplotlib, tqdm.
- Local LLMs via Ollama (`gemma2:27b`, `qwen2.5:32b`); `gpt-4o-mini` judge via OpenAI API.
- Project folder: `C:\oncology-distress-detector`.

## 9. Standing methodological rules

- The 67-item **test set is never used for tuning** (no hyperparameter, threshold, or model
  selection on test). Thresholds and model selection use the 45-item validation set only.
- All randomness is seeded; transformer results are mean +/- std over 5 seeds.
- Report point estimates and bootstrap-reliable conclusions **separately**; never overclaim
  significance the 67-item test did not establish.
- The resulting classifiers are research prototypes evaluated on synthetic text; they are **not**
  validated for clinical triage or individual-level decision-making.

## 10. Repository layout (deliverable)

`README.md`, `.gitignore`, `requirements.txt`, `slides/` (3 decks, PPTX+PDF), `code/` (p1...p13),
`data/gen_v6_low_medium/{tiers,splits}`, `results/` (JSON summaries + bootstrap), `visuals/`
(EDA + result figures + visual abstract), `report/` (PDF + LaTeX source + `hit_logo.png`),
`docs/` (this file + `EXECUTION_LOG.md` + `FINAL_FRAMING.md`). Trained checkpoints under `models/`
are not committed (reproduced by the scripts; excluded by `.gitignore`).
