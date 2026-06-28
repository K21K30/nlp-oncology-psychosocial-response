# EXECUTION_LOG.md

Running, per-stage record of every significant decision with its justification, numbers, results,
and conclusion. Stable configuration lives in `PROJECT_MEMORY.md`; advisor-approved wording lives in
`FINAL_FRAMING.md`. Newest stages appended at the bottom.

---

## Stage 1 - Synthetic corpus generation

**Decision.** Build a fully synthetic, attribute-controlled corpus rather than scrape real patient
text. **Justification:** no public labeled dataset exists for this exact joint framing (7-class
psychosocial response + 3-level ordinal distress), emotional labels are subjective and costly to
annotate, and synthetic generation lets us control attribute coverage and enforce a label-leakage
ban. **Method:** `gemma2:27b` (Ollama) conditioned on a sampled attribute combination (role, cancer
stage, cancer type, tone, channel, age group, length, noise) plus an intended (response, distress)
label, asked to write a natural message that never explicitly names the target response label.
**Result:** 2,033 candidate messages generated.
**Conclusion.** The leakage ban reduces the risk that a classifier succeeds by detecting explicit
class-name tokens; the attribute grid forces diversity beyond what free-form prompting would give.

## Stage 2 - Dual-LLM-judge audit and quality tiers

**Decision.** Audit every candidate with two independent judges blind to the intended label and
assign quality tiers. **Justification:** synthetic data is known to be weaker for subjective tasks,
so we need a purity signal; two judges reduce single-judge bias. **Method:** judges `gpt-4o-mini`
(API) + `qwen2.5:32b` (Ollama) predict both labels; tiers defined as strict (both judges confirm
both intended labels, confidence >= 4, no ambiguity), silver (both confirm, confidence >= 3),
consensus (both judges confidently agree on labels differing from the intended ones; relabelled to
the judge consensus). **Result:** 2,033 candidates -> 2,025 valid audited -> 1,273 model-ready
(strict 471 + silver 75 + consensus 727) + 752 review-only; 8 needs-rejudging.
**Conclusion.** This satisfies the >= 2,000-message requirement (2,033 generated / 2,025 audited)
while keeping a clean 1,273-item training-ready subset. Only model-ready items enter the experiments.

## Stage 3 - Splits

**Decision.** A frozen 67-item test set and a 45-item validation set, with three nested training
tiers. **Justification:** nesting train_A subset of train_B subset of train_C isolates the effect of
adding lower-purity data; freezing the test set up front prevents any leakage into tuning.
**Result (seed 42):** test 67, validation 45, train_A 359 (strict), train_B 434 (+silver),
train_C 1161 (+consensus).
**Conclusion.** The three tiers let us answer "does more synthetic data help?" directly and per task.

## Stage 4 - Exploratory data analysis

**Result.** Response is imbalanced (anxiety 458, guilt 234, hope 201, sadness 177, anger 90,
denial 57, acceptance 56); distress is imbalanced toward medium (low 333, medium 694, high 246);
message length median 31 words (4-107). **Conclusion.** Imbalance motivates class-weighted training
and macro-F1 as the primary metric; the distributions reflect the pipeline, not real-world
prevalence, and this caveat is stated in every writeup.

## Stage 5 - Human validation (single blind annotator, 67-item test)

**Decision.** Re-annotate the test set by a human blind to the synthetic labels before trusting any
model number. **Result.** Response: exact agreement 86.6%, Cohen's kappa 0.84. Distress: exact 83.6%,
weighted kappa 0.80 (linear) / 0.86 (quadratic). Agreement was strong on human-unambiguous items and
substantially weaker on human-ambiguous items (distress weighted kappa 0.82 vs. 0.37). The dominant
disagreement is a systematic human-low -> model-medium shift; high distress is detected reliably and
extreme low<->high errors are essentially absent.
**Conclusion.** The synthetic labels are trustworthy enough to evaluate against; the residual
difficulty is semantic ambiguity at the low/medium boundary, not extreme ordinal error.

## Stage 6 - Response classification

**Result (test macro-F1, 67 items).** Majority 0.040; zero-shot BART 0.732; TF-IDF A/B/C
0.818 / **0.856** / 0.787; DistilBERT A/B/C 0.768 / 0.752 / **0.834**.
**Conclusion.** A sparse lexical model (TF-IDF-B) has the highest point estimate; DistilBERT had its
highest mean on tier C. The lexical-vs-transformer difference is not resolved on 67 items (see
Stage 9). DistilBERT had its highest mean on the consensus tier, though the C-vs-A difference was not
statistically resolved.

## Stage 7 - Distress classification

**Result (test macro-F1).** Majority 0.211; zero-shot BART 0.468; TF-IDF nominal A/B/C
0.793 / **0.805** / 0.802; TF-IDF ordinal A/B/C 0.786 / 0.786 / 0.752; DistilBERT A/B/C
0.806 / **0.864** / 0.827; DistilBERT-B unweighted 0.841.
**Conclusion.** DistilBERT-B has the highest mean. The ordinal TF-IDF variant is consistently weaker
than the nominal one - an honest negative result that may reflect threshold tuning on only 45
validation items or limitations of the two-threshold decomposition. Both families peak at tier B;
the extra consensus data in C slightly hurts.

## Stage 8 - Class-weighting control (distress, DistilBERT-B)

**Decision.** Compare weighted vs. unweighted under identical split/seeds/hyperparameters.
**Result.** macro-F1 0.864 vs. 0.841 (+0.023); linear weighted kappa 0.851 vs. 0.828 (+0.023);
MAE 0.113 vs. 0.128 (-0.015, better); severe (low<->high) 0.000 vs. 0.000.
**Conclusion.** Class weighting gives a modest but real gain concentrated on the harder low level,
without harming the zero-severe-error profile.

## Stage 9 - Statistical significance (5000-resample stratified paired bootstrap)

**Note.** For DistilBERT, the five seeds' predictions are combined by majority vote, so a bootstrap
point estimate can differ slightly from the mean of the five per-seed scores (e.g. distress 0.853
majority-vote vs. 0.864 five-seed mean).
**Result (distress).** DistilBERT-B 0.853 [0.758, 0.941]; TF-IDF nominal-B 0.805 [0.709, 0.899];
zero-shot 0.464 [0.336, 0.592]. DistilBERT-B vs. TF-IDF-B paired difference [-0.066, +0.164] -> not
separable. Both supervised vs. zero-shot reliable (e.g. DistilBERT-B vs. zero-shot +0.391
[+0.228, +0.555]). **Result (response).** Only TF-IDF-B vs. zero-shot is reliably separable (+0.123
[+0.008, +0.241]); lexical vs. transformer is not separable.
**Conclusion.** Supervised systems had higher point estimates than zero-shot on both tasks;
bootstrap-confirmed superiority holds for TF-IDF-B on response and for both supervised systems on
distress. The lexical-vs-transformer comparison is underpowered at this test-set size.

## Stage 10 - Error analysis (three cases)

1. **Response - anger artifact.** Anger has high strict-label F1 but low alignment with the
   independent human reading; the model appears more closely aligned with the dual-judge annotation
   policy than with the independent annotator's reading (in oncology text, anger is often expressed
   as grief/protest that a human reads as sadness).
2. **Response - anxiety over-prediction.** Training on the consensus tier (C) inflates the
   predicted/support ratio for anxiety; the cause is data composition, not class weights.
3. **Distress - low/medium boundary.** The dominant disagreement both vs. human and across models is
   a systematic human-low -> model-medium shift; high distress is detected reliably.

**Conclusion.** Residual errors concentrate at semantically ambiguous boundaries rather than extreme
ordinal errors.

## Stage 11 - Deliverables and advisor review

**Built.** README (12 instructor sections + reproducibility), `.gitignore`, `requirements.txt`,
three presentations (proposal/interim/final, PPTX + PDF), result figures + visual abstract, and a
LaTeX academic report - all carrying HIT institutional branding (official logo, navy/teal palette,
gradient header/footer bands). Three advisor-bot review rounds were applied to the report and a
fourth to the presentations: corpus-funnel visibility (2,033 / 2,025 / 1,273), precise tier
definitions, separation of point estimates from bootstrap conclusions, correct citations
(Aperstein et al. 2025; full Xu et al. 2026), softened anger and consensus-tier wording, removal of
clinical "safety" language, the 0.864-vs-0.853 explanation, and projector-readable figure fonts.
**Final advisor assessment:** ~9.5/10, ready to submit.

**Central conclusion.** Task-specific synthetic supervision was consistently useful, but its value
depended on both task and model family: sparse lexical models were highly competitive for response,
while distress showed a larger descriptive advantage for contextual fine-tuning. More
consensus-labelled data did not uniformly improve performance and sometimes transferred
annotation-policy artifacts or class imbalance. The classifiers are research prototypes on synthetic
text and are not validated for clinical use.
