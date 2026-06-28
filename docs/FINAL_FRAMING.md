# FINAL FRAMING (advisor-locked) - single source of truth for README + slides

This file records the EXACT framings approved by the advisor for the final writeup. All README
sections and presentation slides must use this language. The guiding principle: report point
estimates and bootstrap-reliable conclusions separately, and never overclaim significance that the
67-item test did not establish.

## 1. Distress headline (use verbatim)

> DistilBERT-B achieved the highest mean distress performance (macro-F1 0.864, linear weighted kappa
> 0.851, MAE 0.113). However, its advantage over TF-IDF-B was not resolved reliably on the 67-item
> test. Both supervised systems substantially and bootstrap-reliably outperformed zero-shot BART.

Designations:
- best point estimate: DistilBERT-B
- best transformer: DistilBERT-B
- competitive simpler model: TF-IDF nominal B
- reliable conclusion: task-specific supervised training is markedly better than the evaluated
  zero-shot baseline.

Allowed: "Under the evaluated configurations, distress classification benefited strongly from
task-specific supervised training."
NOT allowed: "DistilBERT is proven better than TF-IDF" / "distress definitively requires a
transformer."
Caveat to always attach: the zero-shot result pertains to OUR two prompt templates and BART-MNLI,
not to all possible zero-shot LLMs.

## 2. Cross-task contrast (use verbatim as the central finding)

> The two tasks responded differently to model family and synthetic-data tier. Response labels were
> highly accessible to a sparse lexical classifier, whereas distress showed a larger descriptive
> advantage for the fine-tuned transformer and a much larger gap between supervised and zero-shot
> systems.

Response: TF-IDF-B best point estimate; TF-IDF-B and DistilBERT-C statistically indistinguishable;
zero-shot relatively competitive; lexical signals strong.
Distress: DistilBERT-B best point estimate; TF-IDF still competitive (0.805); supervised-vs-zero-shot
gap large and stable; low/medium boundary needs a more contextual solution.

Allowed: "Distress appeared to benefit more from contextual fine-tuning than response
classification." NOT: "Distress definitively requires a transformer" (TF-IDF-B reached 0.805 and the
difference interval with DistilBERT-B included zero).

## 3. Ordinal TF-IDF (honest negative result, use verbatim)

> Although distress labels are ordered, the two-binary ordinal decomposition underperformed direct
> multinomial logistic regression. This may reflect the small validation set used for threshold
> selection or limitations of the simple decomposition rather than evidence that ordinal modelling
> is generally unsuitable.

Two stated limitations: (1) thresholds tuned on a 45-message validation set; (2) the two-binary
decomposition is not necessarily the optimal ordinal model. Do NOT run more complex ordinal neural
models - out of scope for completion.

## 4. Human-reference (use verbatim)

> Agreement was strong on human-unambiguous items and substantially weaker on human-ambiguous items,
> with most disagreement concentrated at the low-medium boundary.

Key numbers: human-unambiguous weighted kappa 0.817; human-ambiguous weighted kappa 0.365;
human-low -> model-medium 12 cases; zero low/medium boundary items wrongly sent to high. Reading:
the model makes no dangerous extreme jumps but uses a higher sensitivity threshold for moderate
distress than the annotator.

Name the boundary as a label-scheme limitation (verbatim):
> The distinction between low and medium distress was the least reliable part of the annotation
> scheme and remained sensitive to subjective interpretation.

Mandatory caveat (verbatim):
> Human-reference evaluation was based on one annotator and therefore measures agreement with one
> independent reading rather than agreement with a multi-annotator gold standard.

Also note: high model accuracy vs strict labels on ambiguous items does NOT mean those items are
objectively unambiguous; rather the model appears more closely aligned with the synthetic annotation
policy than with an independent human reading.

## 5. One last control before closing distress: DistilBERT-B weighted vs unweighted

Same 5 seeds, same split, same hyperparameters; tier B only. Reason: in the response task class
weighting mattered (esp. rare classes); for distress weighted loss is part of the main method but
its contribution has not yet been shown. This is a pre-motivated sensitivity analysis (NOT post-hoc
result hunting). It answers: does weighting improve macro-F1; change low/high recall; reduce medium
dominance; affect severe errors. After it, distress is closed. NOT needed: extra encoders, new
zero-shot prompts, complex ordinal architecture, new splits, threshold tuning on test.

## 6. Final assembly priorities (advisor-ordered)

1. ONE cross-task summary table (response + distress side by side):

| Task     | Best lexical | Best transformer | Zero-shot | Reliable conclusion                 |
| -------- | ------------ | ---------------- | --------- | ----------------------------------- |
| Response | TF-IDF-B     | DistilBERT-C     | BART      | supervised TF-IDF > zero-shot       |
| Distress | TF-IDF-B     | DistilBERT-B     | BART      | both supervised systems > zero-shot |

State separately that lexical-vs-transformer differences were NOT resolved by the 67-item test.

2. Central synthetic-data tiers table (A/B/C for BOTH tasks). Main contrast: response DistilBERT
highest mean on C; distress DistilBERT highest mean on B; TF-IDF highest mean on B in both tasks ->
"more synthetic data is not always better; the effect depends on task and model."

3. Consolidated error analysis - three mandatory cases:
   - Response anger artifact: high strict F1, low human alignment, reproduces synthetic judge policy.
   - Response anxiety over-prediction: consensus C inflated anxiety; predicted/support ratio rose;
     cause is data composition, not class weights.
   - Distress low/medium boundary: main human-disagreement source; systematic human-low ->
     model-medium; high recognised well; severe errors nearly absent in trained models.

4. EDA (do not overload): tier sizes; response distribution; distress distribution; response x
   distress heatmap; message-length distribution; class distribution A/B/C; share of
   strict/silver/consensus/review-only; human ambiguity rates. Mandatory caveat (verbatim):
   > The distributions reflect the generation and auditing pipeline, not estimated real-world
   > prevalence.

5. README must enable reproduction of: generation; dual-judge auditing; tier assignment; split
   construction; human check; baselines; fine-tuning; evaluation/bootstrap. Pin: Python version,
   package versions, seeds, model checkpoints, expected input/output files, run commands, and a
   warning NOT to use the test set for tuning.

6. Presentations - 5-slide main narrative:
   1. Motivation and use case
   2. Tasks and synthetic-data pipeline
   3. Dataset tiers and validation
   4. Response + distress results
   5. Error analysis, limitations and conclusions
   Do not crowd the main slide with all models/metrics; show macro-F1, the bootstrap conclusion, and
   one error-analysis example per task.

## 7. Final cross-task conclusion (use verbatim)

> Task-specific synthetic supervision was consistently useful, but its value depended on both the
> task and model family. Sparse lexical models were highly competitive for response classification,
> while distress showed a larger descriptive advantage for contextual fine-tuning. Increasing the
> amount of consensus-labelled data did not uniformly improve performance and sometimes transferred
> annotation-policy artifacts or class imbalance. Human validation showed that the largest residual
> errors were concentrated in semantically ambiguous boundaries rather than severe classification
> failures.
