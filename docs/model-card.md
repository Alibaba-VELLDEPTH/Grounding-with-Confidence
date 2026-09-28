---
language:
- en
base_model: MCG-NJU/TimeLens2-4B
tags:
- video-temporal-grounding
- temporal-localization
- confidence-estimation
---

# Grounding with Confidence — 4B

Model card draft for **Grounding with Confidence: Controllable Generative Video
Temporal Grounding**. The public Hugging Face repository has not been created yet.

Given a video and a text query, the model generates candidate time intervals and
scores them using a lightweight head over decoder states from the same decoding
pass. The candidates support ranking, threshold-based selection, and rejection
without re-decoding or an inference-time external verifier.

The initial model release is the paper's final RL generator paired with its
online confidence head: 200 generator steps and 150 online head updates. The
base model is TimeLens2-4B. The merged generator and confidence head must be used
together.

## Use

Use the repository's `scripts/infer.py` entry point; loading the generator alone
does not apply the confidence head. The release contains a standard Hugging Face
generator and separate `span-confidence-head.safetensors`, `span-head-config.json`,
and `release.json` files.

## Reported OMTG-Bench results

The reported full-system operating point reaches 67.39 tF1@0.5 and 63.24 tIoU
(percent). The manuscript threshold, 0.18404516577720642, was selected on the test
set; it is a retrospective benchmark point. Confidence improves fixed-pool
query-macro Recall@0.5 over generation order by 4.47 and 4.64 percentage points
at 10% and 25% global return budgets, respectively.

## Scope

Confidence selects among generated candidates. It cannot recover an occurrence
absent from the candidate pool or a candidate already suppressed by NMS. Scores
do not establish probability calibration or precision-target transfer to new
datasets. Training code and training data are coming soon.

## Release metadata

Hosting account, public model ID, paper link, and license metadata will be filled
in before publication. This file is a draft, not a claim that weights are already
available.
