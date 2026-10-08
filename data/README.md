# Data inputs and release boundary

No training videos, benchmark videos, original split manifests, or offline
verifier scores are redistributed here. Obtain data under its original terms;
check redistribution and use permissions separately from the code license.

## Official training data

Download the videos and annotations from the upstream Hugging Face repositories:

| Project | Training data on Hugging Face | Code repository |
| --- | --- | --- |
| TimeLens2 | [TimeLens2-93K](https://huggingface.co/datasets/MCG-NJU/TimeLens2-93K) | [MCG-NJU/TimeLens2](https://github.com/MCG-NJU/TimeLens2) |
| OMTG | [OMTG-56K](https://huggingface.co/datasets/insomnia7/omtg56k), including SFT and RL subsets | [Tangent0308/OMTG](https://github.com/Tangent0308/OMTG) |

With the Hugging Face CLI installed, run from this repository's root:

```bash
hf download MCG-NJU/TimeLens2-93K --repo-type dataset \
  --local-dir data/raw/TimeLens2-93K
hf download insomnia7/omtg56k --repo-type dataset \
  --local-dir data/raw/omtg56k
```

Video archives are large. Follow each dataset card's extraction and layout
instructions; downloading annotation tables alone does not provide local video
files. Record the dataset revisions (or pass `--revision COMMIT` to `hf download`)
and file hashes for your run. Check each dataset's terms and the source videos'
licenses; dataset permissions do not replace model or code permissions.

[OMTG Bench](https://huggingface.co/datasets/insomnia7/omtg_bench) is an evaluation
benchmark, not the OMTG-56K training corpus. Keep its videos, and any other
calibration/evaluation videos, out of training and RL monitoring splits.

We thank the **TimeLens2** and **OMTG** authors for releasing their training data
and code, which make this work and follow-up research possible. Please credit
and cite both upstream projects when using their resources.

## Conversion to our training formats

The workflow is **download → normalize annotations → run our preparation scripts**.
The scripts consume the schemas below, not arbitrary raw Hugging Face rows or
upstream chat-formatted SFT files. Source-format normalization is a separate
caller-supplied step; neither script downloads, extracts, or parses the upstream
annotation formats automatically.

1. Resolve each annotation to its downloaded video, query, GT intervals in
   seconds, and actual video duration. Preserve source IDs and construct
   source-video-disjoint splits, checking duplicates across both datasets.
2. For **SFT**, prepare the normalized units, actual offline candidate replay,
   and matching verifier-v2 soft scores described in [SFT.md](../training/SFT.md#caller-input-schemas).
   `scripts/prepare_sft.py` converts these inputs into bounded GT-anchored targets,
   `input.jsonl`, selected teacher scores, and a checksum-verified training manifest.
   Upstream GT annotations are not substitutes for confidence replay or soft scores.
3. For **RL**, write train/monitor request JSONL using the fields in
   [RL.md](../training/RL.md#prepare-caller-owned-data), then run
   `scripts/prepare_rl.py` to produce `train.parquet`, `validation.parquet`, and
   `preparation.json`. Online confidence targets are computed from GT during RL.

See the linked guides for complete commands. This converts prepared public-data
inputs into our training formats; it does not reconstruct the paper-specific
split manifests, candidate replay, or verifier scores from the downloads alone.

## Inference and evaluation

Use one JSON object per request with `request_id`, local `video_path`, `query`,
and `source_duration_s`. Relative video paths are resolved against the JSONL
file. Add `gt_intervals` for evaluation; an empty list denotes an unmatched query.
See [the input example](../examples/requests.jsonl) and
[the inference guide](../docs/inference.md).

Keep every benchmark request, all GT intervals, their annotation order, and
failed predictions. The evaluator's `--requests` option verifies the complete
query denominator rather than accepting an accidentally filtered prediction file.

## SFT

The reported SFT method uses GT-anchored coordinate targets and a **separate**
fixed candidate replay with verifier-v2 soft P(Yes) labels. Boundary alternatives
are not assigned made-up teacher scores. The confidence features are detached
from generator gradients. See [SFT preparation](../training/SFT.md) for the
required user-supplied rows, labels, and manifest materialization.

The manuscript run used 60,405 units, two epochs and global batch 64 (1,888
optimizer steps). Running the same recipe on custom inputs is not exact
reproduction of those weights. Without the original candidate/teacher inputs,
this repository does not reconstruct them from raw videos automatically.

## RL

RL consumes local videos, text queries and GT intervals; overlap targets for the
online head are produced during training, without an inference-time verifier.
See [RL preparation](../training/RL.md) for Parquet conversion and split checks.
The historical run used 8,424 training rows and 128 monitoring rows. User-supplied
replacements must keep monitoring, calibration and evaluation videos out of
training; the original paper split manifests are not bundled.

The synthetic prediction example is solely a CPU evaluator test, not a released
training dataset or a paper result.
