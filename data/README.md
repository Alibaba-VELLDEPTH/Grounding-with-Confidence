# Data inputs and release boundary

No training videos, benchmark videos, original split manifests, or offline
verifier scores are redistributed here. Obtain data under its original terms;
check redistribution and use permissions separately from the code license.

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
