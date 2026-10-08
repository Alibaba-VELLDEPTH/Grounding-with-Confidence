# Inference and evaluation

The public model ID is pending. Once weights are published, `--model` accepts
either that Hugging Face ID or a downloaded release directory. The CLI uses
[`snapshot_download`](https://huggingface.co/docs/huggingface_hub/guides/download)
for Hub downloads; `--revision` can pin a published commit.

## Installation

Use Python 3.10 or newer and a CUDA GPU. Install CUDA-compatible PyTorch wheels
for the target machine, then install the pinned inference dependencies:

```bash
python -m pip install -e .
python -m pip install -r requirements.txt
```

## Inputs

One JSON object per line:

```json
{"request_id":"example-1","video_path":"example.mp4","query":"A person opens a door.","source_duration_s":30.0}
```

Video paths are local, relative to the request file or absolute. Set the actual
duration of the video in seconds. `gt_intervals` is optional for inference and
required for evaluation. Preserve all benchmark GT intervals and their order.

## Run

```bash
python scripts/infer.py \
  --model /path/to/downloaded-model \
  --input examples/requests.jsonl \
  --output-dir outputs/example \
  --threshold 0.5
```

Replace the example video path and duration before running. The threshold above
is illustrative. With the default threshold of zero, all generation-order NMS
survivors are retained. Select a threshold on separate validation data for an
application; the score is not an independently established calibrated probability.

`predictions.jsonl` contains raw candidates, aligned confidence scores, and the
selected intervals. `outputs.jsonl` retains detailed decode information. Scores
use decoder states cached in the original generation pass. Generation-order
NMS at tIoU 0.3 precedes threshold-based selection. A rejected query has an empty
selected set. Failed queries remain explicit rows.

The OMTG visual budget is 2 fps, `min_pixels=2048`, and
`total_pixels=8388608`, with a 16,384-token context and greedy decoding of at most
512 output tokens. SDPA is the default; FlashAttention 2 is optional.

## Evaluate saved predictions (CPU)

No model or GPU is needed. Install the numerical dependencies for official OMTG
matching (already included in `requirements.txt`):

```bash
python -m pip install -r requirements-eval.txt
python scripts/evaluate.py \
  --input examples/predictions.jsonl \
  --threshold 0.5
```

For model outputs with GT annotations, replace the input path with
`outputs/example/predictions.jsonl`. Evaluation recomputes selection from raw
candidates using the supplied threshold, so it can change without decoding again.
Use `--requests /path/to/benchmark-requests.jsonl` to enforce the full original
query ID set and check durations/GT against the annotations. Missing predictions
are errors, not silently omitted queries; inference failures must remain rows.
`--output outputs/evaluation.json` saves a report and refuses to overwrite one.

## Inference followed by evaluation

Every request must include `gt_intervals` (an empty list is allowed):

```bash
python scripts/infer.py \
  --model /path/to/downloaded-model \
  --input /path/to/benchmark-requests.jsonl \
  --output-dir outputs/benchmark --threshold 0.5 --evaluate
```

This writes `predictions.jsonl` and `evaluation.json`. The same fixed threshold
is used in both; it is never fitted on the evaluation GT. Add `--check-only` to
validate requests, model hashes, and actor/head identity without GPU execution
or creating the output directory. A Hub model ID may download files during this
preflight; use a local directory for an offline check.

## Evaluate a locally trained SFT checkpoint

The trainer saves `generator-adapter/`, `span-confidence-head.safetensors`, and
`span-head-config.json` together in its checkpoint/output directory. Pass that
directory without merging, and use the pinned base model as `--model`:

```bash
python scripts/infer.py \
  --model /path/to/TimeLens2-4B \
  --sft-checkpoint outputs/sft/joint \
  --input /path/to/benchmark-requests.jsonl \
  --output-dir outputs/sft-eval --threshold 0.5 --evaluate
```

Use the base revision `ddbb6cb944f13ce21e59e85da23c5f356107260e` and PEFT 0.17.1.
The head configuration must match both adapter checksums. This mode is distinct
from the full final RL release; do not attach a bootstrap SFT head to the RL actor.

## Metrics and validation status

The default `--metric-protocol omtg` matches the frozen OMTG evaluator: query-macro
precision/recall/tF1 at tIoU 0.3/0.5/0.7, count accuracy, `tiou`, and EtF1, all
as fractions. Maximum-total-IoU assignment uses SciPy before IoU thresholding.
The official merge routine modifies its first interval before that assignment;
this behavior is reproduced on private copies, without modifying saved candidates
or GT. If both sets are empty, official precision/recall/F1/EtF1 are 1 and `tiou`
is 0. Preserve original benchmark GT, including duplicate annotations and order;
normalized training GT is not necessarily identical.

The additional `miou` (TimeLens-style multispan overlap) and `set_tiou` fields are
separate diagnostics, not substitutes for OMTG `tiou`. `--metric-protocol set`
selects the earlier dependency-free set metrics without official merge aliasing;
it gives zero occurrence F1 for two empty sets and can disagree on overlapping
spans. Both `infer.py --evaluate` and `evaluate.py` record the selected protocol.

Failed inference contributes an empty prediction to the full denominator and is
also counted explicitly as a failure, including for unmatched queries. The
three-query example is synthetic. Separately, the official metric function was
checked against all 320 archived final selected interval sets with original GT:
tF1@0.5 = 0.6738897539155692, tIoU = 0.6323736351603254, EtF1 = 0.4263628472222222,
with zero difference from the frozen report. This is a CPU metric check, not a
rerun of generation, confidence scoring, or test-threshold selection.

GPU end-to-end validation of this public wrapper remains pending.
