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

```bash
python scripts/evaluate.py \
  --input examples/predictions.jsonl \
  --threshold 0.5
```

For model outputs with GT annotations, replace the input path with
`outputs/example/predictions.jsonl`. Evaluation recomputes selection from raw
candidates using the supplied threshold, so it can change without decoding again.

The evaluator reports query-macro precision, recall, and tF1 at tIoU 0.3/0.5/0.7,
count accuracy, EtF1, official multispan overlap (`miou`), and union overlap
(`set_tiou`). Scores are fractions. Maximum-total-IoU assignment precedes
thresholding. Failed inference contributes an empty prediction to the full
denominator. The included three-query example is synthetic, not a paper result.

GPU end-to-end validation of this public wrapper remains pending; its inference
core and evaluator are extracted from the manuscript implementation.
