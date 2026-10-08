# SFT: v1.3 soft verifier confidence

This is an executable preparation/training path for **caller-owned data**, not an
end-to-end reproduction of the paper dataset. The original normalized training
units, offline candidate replay, and verifier-v2 score files are not distributed.
You must obtain or construct these from data you are authorized to use, keeping
training videos separate from calibration/evaluation videos. The preparer does
not index datasets, run the verifier, or certify split separation.

The final reported SFT configuration is **v1.3 soft confidence**, with 60,405
units, 49,766 source videos, 582,900 unique teacher candidates, and **1,888
optimizer steps**. Those counts describe the historical run, not bundled data.
For your data the executable computes `2 * ceil(number_of_units / 64)`; it does
not silently force 1,888 steps or claim dataset parity.

## Frozen recipe

- Original TimeLens2-4B revision `ddbb6cb944f13ce21e59e85da23c5f356107260e`;
  fresh LoRA rank 32, alpha 64, dropout 0.05. No previous adapter/optimizer.
  Language projection targets: `q_proj`, `k_proj`, `v_proj`, `o_proj`,
  `gate_proj`, `up_proj`, `down_proj`; vision parameters stay frozen.
- Generator learning rate `2e-5`; confidence head `2e-4`; AdamW with betas
  `(0.9, 0.95)`, epsilon `1e-8`, weight decay `0.01`, cosine scheduling,
  3% warmup, gradient clipping 1.0. Two epochs, target global query batch 64.
- Generator targets retain all exact GT anchors, then bounded model-supported
  and synthetic **positive boundary alternatives**. Weighted CE group masses
  are GT/alternatives/format = `0.6/0.3/0.1` (`0.9/0/0.1` without alternatives).
  There is no v1.2 synthetic-fraction cap or residual-prefix intervention.
- Each normalized unit uses `sorted(source_ids)[0]` as its single unchanged
  historical replay. Replay contributes **zero LM loss**. New synthetic
  generation alternatives are never assigned fabricated verifier targets.
- Confidence uses mean interval-text-token hidden states with stop-gradient,
  followed by LayerNorm(2560), Linear(2560,256), GELU, Dropout(0.1), Linear(256,1).
  The objective is weighted CE + `0.25 * (soft BCE + 0.25 * ranking)`.
  Ranking uses ordered within-replay pairs with teacher probability gap >= 0.1.
  Teacher probabilities stay continuous; they are not thresholded for training.
  The initialization name `random_span_hidden_binary_mlp` means a single-logit
  BCE head, **not GT-binary supervision**. Head seed is 20260902; trainer seed
  is 20260813. Copied optional GT-binary utilities are not selected by this CLI.

## Dependencies and local model

Preparation/tests need Python >=3.10 and `tokenizers`, not PyTorch or a GPU:

```bash
python -m pip install tokenizers==0.22.1
```

Use a complete, local, pinned TimeLens2-4B snapshot: `config.json`,
`tokenizer.json`, `tokenizer_config.json`, `preprocessor_config.json`, a chat
template (a `chat_template.json`/`.jinja` file or tokenizer-config entry), and
`model.safetensors` or an index plus every referenced safetensors shard. Preserve
all upstream processor/video-processor files. The text hidden size must be 2560
and model type `qwen3_vl`. An inference LoRA adapter directory is not a base model;
`adapter_config.json` in the base directory is refused to prevent auto-loading an adapter.

Supply a trusted SHA-256 for the tokenizer. Its bytes must equal the base
snapshot's `tokenizer.json`; do not use a tokenizer from an unrelated checkpoint.
Preparation hashes local model files and checks their structure/presence, but
cannot prove that arbitrary local weights really came from the stated upstream
revision. Obtain the snapshot and its provenance under the upstream model terms.
No model download or verifier service is invoked automatically.

## Caller input schemas

All three inputs are JSONL objects, one object per line. UTF-8 query text and
candidate order are significant. Paths on the CLI may be relative; output paths
in the manifest are absolute. A relative `meta.video_path` is resolved relative
to the units JSONL file. Videos must exist locally and match `video_sha256`.
Preparation hashes each distinct video path once but does **not** decode it.

### 1. Normalized units (`--units`)

Illustrative shape only; replace identifiers, paths, checksums and annotations
with your own real records:

```json
{"coverage_unit_id":"unit-001","gt":[[2.0,4.0]],"original_gt":[[2.0,4.0]],"proposals":[[1.8,3.8]],"source_ids":["replay-001"],"meta":{"source_video_uid":"dataset:video-001","video_path":"videos/video-001.mp4","video_sha256":"REPLACE_WITH_VIDEO_SHA256","source_duration_s":10.0,"query":"the person opens the door"}}
```

- `coverage_unit_id` is unique. `(source_video_uid, whitespace-normalized,
  case-folded query)` must also be unique; normalize/aggregate duplicates first.
- `gt` contains sorted, unique, finite, positive-width intervals in seconds,
  within `[0, source_duration_s]`. Overlapping distinct occurrences are allowed.
  All GT anchors must fit 32 candidates and the 480-token target budget; nothing
  is silently dropped when they do not fit.
- `original_gt` retains the sorted original annotations with the same cardinality.
  By default it must equal `gt`. `--gt-rounding-tolerance-s X` permits only
  explicit clipping to video bounds with each coordinate changed by at most X;
  it does not permit arbitrary boundary edits, reordering, merging or deletion.
- `proposals` is an explicit list (possibly empty) of caller-supplied proposal
  intervals, finite and within duration. These are generation-pool inputs, not
  teacher scores. `source_ids` is a nonempty list of distinct replay request IDs.
- Optional `meta.video_contract` must exactly match the fixed training contract:
  `{"fps":2.0,"max_frames":120,"min_pixels":4096,"max_pixels":65536,"total_pixels":7864320}`.
  It is inserted when absent. Reused video identities must agree on checksum
  and duration.

The unchanged `sft_targets.PoolBuilder.row(..., "supported_expansion")` and
`render` implement target construction. Defaults: at most two extras per GT,
model tIoU >=0.5, synthetic tIoU >=0.6, near-duplicate cutoff 0.85, maximum 32
candidates, target <=480 tokens including assistant suffix, decode cap 512,
context cap 16,384. There is no separate prebuilt-row bypass in this CLI.

### 2. Offline replay (`--replay`)

```json
{"request_id":"replay-001","source_video_uid":"dataset:video-001","video_sha256":"REPLACE_WITH_VIDEO_SHA256","query":"the person opens the door","gt_intervals":[[2.0,4.0]],"prompt_text":"REPLACE_WITH_THE_ACTUAL_OFFLINE_GENERATION_PROMPT","direct_target_override":"[[2.0,4.0],[6.0,7.0]]","direct_candidate_slots":[{"candidate_id":"candidate-001-a","slot_role":"offline_model_proposal"},{"candidate_id":"candidate-001-b","slot_role":"offline_model_proposal"}]}
```

Every selected request must exist exactly once. Extra unselected requests may
be present. Video identity, checksum, query, and original GT must match the unit.
`direct_target_override` is the **actual saved JSON interval sequence**, not a
regenerated target. Use ordinary decimal coordinates (no exponent notation);
all candidates must be within duration. Slots align in order, with globally
consistent candidate IDs and nonempty role metadata. Each sequence must have
unique candidate IDs; overlapping or duplicate-coordinate replay proposals are
not automatically rewritten. `direct_prompt_text`, if supplied, takes precedence
over `prompt_text`. Preserve the original replay prompt and candidate roles.

The unchanged `attach_replay` attaches interval character spans and token hashes.
Replay plus `<|im_end|>\n` must fit 2,048 tokens. Target/suffix tokenization must
be stable. Full prompt + video + target context length is checked later by the
GPU collator, not by CPU preparation.

### 3. Original verifier scores (`--teacher-scores`)

Each selected candidate needs one record with:

| Field | Required value |
| --- | --- |
| `candidate_id` | Exact replay slot ID |
| `source_video_uid` | Matching video identity |
| `query` | Exact matching query text |
| `candidate_interval_source_s` | Matching `[start, end]` coordinates |
| `teacher_yes_probability` | Original verifier-v2 numeric P(Yes), strictly between 0 and 1 |

The historical alternative `scores.yes_probability` is accepted instead. If
both probability fields exist they must agree. Optional `video_sha256` and
`request_id` must match when provided. If your score export lacks query/interval
metadata, join it from the original candidate ledger **by candidate ID**, without
altering or recomputing scores. Positional joins are unsafe.

Missing, duplicate, wrong-video, wrong-query, wrong-interval, nonfinite, or hard
0/1 targets are refused. Unused scores can exist in the source export; only
selected candidates are copied, with their values unchanged. No fallback uses
tIoU, GT membership, constants, random probabilities, or synthetic labels.
The preparer validates alignment, not the authenticity of an external verifier.

The unchanged trainer gates require at least **5,000 replay candidate training
occurrences**, including at least 1,000 with P(Yes) >=0.5 and 1,000 below 0.5.
Every unit must have confidence replay. These are validation counts, not binary
training labels. Tiny real datasets that fail these gates are refused; do not
pad them with invented labels to pass.

## Prepare and verify on CPU

Set absolute paths for your checkout, work directory, local model, and trusted
tokenizer digest. The work directory must exist; `prepared` must not exist.

```bash
REPO=/absolute/path/Grounding-with-Confidence
WORK=/absolute/path/sft-work
BASE=/absolute/path/TimeLens2-4B
TOKENIZER_SHA=REPLACE_WITH_TRUSTED_TOKENIZER_SHA256
mkdir -p "$WORK"

python "$REPO/scripts/prepare_sft.py" \
  --units "$WORK/raw/units.jsonl" \
  --replay "$WORK/raw/replay.jsonl" \
  --teacher-scores "$WORK/raw/teacher-scores.jsonl" \
  --base-model "$BASE" \
  --tokenizer "$BASE/tokenizer.json" \
  --expected-tokenizer-sha256 "$TOKENIZER_SHA" \
  --world-size 8 \
  --output-dir "$WORK/prepared"
```

This writes `input.jsonl`, selected `teacher-scores.jsonl`, `input-manifest.json`,
`input-manifest.sha256`, and `preparation-report.json`. It runs the trainer's
`prepare_records` checks before publishing and validates the manifest through
`load_manifest`. Existing output directories are never overwritten. Source
JSONL hashes, tokenizer/model-file inventory, computed schedule and gate results
are recorded. The implementation is in-memory; full datasets require adequate
host RAM and hashing all model/video files can be I/O-intensive.

Recheck the prepared inputs on the training host before launching. Do not edit
paths or data without regenerating the manifest and its checksum. If relocating
assets, rerun preparation with the destination paths. Never use `python -O`;
the retained target builder intentionally uses assertions as contract gates.

```bash
MANIFEST_SHA=$(python -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).read_text().split()[0])' "$WORK/prepared/input-manifest.sha256")
PYTHONPATH="$REPO/scripts:$REPO/src" python - "$WORK/prepared/input-manifest.json" "$MANIFEST_SHA" <<'PY'
from pathlib import Path
import sys
from train_sft import load_manifest, prepare_records
from tvg_confidence.common import sha256_file
manifest = load_manifest(Path(sys.argv[1]), sys.argv[2])
for name, digest in manifest["model"]["files_sha256"].items():
    assert sha256_file(Path(manifest["model"]["path"]) / name) == digest, name
assert sha256_file(Path(manifest["tokenizer_contract"]["path"])) == manifest["tokenizer_contract"]["sha256"]
records, summary = prepare_records(manifest)
for path, digest in {(r["video_path"], r["video_sha256"]) for r in records}:
    assert sha256_file(Path(path)) == digest, path
print(summary)
PY
```

## Train on CUDA

The retained GPU trainer requires **transformers 4.57.1** and CUDA; CPU/MPS
training is not supported. Use a CUDA-compatible PyTorch installation. A pinned
training dependency set, separate from CPU preparation, is:

```bash
python -m pip install \
  torch==2.6.0 torchvision==0.21.0 transformers==4.57.1 tokenizers==0.22.1 \
  peft==0.17.1 accelerate==1.6.0 qwen-vl-utils==0.0.14 \
  safetensors==0.5.3 decord==0.6.0 av==14.2.0 numpy==2.1.2
```

For the default eight-GPU, single-node manifest:

```bash
TVG_VIDEO_READER_BACKEND=decord torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  "$REPO/scripts/train_sft.py" \
  --input-manifest "$WORK/prepared/input-manifest.json" \
  --expected-manifest-sha256 "$MANIFEST_SHA" \
  --output-dir "$WORK/run" \
  --run-mode joint-only --lambda-conf 0.25
```

`run` must be new or empty. The default trainer run mode is `both`, so the
explicit `--run-mode joint-only` above is essential. No direct-reference report
is needed for this standalone v1.3 manifest. Each GPU processes one micro-sample
at a time; direct and detached-replay branches are separate samples. At 60,405
units/eight GPUs, accumulation is 16, each epoch has 944 steps and 22 zero-weight
padding micro-samples. Padding does not duplicate real training supervision.

For fewer GPUs, prepare a new manifest with matching `--world-size` (1, 2, 4,
8, 16, 32, or 64) and match `--nproc_per_node`; accumulation is computed to retain
the query/step contract. This is not a guarantee the 4B video model fits a given
GPU. BF16-capable CUDA hardware, enough VRAM for video/context activations and
sufficient host RAM/storage are required. Peak memory and GPU execution are not
validated by the CPU tests. No distributed fallback environment overrides are
needed for the documented launch.

Compact checkpoints are saved every 500 steps and at the final step, retaining
four. To recover an interrupted matching run, repeat the same command with
`--recover-partial-output`; keep the same manifest, trainer code, world size and
asset paths. Checkpoint contracts are verified rather than manually bypassed.

## SFT output / release inventory

After success, retain these matched artifacts under `run/joint/`:

- `generator-adapter/adapter_model.safetensors` and `adapter_config.json`;
- `span-confidence-head.safetensors` and `span-head-config.json` (the config binds
  the head to adapter hashes); `confidence-head.safetensors` is the same head
  under the trainer's legacy name, not an independently trained model;
- `processor/` and `report.json`, plus root `preflight.json`, `report.json` and
  `_SUCCESS` for provenance.

`joint/trainer-state/` contains resumable optimizer/checkpoint state, not the
minimal inference release. The adapter still requires the pinned base weights.
Do not mix the head with a different adapter or upload caller videos/teacher
files unintentionally. This preparation CLI does not upload weights or run
inference/evaluation; those are separate repository workflows.

## CPU tests

```bash
python -m unittest discover -s "$REPO/tests" -p test_sft_preparation.py -v
```

Tests exercise the executable materializer, checksums, schema/alignment failures,
unchanged soft scores, trainer `prepare_records`, and real `QueryDataset` step
and padding calculations, including 60,405 -> 1,888. Their 1,000-unit/5,000-score
fixtures use explicitly artificial videos, weights and labels **only for tests**;
they cannot be used as training data or evidence of GPU/model correctness.
