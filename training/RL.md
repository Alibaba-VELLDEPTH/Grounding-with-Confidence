# Local RL workflow

Public training videos and annotations are available from
[OMTG-56K](https://huggingface.co/datasets/insomnia7/omtg56k) (including its RL
subset) and [TimeLens2-93K](https://huggingface.co/datasets/MCG-NJU/TimeLens2-93K).
See the [data guide](../data/README.md) for downloads and upstream credits.
Normalize annotations into the request JSONL schema below, then use
`scripts/prepare_rl.py` to convert them into our train/monitor Parquet format;
raw upstream annotation formats are not parsed automatically.

This is executable training code, not a GPU reproduction report. Paper-specific
train/monitor/calibration/test split files, videos, teacher scores, and results
are not bundled. Use data under its original terms and keep calibration/test
videos out of **both** RL splits. The monitoring reward uses its own batch GT
and is not a held-out test metric or a deployable calibrated threshold.

## Environment

Use Linux, Python 3.11/3.12, sufficient CPU RAM/local disk, and exactly **eight
visible CUDA GPUs on one node**. Run **one Python driver**, not eight shell
processes or `torchrun`. No GPU brand is enforced. Use a separate RL environment;
the inference/SFT environment can have different dependency requirements.

```bash
python -m pip install -r requirements-rl.txt
python -m pip install -e .
```

FlashAttention is **not pinned** in `requirements-rl.txt`. Install a
FlashAttention build compatible with your CUDA/PyTorch environment separately;
this is an unresolved environment-specific dependency, not an exact lockfile.
The execution preflight resolves Hydra and imports the actual training stack;
incompatible packages fail rather than switching to a different trainer.

Obtain a full local OMTG checkout separately, pinned at
`3291c9e19490db19c7aae593791f80aaca6f53a8`. The launcher never clones, fetches,
installs packages, or downloads models. It requires a clean checkout, checks every
patch anchor, and copies the checkout to a dedicated cache before patching. It
preserves upstream copyright/license files; it does not modify the source tree or
select a license for this project. Do not install an unrelated `verl` in front of
the copy: the launcher and Ray workers verify their import origin.

## Prepare caller-owned data

Each train/monitor JSONL object needs:

```json
{"request_id":"train-001","video_path":"videos/example.mp4","query":"the person opens the door","source_duration_s":20.0,"gt_intervals":[[2.0,5.0]]}
```

This is a schema illustration, not supplied training data. Video paths may be
relative to the input JSONL. `official_row_index` (or `original_index`) is
optional and remains null when unknown. GT must be nonempty and in bounds.
Train/monitor request IDs, resolved video paths, and video content hashes must be
disjoint. An exclusion JSONL can contain `request_id`, `video_path`, and/or
`video_sha256` per object; repeat the option for multiple caller-owned manifests.
Only request IDs cannot establish video disjointness: include paths or hashes for
all calibration/test videos. No unavailable original split is inferred.

```bash
python scripts/prepare_rl.py \
  --train /local/data/train.jsonl --monitor /local/data/monitor.jsonl \
  --exclude-manifest /local/data/calibration-excluded.jsonl \
  --exclude-manifest /local/data/test-excluded.jsonl \
  --output-dir /local/data/rl-prepared --check-only
# Remove --check-only to write train.parquet, validation.parquet, preparation.json.
```

PyArrow is imported only for Parquet IO. The prepared schema is `data_source =
omtg_rl`, a user prompt containing `<video>` plus the shared inference prompt,
`videos = [{video: absolute_path, fps: 2, min_pixels: 2048, total_pixels: 8388608}]`,
a rule `reward_model` with JSON-string GT, and `extra_info` with query ID, original
row index, video ID/path, duration, and file size. Runtime preprocessing verifies
actual decoded frames, visual tokens, duration, and the 16,384-token budget.
The monitor row count must be divisible by eight for the static head-scoring
pass. With the historical three-epoch cap, reaching step 200 requires at least
4,288 train queries; the launcher rejects undersized data rather than silently
ending early. No rows or teacher scores are synthesized.

## Merge the paired SFT parent

The historical parent was **v1.3 soft SFT checkpoint 1888**, merged with
**PEFT 0.17.1**, not GTbinary. The merge utility accepts local base/adapter/head
artifacts, validates adapter/head pairing, and copies the head configuration
unchanged. It records the actual SFT step when `trainer_state.json` is available;
it never relabels a caller's checkpoint as 1888. The RL launcher requires the
v1.3-soft protocol/target but permits caller-trained steps and new weight hashes.

```bash
python scripts/merge_sft.py \
  --base /local/models/base \
  --adapter /local/sft/checkpoint-1888/generator-adapter \
  --head /local/sft/checkpoint-1888/span-confidence-head.safetensors \
  --head-config /local/sft/checkpoint-1888/span-head-config.json \
  --output-dir /local/models/merged-sft
# Review the read-only plan, then repeat with --execute (CPU merge, substantial RAM).
```

The fresh merged directory contains `_MERGED.json`, the complete HF generator,
processor/tokenizer, and unchanged bootstrap head/config. It is a **training
initialization**, not a completed online RL inference release. Use the explicit
SHA-256 of its bootstrap `.safetensors` head in every phase below.

## Train the three phases

Shell variables below are placeholders for your own existing local paths/digest.
Use a fresh output directory for each phase and retain cp50/cp100.

```bash
MODEL=/local/models/merged-sft
HEAD=$MODEL/span-confidence-head.safetensors
HEAD_SHA=YOUR_64_CHARACTER_LOWERCASE_SHA256
DATA=/local/data/rl-prepared
UPSTREAM=/local/source/OMTG
CACHE=/local/cache/grounding-rl

python scripts/train_rl.py --phase warmup \
  --model "$MODEL" --head "$HEAD" --head-sha256 "$HEAD_SHA" \
  --train "$DATA/train.parquet" --val "$DATA/validation.parquet" \
  --upstream-root "$UPSTREAM" --cache-dir "$CACHE" --output-dir /local/runs/warmup

python scripts/train_rl.py --phase online \
  --model "$MODEL" --head "$HEAD" --head-sha256 "$HEAD_SHA" \
  --train "$DATA/train.parquet" --val "$DATA/validation.parquet" \
  --upstream-root "$UPSTREAM" --cache-dir "$CACHE" --output-dir /local/runs/online \
  --resume /local/runs/warmup/global_step_50 \
  --bootstrap-checkpoint /local/runs/warmup/global_step_50

python scripts/train_rl.py --phase low-lr \
  --model "$MODEL" --head "$HEAD" --head-sha256 "$HEAD_SHA" \
  --train "$DATA/train.parquet" --val "$DATA/validation.parquet" \
  --upstream-root "$UPSTREAM" --cache-dir "$CACHE" --output-dir /local/runs/low-lr \
  --resume /local/runs/online/global_step_100 \
  --bootstrap-checkpoint /local/runs/warmup/global_step_50
```

Each command **defaults to dry-run**: it hashes inputs/checkpoints, validates
schema/lineage/source anchors, and prints environment/Hydra argv without creating
output/cache or starting Ray. Subsequent phase plans require the actual parent
checkpoint. Add `--execute` to each reviewed command to train. The driver cleans
inherited rank/TORCHELASTIC variables, enables offline model/dataset caches, and
propagates public `src`, `scripts`, `scripts/rl`, copied `rl/verl`, and copied `rl`
imports into Ray TaskRunner and FSDP workers. The caller's upstream is read-only.

| Phase | Actor steps | Head | Actor LR | Saves |
|---|---:|---|---:|---|
| warmup | 1–50 | frozen SFT head | 1e-6, five-step warmup then constant | 50 |
| online | 51–100 | one online update per outer step | 1e-6 | 51, 100 |
| low-lr (original L) | 101–200 | continued online head | 2e-7 after complete restore | 101, 125, 150, 175, 200 |

Warmup/online retain the original recall plus fixed-threshold selected-tIoU reward.
L uses `0.5 * pool recall + 0.25 * selected F1 + 0.25 * selected tIoU` with one
macro-F1-optimizing threshold across the entire training batch, not per query.
All phases use 64 queries × 8 rollouts, PPO minibatch 16, one microbatch per GPU,
synchronous vLLM TP2, no KL loss/reward, GRPO without standard-deviation
normalization, and token importance correction threshold 2.0. Historical Hydra
values are exposed in `build_overrides`; the only planned step-budget change is
STOP50/100/200 with the existing constant post-warmup scheduler.

Actor/head state restoration precedes any L override. Its identity-bound JSON is
`{start_step:100, parent_actor_lr:1e-6, actor_lr:2e-7,
parent_marker_sha256:SHA256(cp100/_COMPLETE.json)}`; the phase SHA uses UTF-8
`json.dumps(phase, sort_keys=True)` (including its default spaces), matching the
historical runtime. It is part of the experiment identity. Parent restore changes
LR/base LR/last LR only, not scheduler progression or optimizer moments. Resuming
an own L checkpoint verifies and preserves its already-reduced LR.

Checkpoints seal actor model/optimizer/extra-state shards for **all eight ranks**,
dataloader state, and (after step 50) head weights/optimizer/replica state.
Visibility waits are bounded and checksums mandatory. Online scoring captures
unpacked, detached, consumed-token final-norm states **before** actor updates;
the head update occurs **after** the actor update. The historical response-only
513-position vocabulary projection, common left-padding trim, and real-actor
validation parity gate are enabled for **L only**; warmup/online retain full
projection. The DeepStack clone applies to all phases. A failed parity gate aborts;
none of those GPU checks have been executed as part of releasing this code.

For interruption recovery, pass `--resume` pointing to a complete checkpoint of
that exact phase/experiment, retaining the same bootstrap cp50 argument. There is
no automatic rollback or incomplete-checkpoint recovery. An existing output must
have a matching `workflow-plan.json`; later/incomplete checkpoint directories are
rejected. Choose a new output directory when inspecting an interrupted partial
save. Code/model/data drift requires a new lineage rather than silently resuming.
The identity includes source video content hashes and the local reward, semantic,
and preprocessing dependencies; same-size video replacement is a data change.
Video cache keys are content-based, and unrecorded model artifacts are refused.

## Export and infer your new model

Training state uses PyTorch pickle. **Only deserialize trusted, self-produced
checkpoints**: CPU loading and hashes do not prevent pickle code execution.
Export checks the full completion marker and requires explicit trust before any
actor/head pickle loading. It reconstructs the eight-rank FSDP2 tensors on CPU and
verifies head optimizer update count and eight matching replica digests.

```bash
python scripts/export_rl.py \
  --checkpoint /local/runs/low-lr/global_step_200 --expected-step 200 \
  --expected-marker-sha256 YOUR_CP200_MARKER_SHA256 \
  --output /local/releases/my-rl --check-only
# Then remove --check-only and add --trust-self-produced-checkpoint.

python scripts/infer.py --model /local/releases/my-rl/hf-model \
  --input /local/data/test-requests.jsonl --output-dir /local/results/my-rl \
  --evaluate
```

Export writes `hf-model/`, `online-head/`, and receipts. The head at actor step 200
has **150 online updates**. `hf-model/release.json` hashes the actual newly trained
model/head/config and binds their checkpoint marker/step. It is accepted directly
by `infer.model_arguments`; **do not copy `configs/release-model.json`** into a new
actor release. Exporting frozen-head cp50 as an online inference release is
intentionally rejected. The complete head-only exporter is `scripts/rl/export_head.py`.
Public generator configuration removes machine-specific name/cache fields, and
the public head identity retains only step and checkpoint hash. Operational
receipts and `online-head/` training state stay private: only `hf-model/` is the
inference bundle, and licensing/model-card review is still required before upload.

## CPU checks and remaining limitations

```bash
TVG_TEST_OMTG_ROOT=/local/source/OMTG \
  python -m unittest discover -s tests -p test_rl_workflow.py -v
```

Tests cover phase/env/argv, LR transition and own-phase resume, markers/shards,
output guards, caller-owned schema/disjointness, dynamic release pairing, and
idempotent fail-closed hooks on a **temporary read-only upstream copy**. The
Parquet round-trip test skips without PyArrow; upstream integration skips without
the explicit local checkout variable. These checks do not establish GPU numerical
reproduction, actual model merging/export success on full weights, CUDA memory
fit, or equivalence of caller-owned data to unavailable original splits.
