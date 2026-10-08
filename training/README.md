# Training → inference → evaluation

The supplied workflows implement the reported method on caller-supplied data.
They do not include the original videos, offline verifier labels, or dataset
split manifests, and are not a claim of exact end-to-end paper reproduction.
Use separate Python environments for SFT/inference and RL.

## 1. SFT: generator and detached confidence head

Follow [SFT.md](SFT.md) to prepare normalized GT units, authentic offline replay,
and verifier-v2 soft scores. The preparer creates `input-manifest.json` and its
SHA256 without inventing labels. With `WORK` pointing to that prepared work area:

```bash
python -m pip install -e .
python -m pip install -r requirements.txt
MANIFEST_SHA=$(python -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).read_text().split()[0])' "$WORK/prepared/input-manifest.sha256")

torchrun --standalone --nnodes=1 --nproc_per_node=8 scripts/train_sft.py \
  --input-manifest "$WORK/prepared/input-manifest.json" \
  --expected-manifest-sha256 "$MANIFEST_SHA" \
  --output-dir "$WORK/sft" --run-mode joint-only --lambda-conf 0.25
```

The manifest must be prepared for the same world size. It binds the pinned base,
input identities, and v1.3 soft-confidence contract. The historical run used
60,405 units, global query batch 64, and two epochs (1,888 steps). Confidence
replay contributes no language-model loss and cannot backpropagate into the
generator. This is not the optional GT-binary control.

## 2. Evaluate the SFT result immediately

The trainer writes the paired adapter and head under `$WORK/sft/joint`. In the
SFT/inference environment, with GT-annotated benchmark requests:

```bash
python scripts/infer.py \
  --model /path/to/pinned-TimeLens2-4B \
  --sft-checkpoint "$WORK/sft/joint" \
  --input /path/to/benchmark-requests.jsonl \
  --output-dir "$WORK/sft-eval" --threshold 0.5 --evaluate
```

No adapter merge is needed. `evaluation.json` uses the complete input query set;
failed queries are not dropped. The threshold is illustrative: select it on
separate validation data, not these evaluation labels. Add `--check-only` to
check request and artifact identity without loading the model.

## 3. RL and full-model export

Follow [RL.md](RL.md) for data conversion, SFT merge, pinned upstream framework,
and the three-phase eight-GPU launcher. The required sequence is:

| Actor steps | Head | Actor LR | Reward |
| --- | --- | --- | --- |
| 1–50 | Frozen SFT head | 1e-6 | Pool recall + selected temporal IoU |
| 51–100 | Online detached BCE | 1e-6 | Same warm-start reward |
| 101–200 | Online detached BCE | 2e-7 after optimizer restoration | Pool recall + batch-threshold selected F1 + temporal IoU |

Resume from **complete paired checkpoints**, not actor weights alone. The final
step-200 generator requires its head after 150 online updates. Threshold search
in the final reward is shared across the complete training response batch, not
performed independently for each query. Never use evaluation GT to choose it.

Export only trusted self-produced distributed checkpoints. The RL guide covers
conversion to a full generator with matching online head and portable metadata.
The manuscript release preparer is different: it accepts only the originally
frozen weight hashes, not arbitrary retrained checkpoints.

## 4. Evaluate the exported RL model

Switch back to the SFT/inference environment and pass the portable full-model
release directory described in the RL export guide:

```bash
python scripts/infer.py \
  --model /path/to/exported-release \
  --input /path/to/benchmark-requests.jsonl \
  --output-dir "$WORK/rl-eval" --threshold 0.5 --evaluate

python scripts/evaluate.py \
  --input "$WORK/rl-eval/predictions.jsonl" \
  --requests /path/to/benchmark-requests.jsonl \
  --threshold 0.7 --output "$WORK/rl-eval/evaluation-threshold-0.7.json"
```

The second command reselects from saved candidates without another decode.
Generation-order NMS at tIoU 0.3 always comes first. Training reward uses
maximum-cardinality matching, while reported evaluation uses maximum-total-IoU
assignment before thresholding; these must not be conflated.

## Acceptance boundary

CPU tests cover data/manifest contracts and workflow wiring. Actual SFT, adapter
merging, FSDP export of real weights, and CUDA inference require target-machine
acceptance. Run the [release checks](../docs/release.md) before publishing;
original-data availability and upstream/model license approval are separate gates.
