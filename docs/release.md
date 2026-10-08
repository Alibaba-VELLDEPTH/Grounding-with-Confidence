# Preparing a public release

This is a standalone code repository, not an export of the research workspace or
its Git history. Public GitHub/Hugging Face hosting IDs and license metadata must
be confirmed by the maintainers before publication. No command below uploads or
publishes anything.

## Final manuscript weights

The first model release pairs the final 4B generator at step 200 with its
confidence head after 150 online updates. `configs/release-model.json` binds the
accepted generator and head hashes. Do not replace the head with the SFT bootstrap
head or a later experimental head.

Assemble a local Hugging Face directory from the accepted full-model exports:

```bash
python scripts/prepare_model_release.py \
  --model-dir /path/to/merged-generator \
  --head-dir /path/to/matched-head \
  --output-dir /path/to/new-model-release --check-only

python scripts/prepare_model_release.py \
  --model-dir /path/to/merged-generator \
  --head-dir /path/to/matched-head \
  --output-dir /path/to/new-model-release
```

The preparer verifies every generator file and the matched head, copies only
inference files, removes machine-specific configuration fields, and writes a
portable `release.json` and `MANIFEST.sha256`. It leaves source exports unchanged
and refuses an existing destination. Original weight bytes do not change; portable
configuration checksums can change. This preparer is deliberately restricted to
the frozen manuscript RL model, not arbitrary retrained or SFT checkpoints.

Upload the **whole resulting model directory** after the checks below, including:

- All three generator SafeTensors shards and their index.
- Generator configuration, tokenizer, chat template, and image/video processors.
- `span-confidence-head.safetensors` and `span-head-config.json`.
- `release.json`, `MANIFEST.sha256`, `.gitattributes`, and completed `README.md` model card.
- `LICENSE-TIMELENS2.txt` retaining the pinned upstream source notice; add any
  other notices and approved model license required for the final release.

Do not upload training `.pt` shards, optimizer/RNG states, local requests, logs,
internal receipts, or storage addresses. The full merged generator needs no
additional base-model download at inference.

## Validation and publication gates

1. Confirm the code license and model redistribution terms with the rights holder;
   add the chosen license and attribution. No license is implicitly granted by
   the presence of source files or these instructions. The TimeLens2 source at
   revision `f828dde4ee7f2aea5022c6bd7f5aa1d42d4b3f35` carries academic-only and
   geographic restrictions; do not label derived weights MIT/Apache by default.
   Review the exact model repository's license and all upstream terms separately.
2. Run CPU checks from this repository, in the documented Python environment:

   ```bash
   python -m pip install -r requirements-dev.txt
   python -m unittest discover -s tests -v
   python scripts/evaluate.py --input examples/predictions.jsonl --threshold 0.5
   ```

3. On a CUDA machine, use actual local videos and an assembled model:

   ```bash
   python scripts/infer.py --model /path/to/new-model-release \
     --input /path/to/smoke-requests.jsonl --output-dir outputs/smoke \
     --threshold 0.5 --evaluate
   ```

   Include GT for all requests. Inspect candidate/score alignment, rejected
   queries, failed-query retention, and `evaluation.json`. CPU tests and a mocked
   inference backend do **not** establish GPU execution correctness.
4. Validate the SFT and three-phase RL recipes on the target GPUs before claiming
   end-to-end training reproduction. Original videos, teacher labels, and paper
   split manifests are not included: document whether supplied replacements are
   custom-data runs or exact paper inputs.
5. Scan the code and Git history for private paths, credentials, unlicensed data,
   and large binaries. Only this standalone repository belongs on GitHub.
6. Finalize hosting IDs and model-card license/ownership. The public paper is
   [arXiv:2609.39883](https://arxiv.org/pdf/2609.39883).
   HF links currently use the [Hugging Face homepage](https://huggingface.co/)
   as a placeholder, not a model download. After manual HF upload, replace them
   with the actual model link in `index.html`, `README.md`, `models/README.md`,
   and the model card, and record the model ID and immutable revision. Re-test
   the Hub download path with `infer.py --model MODEL_ID --revision COMMIT`.

## Validation scope

The original same-decode scoring and legacy set-metric implementations are
retained, with matching key function ASTs from the frozen anonymous supplement.
The default OMTG metric path additionally reproduces the official merge-before-
matching behavior and empty-set rules; recomputation on all 320 archived final
selected sets and original GT exactly matches the frozen report. This does not
rerun confidence scoring, generation, or threshold selection.

CPU tests cover NMS-before-threshold selection, official/legacy metric differences,
query denominators, portable model packaging, artifact tampering, SFT adapter/head
identity, data preparation, RL phase/resume/patch contracts, and a mocked
inference-to-eval workflow. Set `TVG_TEST_OMTG_ROOT` to a local pinned checkout to
include upstream patch integration. PyArrow is needed for the Parquet round-trip.
Model loading, CUDA inference, and distributed training remain separate GPU
acceptance gates, not inferred from these CPU checks.

The local preparation check passed 43 tests, including the Parquet round-trip
and temporary-copy integration against the pinned upstream. It used Python
3.12.13, tokenizers 0.22.1, NumPy 2.5.1, SciPy 1.18.0, and PyArrow 25.0.0 from
an existing CPU environment after the pinned dependency installation stalled.
This is **not** validation of the pinned GPU or CPU lockfiles; re-run in the
listed dependency environments on the release machine.
