# Initial release

This standalone repository contains inference and evaluation code. Training
code and data remain coming soon. The public hosting account is not yet set.

The first model release pairs the final 4B generator at step 200 with its
confidence head after 150 online updates. `configs/release-model.json` binds
the expected generator and head hashes.

To assemble a local Hugging Face model directory from the accepted exports:

```bash
python scripts/prepare_model_release.py \
  --model-dir /path/to/merged-generator \
  --head-dir /path/to/matched-head \
  --output-dir /path/to/model-release
```

The preparer verifies the matched weights, copies only inference files, removes
machine-specific configuration fields, and writes a portable `release.json`.
It leaves the source exports unchanged. The model card is a draft until the
hosting identity, links, and license metadata are filled in. No upload occurs.

The original inference core is retained in `scripts/_infer_core.py`; the public
wrapper supplies portable inputs and the reported OMTG video budget. Metric
functions and generation-order NMS are extracted from the same implementation.
CPU example evaluation has passed. A GPU end-to-end run with the assembled model
is still required before marking the initial release as available.
