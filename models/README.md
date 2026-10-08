# Model weights

Hugging Face model weights are being prepared. The first release will pair the
final 4B grounding generator with its confidence head (200 generator steps,
150 online head updates). The public model ID will be added after the hosting
account is selected. For now, the [Hugging Face link](https://huggingface.co/)
is a homepage placeholder, not a model download.

The release will contain the merged generator, tokenizer, processor, confidence
head in SafeTensors format, head configuration, and `release.json` checksums.

Use `scripts/prepare_model_release.py` to verify and assemble the accepted
exports. See [the release guide](../docs/release.md). Upload the resulting full
directory, not only the weight shards: the public loader requires `release.json`
and both head files. `MANIFEST.sha256` records the packaged file hashes.

Locally trained SFT checkpoints can be evaluated with `infer.py --sft-checkpoint`
using the pinned upstream base; they are not interchangeable with the final RL
release. A separate SFT Hub repository is optional and not published yet.

Model weights are distributed separately from the code repository.
