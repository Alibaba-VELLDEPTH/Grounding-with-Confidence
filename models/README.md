# Model weights

**[CTVG-4B](https://huggingface.co/Alibaba-VELLDEPTH/CTVG-4B)** is available on
Hugging Face as `Alibaba-VELLDEPTH/CTVG-4B`. It pairs the final 4B grounding
generator with its matching confidence head (200 generator steps and 150 online
head updates).

Verified release revision: `f732bd4e21fa62e63029374b35a36123bc6f8a11`.

Download the complete release from the repository root:

```bash
hf download Alibaba-VELLDEPTH/CTVG-4B \
  --revision f732bd4e21fa62e63029374b35a36123bc6f8a11 \
  --local-dir models/CTVG-4B
```

The release includes the merged BF16 generator, tokenizer, processor, confidence
head in SafeTensors format, head configuration, and `release.json` checksums.
No separate base-model download is required. Keep the whole directory: the public
loader requires `release.json` and both head files. Loading the generator alone
does not apply confidence scoring. Use `scripts/infer.py --model models/CTVG-4B`
with the remaining arguments in the [inference guide](../docs/inference.md).

All 22 published file identities were verified; small files and the confidence
head were downloaded for readback, and generator-shard SHA256 values were checked
against the Hub metadata. `MANIFEST.sha256` matches the published files, including
the Hub-added tokenizer storage rule in `.gitattributes`. This is artifact
validation, not a new GPU inference or benchmark run.

`prepare_model_release.py` assembles the accepted source exports; it is not needed
to use the downloaded model. See [the release guide](../docs/release.md).

Locally trained SFT checkpoints can be evaluated with `infer.py --sft-checkpoint`
using the pinned upstream base; they are not interchangeable with the final RL
release. A separate SFT Hub repository is optional and not published yet.

Model weights are distributed separately from the code repository.
