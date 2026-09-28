<div align="center">

# Grounding with Confidence

### Controllable Generative Video Temporal Grounding

**Generate once. Rank, select, or reject.**

[Overview](#overview) · [Results](#results) · [Models](#models) · [Quick Start](#quick-start) · [Evaluation](#evaluation)

</div>

Official implementation of **Grounding with Confidence: Controllable Generative Video Temporal Grounding**.

> **Training code and data are coming soon.** Model weights will be released on Hugging Face. Inference and evaluation code are included in this repository.

## Overview

Generating candidate intervals and deciding which ones to return are distinct tasks. **Grounding with Confidence** equips generative video temporal grounding with an interval-level confidence head, allowing the same decoded candidates to serve different application requirements.

- **Score in the original grounding pass.** A lightweight head reads cached decoder states, without adding confidence tokens or invoking an external verifier at inference.
- **Choose what to return.** Use confidence for ranking, threshold-based selection, or rejection, without generating the candidates again.
- **Learn generation and selection.** Distinct supervision trains the generator and confidence head, while set-level reinforcement learning rewards candidate coverage and the quality of the selected set.

<p align="center">
  <img src="assets/method_overview.png" width="100%" alt="Method overview: candidate generation and confidence scoring share one decoding pass; generation-order NMS and a confidence threshold determine the returned intervals.">
</p>

## Results

**Learning what to return increases the value of a single generation.** On OMTG-Bench, confidence improves selection from the same candidate pool, with the largest gains under tight return budgets.

Official query-macro **Recall@0.5 (%)** on all 320 queries, using the same 1,314 post-NMS candidates:

| Global return budget | Generation order | Token likelihood | External verifier | **Confidence (ours)** |
| :---: | :---: | :---: | :---: | :---: |
| 10% | 9.95 | 9.41 | 11.00 | **14.42** |
| 25% | 26.48 | 22.37 | 26.31 | **31.12** |
| 50% | 51.34 | 43.05 | 47.72 | **52.82** |
| 75% | 64.76 | 61.00 | 62.45 | **65.30** |

The return budget is shared across queries; each selector returns the same total number of intervals. All candidate intervals remain fixed.

## Models

| Model | Base model | Hugging Face |
| --- | --- | --- |
| Grounding-with-Confidence-4B | [TimeLens2-4B](https://huggingface.co/MCG-NJU/TimeLens2-4B) | Coming soon |

The checkpoint includes the grounding generator and its matched confidence head. See [model details](models/README.md).

## Quick Start

### Installation

Use Python 3.10+ and a CUDA GPU. From the repository root:

```bash
python -m pip install -e .
python -m pip install -r requirements.txt
```

### Inference

Once the weights are available, prepare a JSONL file following [examples/requests.jsonl](examples/requests.jsonl), using your own video path, query, and video duration. Then run:

```bash
python scripts/infer.py \
  --model /path/to/downloaded-model \
  --input examples/requests.jsonl \
  --output-dir outputs/example \
  --threshold 0.5
```

The output contains candidate intervals, their confidence scores, and the selected intervals. Changing the threshold changes the accepted set; it does not require another decoding pass. The example uses a cutoff of 0.5; choose a threshold on validation data for your application.

See the [inference guide](docs/inference.md) for input formats, decoding settings, and Hugging Face model loading.

## Evaluation

Evaluate saved candidates at a fixed confidence threshold:

```bash
python scripts/evaluate.py \
  --input examples/predictions.jsonl \
  --threshold 0.5
```

The included synthetic example runs on CPU. For benchmark evaluation, provide predictions with ground-truth intervals. The evaluator reports query-macro precision, recall, and tF1 at multiple IoU thresholds, together with count accuracy, temporal overlap, and EtF1. See the [evaluation guide](docs/inference.md#evaluate-saved-predictions-cpu) for metric definitions.

## Training and Data

Training code, training data, and data preparation tools are **coming soon**.

## Acknowledgements

Our implementation builds on [TimeLens2](https://github.com/MCG-NJU/TimeLens2) and [Hugging Face Transformers](https://github.com/huggingface/transformers). We thank their authors for sharing these tools and models.
