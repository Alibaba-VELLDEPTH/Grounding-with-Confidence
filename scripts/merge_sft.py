#!/usr/bin/env python3
"""Merge a LOCAL SFT adapter using PEFT 0.17.1; preserve bootstrap-head provenance.

No downloads. The default is a read-only preflight; --execute writes a fresh HF
model directory and copies the head/config byte-for-byte (no pairing relabeling).
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
from prepare_rl import sha

PEFT_VERSION = "0.17.1"
SOFT_PROTOCOL = "timelens2_bounded_coverage_pool_soft_confidence_sft_v1_3"
BASE_REVISION = "ddbb6cb944f13ce21e59e85da23c5f356107260e"


def model_files(directory):
    directory = Path(directory)
    if not (directory / "config.json").is_file() or not list(directory.glob("model*.safetensors")):
        raise ValueError("Expected a local full Hugging Face base/model, not an adapter")
    return {p.name: sha(p) for p in sorted(directory.iterdir()) if p.is_file()
            and p.suffix in {".json", ".safetensors", ".txt", ".jinja"}
            and p.name not in {"_MERGED.json", "release.json"}}


def inspect_sources(base, adapter, head, config):
    base, adapter, head, config = [Path(p).expanduser().resolve(strict=True) for p in (base, adapter, head, config)]
    if head.suffix != ".safetensors":
        raise ValueError("Bootstrap head must be safetensors")
    identity = {"adapter_model_sha256": sha(adapter / "adapter_model.safetensors"),
                "adapter_config_sha256": sha(adapter / "adapter_config.json")}
    head_config = json.loads(config.read_text())
    if any(head_config.get("generator_identity", {}).get(k) != v for k, v in identity.items()):
        raise ValueError("SFT adapter and bootstrap-head pairing mismatch")
    step = None
    state_path = adapter.parent / "trainer_state.json"
    if state_path.is_file():
        step = json.loads(state_path.read_text())["global_step"]
        if type(step) is not int or step <= 0:
            raise ValueError("Invalid SFT trainer step")
    complete_path = adapter.parent / "_COMPLETE"
    if complete_path.is_file():
        complete = json.loads(complete_path.read_text())
        if complete.get("global_step") != step:
            raise ValueError("SFT completion marker step mismatch")
        for path in (adapter / "adapter_model.safetensors", adapter / "adapter_config.json", head, config):
            name = str(path.relative_to(adapter.parent))
            if complete["files"].get(name) != sha(path):
                raise ValueError("SFT marker omits or mismatches paired artifact: " + name)
    return {"base_model_revision": BASE_REVISION, "base_files": model_files(base),
        "adapter": identity, "head_sha256": sha(head), "head_config_sha256": sha(config),
        "sft_protocol": head_config.get("protocol_version"), "sft_target": head_config.get("target"),
        "sft_step": step, "sft_marker_sha256": sha(complete_path) if complete_path.is_file() else None,
        "head_generator_identity": head_config["generator_identity"]}


def verify_merged(model, head, head_sha256):
    model, head = Path(model), Path(head)
    receipt = json.loads((model / "_MERGED.json").read_text())
    if receipt.get("format_version") != 1 or receipt.get("peft_version") != PEFT_VERSION:
        raise ValueError("Expected merge_sft.py PEFT 0.17.1 receipt")
    source = receipt["source_identity"]
    if source["head_sha256"] != head_sha256 or sha(head) != head_sha256:
        raise ValueError("Merged generator/bootstrap head mismatch")
    if source["sft_protocol"] != SOFT_PROTOCOL or source["sft_target"] != "verifier_v2_yes_probability":
        raise ValueError("This RL lineage requires v1.3 soft SFT, not GTbinary")
    if not receipt.get("files"):
        raise ValueError("Empty merged model manifest")
    if any(Path(name).name != name for name in receipt["files"]):
        raise ValueError("Merged model manifest must use flat filenames")
    if model_files(model) != receipt["files"]:
        raise ValueError("Merged model file inventory or identity mismatch")
    if sha(model / "span-head-config.json") != source["head_config_sha256"]:
        raise ValueError("Bootstrap head configuration changed")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True, help="generator-adapter directory")
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--head-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--execute", action="store_true")
    group.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    base, adapter, head, config = [p.expanduser().resolve(strict=True)
        for p in (args.base, args.adapter, args.head, args.head_config)]
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        parser.error("output-dir exists; refusing to overwrite a model")
    identity = inspect_sources(base, adapter, head, config)
    if not args.execute:
        print(json.dumps({"status": "dry_run", "source_identity": identity, "required_peft": PEFT_VERSION}))
        return 0
    if importlib.metadata.version("peft") != PEFT_VERSION:
        raise ValueError("Merging requires peft==" + PEFT_VERSION)
    import os
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
    model = AutoModelForImageTextToText.from_pretrained(base, torch_dtype=torch.bfloat16,
        device_map="cpu", low_cpu_mem_usage=True, attn_implementation="eager", local_files_only=True)
    model = PeftModel.from_pretrained(model, adapter, is_trainable=False, local_files_only=True)
    model = model.merge_and_unload(safe_merge=True)
    if any("lora_" in name for name, _ in model.named_parameters()):
        raise RuntimeError("Adapter parameters remain after merge")
    output.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(output, safe_serialization=True, max_shard_size="4GB")
    AutoProcessor.from_pretrained(base, local_files_only=True).save_pretrained(output)
    AutoTokenizer.from_pretrained(base, local_files_only=True).save_pretrained(output)
    shutil.copyfile(head, output / "span-confidence-head.safetensors")
    shutil.copyfile(config, output / "span-head-config.json")
    if inspect_sources(base, adapter, head, config) != identity:
        raise RuntimeError("Source artifacts changed during merge")
    receipt = {"format_version": 1, "peft_version": PEFT_VERSION, "source_identity": identity,
        "files": model_files(output), "bootstrap_config_copied_unchanged": True,
        "reference_parent": {"protocol": SOFT_PROTOCOL, "step": 1888},
        "note": "Reference lineage metadata is not a claim that caller weights reproduce it."}
    (output / "_MERGED.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": "merged", "output": str(output), "source_identity": identity}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
