#!/usr/bin/env python3
"""Assemble the matched generator and confidence head for a model release."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tvg_confidence.common import sha256_file
from head_checkpoint import validate_config


def portable_config(value):
    if isinstance(value, dict):
        return {key: portable_config(item) for key, item in value.items()
                if key not in {"_name_or_path", "name_or_path", "_commit_hash", "cache_dir"}}
    if isinstance(value, list):
        return [portable_config(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--head-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("output-dir already exists; choose a new directory")
    release = json.loads((ROOT / "configs/release-model.json").read_text())
    for name, expected in release["model_files"].items():
        if sha256_file(args.model_dir / name) != expected:
            raise ValueError(f"Generator checksum mismatch: {name}")
    head_path = args.head_dir / release["head"]["weights"]
    if sha256_file(head_path) != release["head"]["sha256"]:
        raise ValueError("Confidence-head checksum mismatch")
    original = json.loads((args.head_dir / release["head"]["config"]).read_text())
    required = (
        "protocol_version", "training_contract", "target", "pooling", "semantic_roles",
        "head_architecture", "generator_confidence_stop_gradient",
        "generator_frozen_during_training", "head_updates", "bootstrap_step", "head_sha256",
    )
    head_config = {key: original[key] for key in required}
    head_config["generator_identity"] = {
        key: original["generator_identity"][key]
        for key in ("step", "checkpoint_marker_sha256")
    }
    validate_config(head_config, release["head"]["sha256"], {
        "status": "complete", "adapter_loaded": False,
        "step": release["step"],
        "checkpoint_marker_sha256": release["checkpoint_marker_sha256"],
        "online_head": {"updates": release["head"]["updates"],
                        "head_sha256": release["head"]["sha256"]},
    })
    args.output_dir.mkdir(parents=True)
    for name in release["model_files"]:
        source, target = args.model_dir / name, args.output_dir / name
        if name.endswith("config.json"):
            value = portable_config(json.loads(source.read_text()))
            target.write_text(json.dumps(value, indent=2) + "\n")
        else:
            shutil.copy2(source, target)
        release["model_files"][name] = sha256_file(target)
    shutil.copy2(head_path, args.output_dir / head_path.name)
    config_path = args.output_dir / release["head"]["config"]
    config_path.write_text(json.dumps(head_config, indent=2) + "\n")
    release["head"]["config_sha256"] = sha256_file(config_path)
    (args.output_dir / "release.json").write_text(json.dumps(release, indent=2) + "\n")
    shutil.copy2(ROOT / "docs/model-card.md", args.output_dir / "README.md")
    (args.output_dir / ".gitattributes").write_text("*.safetensors filter=lfs diff=lfs merge=lfs -text\n")
    print(f"Prepared model files in {args.output_dir}; hosting metadata remains to be finalized.")


if __name__ == "__main__":
    main()
