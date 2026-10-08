"""Actual-input budget checks and complete-checkpoint receipts for GPU GRPO."""
import fcntl
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

CONTRACT = {"fps": 2.0, "min_pixels": 2048, "total_pixels": 8388608}


@lru_cache(maxsize=4096)
def decoded_frame_count(path, source_size, source_mtime_ns):
    """Count frames when the container omits nb_frames; cache by source identity."""
    import av
    if Path(path).stat().st_size != source_size:
        raise RuntimeError("cached video size differs from source")
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        return sum(1 for _ in container.decode(stream))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temp, path)


def cached_video(video):
    if {k: video.get(k) for k in CONTRACT} != CONTRACT or any(
            k in video for k in ("nframes", "max_frames", "max_pixels", "resized_height", "resized_width")):
        raise ValueError("rollout video budget drift")
    source = Path(video["video"])
    cache = Path(os.environ["TVG_RL_CACHE"]) / "videos"
    cache.mkdir(parents=True, exist_ok=True)
    source_sha256 = sha(source)
    dest = cache / (source_sha256 + source.suffix)
    with (cache / "cache.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not dest.exists() or sha(dest) != source_sha256:
            temporary = dest.with_name(dest.name + ".tmp")
            shutil.copyfile(source, temporary)
            if sha(temporary) != source_sha256:
                temporary.unlink()
                raise RuntimeError("source video changed during cache copy")
            os.replace(temporary, dest)
        dest.touch()
        # Only the single driver-side dataloader uses this cache; vLLM receives decoded arrays.
        files = sorted(cache.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
        size = sum(p.stat().st_size for p in files)
        for old in files:
            if size <= 32 * 1024**3:
                break
            if old != dest:
                size -= old.stat().st_size
                old.unlink()
    return {**video, "video": str(dest)}


def validate_video(processor, videos, metadata, inputs, row, source_videos):
    import av
    from video_budget import runtime_contract
    from qwen_vl_utils.vision_process import smart_nframes
    runtime_contract()
    if len(videos) != 1 or len(metadata) != 1:
        raise RuntimeError("expected one full video")
    tensor, meta = videos[0], metadata[0]
    get = lambda key: meta[key] if isinstance(meta, dict) else getattr(meta, key)
    shape = list(tensor.shape)
    indices = list(map(int, get("frames_indices")))
    source_frames, source_fps = int(get("total_num_frames")), float(get("fps"))
    # OMTG uses container duration; TL2 annotations may use frame count / fps.
    # Compare each against independently read metadata without relaxing tolerance.
    if len(source_videos) != 1:
        raise RuntimeError("expected one source video")
    local_video = cached_video(source_videos[0])["video"]
    with av.open(local_video) as container:
        stream = container.streams.video[0]
        stream_duration = float(stream.duration * stream.time_base) if stream.duration is not None else None
        indexed_basis_duration = (float(container.duration / av.time_base) if container.duration is not None
            else stream_duration if stream_duration is not None else float(stream.frames / stream.average_rate))
        header_frames, header_fps = stream.frames, float(stream.average_rate) if stream.average_rate else None
    verified_frames = header_frames
    frame_count_basis = "container_header"
    if not header_frames:
        # Fragmented MP4 may expose nb_frames=0 even though all frames decode.
        # Independently count them; never accept unknown or mismatching counts.
        source_stat = Path(source_videos[0]["video"]).stat()
        verified_frames = decoded_frame_count(local_video, source_stat.st_size, source_stat.st_mtime_ns)
        frame_count_basis = "pyav_full_decode_missing_header"
    annotation_duration = row["extra_info"]["duration_s"]
    duration_basis = "container"
    reference_duration = indexed_basis_duration
    if (row.get("data_source") == "tl2_single_rl"
            and abs(reference_duration - annotation_duration) >= 0.01
            and verified_frames and header_fps):
        reference_duration = verified_frames / header_fps
        duration_basis = "independently_verified_frames_over_fps"
    grid = inputs["video_grid_thw"][0].tolist()
    video_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    actual = int((inputs["input_ids"] == video_id).sum())
    checks = {
        "sampled_frames": len(indices) == shape[0] == smart_nframes(CONTRACT, source_frames, source_fps),
        "full_video": indices[0] == 0 and indices[-1] == source_frames - 1,
        "grid": grid == [math.ceil(shape[0]/2), shape[-2]//16, shape[-1]//16],
        "merge": shape[-2] % 32 == shape[-1] % 32 == 0,
        "minimum": shape[-2] * shape[-1] >= 2048,
        "total": grid[0] * shape[-2] * shape[-1] <= 8388608,
        "tokens": actual == math.prod(grid)//4,
        "context": inputs["input_ids"].shape[-1] + 512 <= 16384,
        "source_duration": abs(reference_duration - annotation_duration) < 0.01,
        "source_frame_count": bool(verified_frames) and verified_frames == source_frames,
        "source_fps": header_fps is not None and math.isclose(header_fps, source_fps, rel_tol=1e-6),
    }
    observation = {"phase": os.getenv("TVG_RL_INPUT_PHASE", "training"),
        "query_id": row["extra_info"]["query_id"], "frames": shape[0],
        "grid": grid, "visual_tokens": actual, "input_tokens": inputs["input_ids"].shape[-1],
        "indexed_container_duration_s": row["extra_info"]["duration_s"],
        "actual_container_duration_s": indexed_basis_duration, "actual_stream_duration_s": stream_duration,
        "annotation_duration_s": annotation_duration, "source_duration_basis": duration_basis,
        "verified_reference_duration_s": reference_duration,
        "actual_decoded_duration_s": source_frames / source_fps,
        "source_frames": source_frames, "source_fps": source_fps,
        "header_frame_count": header_frames, "verified_frame_count": verified_frames,
        "source_frame_count_basis": frame_count_basis, "checks": checks,
        "all_checks": all(checks.values())}
    if not all(checks.values()):
        print(json.dumps({"stage": "visual_budget_failed", **observation}), flush=True)
        raise RuntimeError("actual visual budget failed: " + json.dumps(observation))
    log = Path(os.environ["TVG_RL_OUTPUT"]) / "visual-budget.jsonl"
    with log.open("a") as f:
        f.write(json.dumps(observation) + "\n")


def checkpoint_receipt(root, step):
    root = Path(root)
    directory = root / f"global_step_{step}"
    required = [directory / "data.pt"]
    for rank in range(8):
        required.extend(directory / "actor" / f"{kind}_world_size_8_rank_{rank}.pt"
                        for kind in ("model", "optim", "extra_state"))
    online_head = os.environ.get("TVG_RL_TRAIN_HEAD", "0") == "1"
    if online_head:
        required.extend(directory / "actor" / name for name in (
            "span-confidence-head.safetensors", "span-head-training-state.pt", "span-head-replica-receipt.json"))
    from checkpoint_io import hash_checkpoint_files, wait_for_checkpoint_size
    for path in required:
        wait_for_checkpoint_size(path)
    names = sorted({str(p.relative_to(directory)) for p in required} |
        {str(p.relative_to(directory)) for p in directory.rglob("*")
         if p.is_file() and p.name != "_COMPLETE.json"})
    files = hash_checkpoint_files(directory, names, phase="save_receipt")
    write_json(directory / "_COMPLETE.json", {"step": step, "world_size": 8,
        "files": files, "source_head_sha256": os.environ["TVG_RL_HEAD_SHA"],
        "online_head": online_head, "time": time.time(),
        "experiment": json.loads(os.environ["TVG_RL_EXPERIMENT_IDENTITY"]),
        "first_checkpoint_step": int(os.environ["TVG_RL_FIRST_CHECKPOINT_STEP"])})


def verify_complete(directory, require_online_head=False, experiment=None):
    directory = Path(directory)
    receipt = json.loads((directory / "_COMPLETE.json").read_text())
    if receipt["step"] != int(directory.name.removeprefix("global_step_")) or receipt["world_size"] != 8:
        raise RuntimeError("checkpoint step/world size mismatch")
    if experiment is not None and receipt.get("experiment") != experiment:
        raise RuntimeError("checkpoint belongs to a different RL experiment")
    if (experiment is not None
            and receipt.get("first_checkpoint_step") != experiment["first_checkpoint_step"]):
        raise RuntimeError("checkpoint first-save contract mismatch")
    required = ["data.pt"] + [f"actor/{kind}_world_size_8_rank_{rank}.pt"
        for rank in range(8) for kind in ("model", "optim", "extra_state")]
    if require_online_head or receipt.get("online_head"):
        if not receipt.get("online_head", False):
            raise RuntimeError("online-head run cannot resume a checkpoint missing the trained head")
        required.extend("actor/" + name for name in ("span-confidence-head.safetensors",
            "span-head-training-state.pt", "span-head-replica-receipt.json"))
    if any(name not in receipt["files"] for name in required):
        raise RuntimeError("checkpoint receipt omits required continuation state")
    from checkpoint_io import hash_checkpoint_files
    hash_checkpoint_files(directory, receipt["files"], expected=receipt["files"], phase="resume_verify")
    return receipt


def latest_complete(root, require_online_head=False):
    root = Path(root)
    complete = sorted(root.glob("global_step_*/_COMPLETE.json"),
                      key=lambda p: int(p.parent.name.split("_")[-1]), reverse=True)
    if not complete:
        return None
    # Never silently select another experiment or roll back past a partial save.
    marker = complete[0]
    if any(int(p.name.removeprefix("global_step_")) > int(marker.parent.name.removeprefix("global_step_"))
           for p in root.glob("global_step_*") if p.is_dir()):
        raise RuntimeError("a newer incomplete checkpoint requires inspection")
    verify_complete(marker.parent, require_online_head=require_online_head,
        experiment=json.loads(os.environ["TVG_RL_EXPERIMENT_IDENTITY"]))
    return str(marker.parent)


def first_online_checkpoint(step):
    return (os.environ.get("TVG_RL_TRAIN_HEAD", "0") == "1"
        and step == int(os.environ.get("TVG_RL_FIRST_CHECKPOINT_STEP",
            str(int(os.environ["TVG_RL_HEAD_BOOTSTRAP_STEP"]) + 1))))


def worker_environment():
    """Explicit Ray runtime environment; avoid depending on worker inheritance."""
    names = {"PYTHONPATH", "TMPDIR", "HF_HOME", "HF_DATASETS_CACHE", "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "TOKENIZERS_PARALLELISM",
        "VLLM_WORKER_MULTIPROC_METHOD", "FORCE_QWENVL_VIDEO_READER", "OMP_NUM_THREADS",
        "RAY_DEDUP_LOGS", "WANDB_MODE", "PYTHONDONTWRITEBYTECODE"}
    return {k: v for k, v in os.environ.items() if k in names or k.startswith("TVG_RL_")}


def verify_patched_imports():
    """Run before verl imports in the local driver AND remote TaskRunner."""
    import importlib.util
    from patch_verl import verify_copy
    root = verify_copy(Path(os.environ["TVG_RL_UPSTREAM_COPY"]))
    spec = importlib.util.find_spec("verl")
    expected = root / "rl/verl/verl/__init__.py"
    if spec is None or Path(spec.origin).resolve() != expected.resolve():
        raise RuntimeError("installed/unpatched verl would shadow the verified cache copy")


def log_metrics(metrics, step):
    from numbers import Real
    clean = {k: float(v) for k, v in metrics.items() if isinstance(v, Real)}
    path = Path(os.environ["TVG_RL_OUTPUT"]) / "training-metrics.jsonl"
    with path.open("a") as f:
        f.write(json.dumps({"step": step, "time": time.time(), **clean}) + "\n")
