"""Appendix-E visual parameters with an independently recorded TL2 processor path.

This aligns the three published visual parameters, not the author's entire
inference environment. The historical 120-frame core remains unchanged.
"""
from functools import lru_cache
import hashlib
import importlib.metadata
import math
from pathlib import Path


PROTOCOL = "timelens2_omtg_appendix_e_visual_budget_v1"
VIDEO_CONTRACT = {"fps": 2.0, "min_pixels": 2048, "total_pixels": 8_388_608}
VISION_SOURCE_SHA256 = "f3710ced8ffe735da57d08cf5236cf9d326ff646ed1bdad2e62593a7801adc33"
CONTEXT_TOKENS = 16_384
COMPLETION_TOKENS = 512


@lru_cache(maxsize=1)
def runtime_contract():
    import qwen_vl_utils.vision_process as vision

    source_sha = hashlib.sha256(Path(vision.__file__).read_bytes()).hexdigest()
    if source_sha != VISION_SOURCE_SHA256:
        raise RuntimeError("qwen video implementation differs from the audited source")
    versions = {name: importlib.metadata.version(name)
                for name in ("qwen-vl-utils", "transformers", "torch")}
    if versions["qwen-vl-utils"] != "0.0.14" or versions["transformers"] != "4.57.1":
        raise RuntimeError(f"unaudited processor versions: {versions}")
    return {
        "versions": versions,
        "vision_source_sha256": source_sha,
        "frame_factor": vision.FRAME_FACTOR,
        "spatial_merge_size": vision.SPATIAL_MERGE_SIZE,
        "default_min_frames": vision.FPS_MIN_FRAMES,
        "default_max_frames": vision.FPS_MAX_FRAMES,
        "default_max_frames_source": "installed utility default; not specified in Appendix E",
        "entry_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def prepare_inputs(processor, row):
    from qwen_vl_utils import process_vision_info
    from qwen_vl_utils.vision_process import smart_nframes

    runtime = runtime_contract()
    if row["video_contract"] != VIDEO_CONTRACT:
        raise ValueError("row does not use the frozen Appendix-E three-parameter contract")
    prompt = str(row["prompt_text"])
    messages = [{"role": "user", "content": [
        {"type": "video", "video": str(row["video_path"]), **VIDEO_CONTRACT},
        {"type": "text", "text": prompt},
    ]}]
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    patch = int(processor.image_processor.patch_size)
    merge = int(processor.video_processor.merge_size)
    temporal = int(processor.video_processor.temporal_patch_size)
    if (patch, merge, temporal, runtime["frame_factor"]) != (16, 2, 2, 2):
        raise RuntimeError("unexpected TL2 visual patch geometry")
    images, packed_videos, kwargs = process_vision_info(
        messages, image_patch_size=patch, return_video_kwargs=True, return_video_metadata=True)
    if images is not None or packed_videos is None or len(packed_videos) != 1:
        raise RuntimeError("expected exactly one video and no extra visual inputs")
    videos, metadata = zip(*packed_videos)
    shape = [int(v) for v in videos[0].shape]
    meta = metadata[0]
    value = lambda name: meta[name] if isinstance(meta, dict) else getattr(meta, name)
    indices = [int(v) for v in value("frames_indices")]
    source_fps = float(value("fps"))
    source_frames = int(value("total_num_frames"))
    expected_frames = smart_nframes(VIDEO_CONTRACT, source_frames, source_fps)
    inputs = processor(text=[rendered], images=images, videos=list(videos),
        video_metadata=list(metadata), do_resize=False, return_tensors="pt", **(kwargs or {}))
    grid_tensor = inputs.get("video_grid_thw")
    if grid_tensor is None or list(grid_tensor.shape) != [1, 3]:
        raise RuntimeError("unexpected video grid")
    grid = [int(v) for v in grid_tensor[0].tolist()]
    visual_tokens = math.prod(grid) // (merge * merge)
    # fetch_video budgets area after temporal grouping, not the sum of RGB areas.
    native_budget_area = grid[0] * shape[-2] * shape[-1]
    token_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    actual_video_tokens = int((inputs["input_ids"] == token_id).sum().item())
    checks = {
        "exact_smart_nframes": len(indices) == expected_frames == shape[0],
        "full_video_temporal_sampling": indices[0] == 0 and indices[-1] == source_frames - 1,
        "monotone_indices": all(a <= b for a, b in zip(indices, indices[1:])),
        "grid_matches_resized_video": grid == [math.ceil(shape[0] / temporal), shape[-2] // patch, shape[-1] // patch],
        "spatial_merge_aligned": shape[-2] % (patch * merge) == shape[-1] % (patch * merge) == 0,
        "minimum_area": shape[-2] * shape[-1] >= VIDEO_CONTRACT["min_pixels"],
        "native_total_budget": native_budget_area <= VIDEO_CONTRACT["total_pixels"],
        "video_token_count": actual_video_tokens == visual_tokens,
        "context_with_completion": int(inputs["input_ids"].shape[-1]) + COMPLETION_TOKENS <= CONTEXT_TOKENS,
    }
    contract = {
        "protocol_version": PROTOCOL, "requested": dict(VIDEO_CONTRACT), "runtime": runtime,
        "source_fps": source_fps, "source_frame_count": source_frames,
        "sampled_frames": len(indices), "sampled_source_frame_indices": indices,
        "source_duration_s": source_frames / source_fps,
        "expected_sampled_frames_from_smart_nframes": expected_frames,
        "resized_video_shape_tchw": shape, "video_grid_thw": [grid],
        "visual_tokens": visual_tokens, "actual_video_pad_tokens": actual_video_tokens,
        "raw_rgb_frame_area_sum": shape[0] * shape[-2] * shape[-1],
        "native_temporal_grouped_area": native_budget_area,
        "native_budget_utilization": native_budget_area / VIDEO_CONTRACT["total_pixels"],
        "input_tokens": int(inputs["input_ids"].shape[-1]),
        "completion_tokens": COMPLETION_TOKENS, "context_tokens": CONTEXT_TOKENS,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "video_kwargs": kwargs, "checks": checks,
        "author_environment_fully_reproduced": False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Appendix-E processor check failed: {contract}")
    return inputs, rendered, contract


def install_into_core(core, expected_core_sha256):
    """Attach only preprocessing; same-decode confidence and candidate stops stay in core."""
    if hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest() != expected_core_sha256:
        raise RuntimeError("historical core SHA mismatch")
    runtime_contract()

    def prepare(processor, model, row, **flags):
        if any(flags.values()):
            raise ValueError("explicit Yes/No coordinate formats are outside this entry")
        inputs, rendered, contract = prepare_inputs(processor, row)
        return core._move_batch_to_model(inputs, model), rendered, contract

    core.EXPECTED_VIDEO_CONTRACT = dict(VIDEO_CONTRACT)
    core.prepare = prepare
