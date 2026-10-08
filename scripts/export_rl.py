"""Export a trusted, self-produced eight-rank FSDP actor and paired online head.

CPU torch.load(weights_only=False) deserializes pickle, which can execute code.
Hashes prove file identity, NOT safety. Never export untrusted checkpoints.
The resulting hf-model/release.json describes these NEW weights, not the paper release.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent / 'rl'))
BASE_REVISION = 'ddbb6cb944f13ce21e59e85da23c5f356107260e'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.partial')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def release_manifest(hf, head_directory, receipt):
    """Flat artifacts accepted directly by scripts/infer.py --model hf-model."""
    from export_head import validate_config
    hf, head_directory = Path(hf), Path(head_directory)
    online = receipt['online_head']
    for name, expected in (('span-confidence-head.safetensors', online['head_sha256']),
                           ('span-head-config.json', online['head_config_sha256'])):
        if sha(head_directory / name) != expected:
            raise ValueError('Exported head changed: ' + name)
        if (hf / name).exists():
            raise ValueError('Refusing to overwrite an existing model artifact')
        shutil.copyfile(head_directory / name, hf / name)
    config_path = hf / 'span-head-config.json'
    config = json.loads(config_path.read_text())
    validate_config(config, online['head_sha256'], receipt)
    config['generator_identity'] = {key: config['generator_identity'][key]
                                   for key in ('step', 'checkpoint_marker_sha256')}
    dump(config_path, config)
    metadata = {'format_version': 1, 'base_model_revision': BASE_REVISION,
        'step': receipt['step'], 'checkpoint_marker_sha256': receipt['checkpoint_marker_sha256'],
        'model_files': receipt['model_files'], 'kind': 'caller_trained_rl',
        'head': {'weights': 'span-confidence-head.safetensors', 'config': 'span-head-config.json',
                 'sha256': online['head_sha256'], 'config_sha256': sha(config_path),
                 'updates': online['updates']},
        'video_contract': {'fps': 2.0, 'min_pixels': 2048, 'total_pixels': 8388608},
        'max_new_tokens': 512, 'context_tokens': 16384, 'nms_iou': 0.3}
    dump(hf / 'release.json', metadata)
    return metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--expected-marker-sha256', required=True)
    p.add_argument('--expected-step', type=int, required=True)
    p.add_argument('--output', '--output-dir', dest='output', type=Path, required=True)
    p.add_argument('--trust-self-produced-checkpoint', action='store_true',
                   help='Acknowledge CPU torch pickle deserialization can execute code')
    p.add_argument('--check-only', action='store_true', help='Verify hashes without torch deserialization')
    args = p.parse_args()
    if not __debug__:
        raise RuntimeError('Do not use Python -O: export integrity assertions are required')
    source, out = args.checkpoint.expanduser().resolve(strict=True), args.output.expanduser().resolve()
    marker = source / '_COMPLETE.json'
    assert sha(marker) == args.expected_marker_sha256
    complete = json.loads(marker.read_text())
    assert complete['step'] == args.expected_step and complete['world_size'] == 8
    assert not out.exists(), 'existing export must be inspected, not overwritten'
    if source in out.parents:
        raise ValueError('Export output must be outside the immutable checkpoint')
    if not complete.get('online_head'):
        raise ValueError('Inference release export requires a paired online head (step > 50)')
    for key in list(os.environ):
        if key.startswith('TVG_RL_'):
            del os.environ[key]
    from runtime import verify_complete
    verify_complete(source, require_online_head=True)
    if args.check_only:
        print(json.dumps({'status': 'checked', 'step': args.expected_step, 'pickle_loaded': False}))
        return 0
    if not args.trust_self_produced_checkpoint:
        p.error('--trust-self-produced-checkpoint is required for torch pickle loading')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    out.mkdir(parents=True)
    from export_head import export_head
    online_head = export_head(source, args.expected_marker_sha256, args.expected_step, out / 'online-head', trusted=True)
    def progress(stage, **extra):
        value = {'stage': stage, 'time': time.time(), **extra}
        dump(out / 'progress.json', value)
        print(json.dumps(value), flush=True)
    staged = out / 'source'
    wanted = {k: v for k, v in complete['files'].items()
              if k.startswith('actor/model_') or k.startswith('actor/huggingface/') or k == 'actor/fsdp_config.json'}
    assert len([k for k in wanted if k.startswith('actor/model_')]) == 8
    progress('stage_inference_files', files=len(wanted))
    def copy_one(item):
        name, identity = item
        src, dst = source / name, staged / name
        assert src.stat().st_size == identity['bytes'], name
        dst.parent.mkdir(parents=True, exist_ok=True)
        digest, count = hashlib.sha256(), 0
        with src.open('rb') as f, dst.open('xb') as g:
            for block in iter(lambda: f.read(8 * 1024**2), b''):
                count += len(block)
                digest.update(block)
                g.write(block)
        assert count == identity['bytes'] and digest.hexdigest() == identity['sha256'], name
        assert sha(dst) == identity['sha256'], name
        return name, identity
    with ThreadPoolExecutor(max_workers=4) as pool:
        verified = dict(pool.map(copy_one, wanted.items()))
    assert sha(marker) == args.expected_marker_sha256, 'source marker changed'
    dump(out / 'checkpoint-input-acceptance.json', {'checkpoint': str(source),
        'marker_sha256': args.expected_marker_sha256, 'step': args.expected_step, 'world_size': 8,
        'inference_files': verified, 'optimizer_loaded': False,
        'source_head_sha256': complete['source_head_sha256']})
    progress('merge_cpu', files=len(verified))
    import torch
    from torch.distributed.tensor import DTensor
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForImageTextToText
    torch.set_num_threads(4)
    actor = staged / 'actor'
    states = [torch.load(actor / f'model_world_size_8_rank_{rank}.pt',
                         map_location='cpu', weights_only=False, mmap=True) for rank in range(8)]
    keys = set(states[0])
    assert all(set(s) == keys for s in states)
    merged, layouts = {}, {}
    for key in sorted(keys):
        values = [s.pop(key) for s in states]
        assert all(isinstance(v, DTensor) for v in values), key
        placements = values[0].placements
        assert len(placements) == 1 and all(v.placements == placements for v in values), key
        assert all(tuple(v.shape) == tuple(values[0].shape) for v in values), key
        assert all(v.device_mesh.mesh.tolist() == list(range(8)) for v in values), key
        local = [v.to_local() for v in values]
        assert all(torch.isfinite(v).all().item() for v in local), key
        place = placements[0]
        if place.is_shard():
            tensor = torch.cat([v.bfloat16() for v in local], dim=place.dim).contiguous()
        elif place.is_replicate():
            assert all(torch.equal(local[0], v) for v in local[1:]), key
            tensor = local[0].bfloat16().clone().contiguous()
        else:
            raise ValueError(f'unsupported placement {place}: {key}')
        assert tuple(tensor.shape) == tuple(values[0].shape), key
        merged[key] = tensor
        layouts[key] = {'shape': list(tensor.shape), 'placement': str(place)}
    del states, values, local
    gc.collect()
    config = AutoConfig.from_pretrained(actor / 'huggingface', local_files_only=True)
    with init_empty_weights():
        model = AutoModelForImageTextToText.from_config(config, torch_dtype=torch.bfloat16)
    expected = model.state_dict()
    assert set(expected) == set(merged), {'missing': sorted(set(expected) - set(merged)),
                                        'extra': sorted(set(merged) - set(expected))}
    assert all(tuple(expected[k].shape) == tuple(v.shape) for k, v in merged.items())
    assert not any('lora_' in k for k in merged)
    hf = out / 'hf-model'
    progress('save_hf', tensors=len(merged), tensor_elements=sum(v.numel() for v in merged.values()))
    model.save_pretrained(hf, state_dict=merged, safe_serialization=True, max_shard_size='4GB')
    for path in (actor / 'huggingface').iterdir():
        if path.is_file() and path.name != 'config.json':
            shutil.copyfile(path, hf / path.name)
    del model, expected, merged
    gc.collect()
    from prepare_model_release import portable_config
    for path in hf.glob('*config.json'):
        dump(path, portable_config(json.loads(path.read_text())))
    model_files = {p.name: sha(p) for p in sorted(hf.iterdir()) if p.is_file()}
    receipt = {'status': 'complete', 'checkpoint': str(source), 'step': args.expected_step, 'world_size': 8,
        'checkpoint_marker_sha256': args.expected_marker_sha256, 'model_path': str(hf),
        'model_files': model_files, 'source_head_sha256': complete['source_head_sha256'],
        'source_files': verified, 'tensor_layouts': layouts, 'torch': torch.__version__,
        'inference_dtype': 'bfloat16', 'adapter_loaded': False, 'optimizer_steps': 0,
        'exporter_sha256': sha(Path(__file__)), 'time': time.time()}
    if online_head is not None:
        receipt['online_head'] = online_head
        receipt['online_head_export_receipt_sha256'] = sha(out / 'online-head/export-receipt.json')
        receipt['source_head_semantics'] = 'bootstrap SFT head; use online_head for this actor checkpoint'
    if complete.get('experiment') is not None:
        receipt['training_experiment'] = complete['experiment']
    assert sha(marker) == args.expected_marker_sha256, 'checkpoint marker changed during actor export'
    release_manifest(hf, out / 'online-head', receipt)
    dump(out / 'export-receipt.json', receipt)
    dump(out / '_SUCCESS', {'export_receipt_sha256': sha(out / 'export-receipt.json')})
    # Preserve the source checkpoint and compact provenance, not duplicate local shards.
    shutil.rmtree(staged)
    progress('complete', receipt_sha256=sha(out / 'export-receipt.json'))


if __name__ == '__main__':
    main()
