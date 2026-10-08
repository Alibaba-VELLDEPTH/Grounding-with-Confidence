"""Change actor LR once, after a complete parent optimizer/scheduler restore."""
import hashlib
import json
import math
import os
from pathlib import Path
import time


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def check_lrs(optimizer, scheduler, expected):
    groups = optimizer.param_groups
    values = [g['lr'] for g in groups] + [g['initial_lr'] for g in groups]
    values += list(scheduler.base_lrs) + list(scheduler.get_last_lr())
    if not groups or len(scheduler.base_lrs) != len(groups):
        raise RuntimeError('actor optimizer/scheduler group mismatch')
    if not all(math.isclose(float(x), expected, rel_tol=1e-10, abs_tol=0) for x in values):
        raise RuntimeError('restored actor learning rates differ from the frozen phase')
    # This experiment uses the existing constant scheduler after its warmup.
    if scheduler.__class__.__name__ != 'LambdaLR' or scheduler.last_epoch < 5:
        raise RuntimeError('actor LR phase requires the accepted post-warmup LambdaLR')
    if not all(fn(epoch) == 1.0 for fn in scheduler.lr_lambdas
               for epoch in (scheduler.last_epoch, scheduler.last_epoch + 1)):
        raise RuntimeError('actor LR phase requires a constant current/next multiplier')


def restore(optimizer, scheduler, actor_path, rank, phase=None, experiment=None, output=None):
    phase = phase if phase is not None else json.loads(os.environ['TVG_RL_ACTOR_LR_PHASE'])
    experiment = experiment if experiment is not None else json.loads(os.environ['TVG_RL_EXPERIMENT_IDENTITY'])
    if experiment.get('actor_lr_phase_sha256') != canonical_sha(phase):
        raise RuntimeError('actor LR phase identity mismatch')
    directory = Path(actor_path).parent
    marker_bytes = (directory / '_COMPLETE.json').read_bytes()
    marker = json.loads(marker_bytes)
    marker_sha = hashlib.sha256(marker_bytes).hexdigest()
    if marker['world_size'] != 8 or not marker['online_head']:
        raise RuntimeError('actor LR phase requires a complete eight-rank online-head checkpoint')
    parent = marker_sha == phase['parent_marker_sha256']
    if parent:
        if marker['step'] != phase['start_step']:
            raise RuntimeError('actor LR parent step mismatch')
        expected = phase['parent_actor_lr']
    else:
        if marker.get('experiment') != experiment or marker['step'] <= phase['start_step']:
            raise RuntimeError('actor LR resume is neither the accepted parent nor this experiment')
        expected = phase['actor_lr']
    check_lrs(optimizer, scheduler, expected)
    before = scheduler.state_dict()
    if parent:
        for group in optimizer.param_groups:
            group['lr'] = phase['actor_lr']
            group['initial_lr'] = phase['actor_lr']
        scheduler.base_lrs = [phase['actor_lr']] * len(optimizer.param_groups)
        scheduler._last_lr = [phase['actor_lr']] * len(optimizer.param_groups)
    check_lrs(optimizer, scheduler, phase['actor_lr'])
    after = scheduler.state_dict()
    changed_fields = {'base_lrs', '_last_lr'}
    if {k: v for k, v in before.items() if k not in changed_fields} != {
            k: v for k, v in after.items() if k not in changed_fields}:
        raise RuntimeError('actor LR phase modified scheduler progression')
    result = {'status': 'passed', 'rank': rank, 'time': time.time(),
        'checkpoint': str(directory), 'checkpoint_step': marker['step'],
        'checkpoint_marker_sha256': marker_sha, 'phase_sha256': canonical_sha(phase),
        'action': 'parent_lr_override' if parent else 'own_checkpoint_lr_unchanged',
        'previous_actor_lr': expected, 'actor_lr': phase['actor_lr'],
        'scheduler_last_epoch': scheduler.last_epoch, 'scheduler_step_count': scheduler._step_count,
        'optimizer_groups': len(optimizer.param_groups), 'experiment': experiment}
    destination = Path(output if output is not None else os.environ['TVG_RL_OUTPUT'])
    from runtime import write_json
    write_json(destination / f'actor-lr-phase-rank{rank}.json', result)
    return result
