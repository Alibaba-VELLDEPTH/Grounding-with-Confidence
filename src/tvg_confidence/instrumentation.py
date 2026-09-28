from __future__ import annotations
from typing import Any, Dict

def _distribution_stats(logits: Any, selected_token_id: int, top_k: int = 50) -> Dict[str, float]:
    import torch

    values = logits.float()
    log_probs = torch.log_softmax(values, dim=-1)
    probabilities = torch.softmax(values, dim=-1)
    selected_logprob = float(log_probs[int(selected_token_id)].item())
    finite = torch.isfinite(log_probs)
    entropy = float(
        -(probabilities[finite] * log_probs[finite]).sum().item()
    )
    top_values = torch.topk(probabilities, k=min(2, probabilities.numel())).values
    margin = (
        float((top_values[0] - top_values[1]).item())
        if top_values.numel() > 1
        else 1.0
    )
    mass = float(
        torch.topk(probabilities, k=min(top_k, probabilities.numel())).values.sum().item()
    )
    return {
        "selected_logprob": selected_logprob,
        "entropy": entropy,
        "top2_probability_margin": margin,
        "topk_probability_mass": mass,
    }

def _move_batch_to_model(inputs: Any, model: Any) -> Dict[str, Any]:
    import torch

    device = getattr(model, "device", None)
    if device is None or str(device) == "meta":
        device = next(model.parameters()).device
    dtype = getattr(model, "dtype", None)
    moved: Dict[str, Any] = {}
    for key, value in dict(inputs).items():
        if torch.is_tensor(value):
            value = value.to(device)
            if dtype is not None and torch.is_floating_point(value):
                value = value.to(dtype)
        moved[key] = value
    return moved
