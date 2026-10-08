"""Preserve DeepStack residual math when FSDP supplies an autograd view."""
from functools import wraps
import hashlib
import inspect

import torch

EXPECTED_SOURCE = "5ddf273c61e5c85319a7f372a50f021ff103f43207ddcb18d85a5987e33be433"


def install_deepstack_clone():
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

    original = Qwen3VLTextModel._deepstack_process
    if getattr(original, "_tvg_deepstack_clone", False):
        return original
    if hashlib.sha256(inspect.getsource(original).encode()).hexdigest() != EXPECTED_SOURCE:
        raise RuntimeError("unaudited Qwen3-VL DeepStack implementation")

    @wraps(original)
    def safe(self, hidden_states, visual_pos_masks, visual_embeds):
        # The original indexed assignment is invalid on a custom-autograd view.
        # A differentiable clone keeps its backward and the residual values.
        if torch.is_grad_enabled():
            hidden_states = hidden_states.clone()
        return original(self, hidden_states, visual_pos_masks, visual_embeds)

    safe._tvg_deepstack_clone = True
    safe._tvg_original = original
    Qwen3VLTextModel._deepstack_process = safe
    return safe
