from typing import Iterable, Tuple, Dict

import torch
from torch.nn.parameter import Parameter


def clip_grad_norm_local(
        parameters: Iterable[Tuple[str, Parameter]],
        max_norm: float,
        norm_type: float = 2.0
    ) -> None:
    # parameter-wise clip gradient norm
    # code modified from torch.nn.utils.clip_grad_norm_
    parameters = [p for p in parameters if p.grad is not None]
    if len(parameters) == 0:
        return
    
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    
    for p in parameters:
        if norm_type == float("inf"):
            norm = p.grad.detach().abs().max()
        else:
            #norm = torch.linalg.vector_norm(p.grad.detach(), norm_type)    # not supported in pytorch==1.7.1
            norm = torch.norm(p.grad.detach(), norm_type)
        clip_coef = max_norm / (norm + 1e-6)
        # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
        # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
        # when the gradients do not reside in CPU memory.
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        p.grad.detach().mul_(clip_coef_clamped)

def adaptive_gradient_clipping(
        parameters: Iterable[Tuple[str, Parameter]],
        max_norm: float
    ) -> None:
    # from "High-Performance Large-Scale Image Recognition Without Normalization"
    # https://arxiv.org/abs/2102.06171
    parameters = [p for p in parameters if p.grad is not None]
    if len(parameters) == 0:
        return
    
    max_norm = float(max_norm)
