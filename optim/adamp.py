"""
Source: https://github.com/clovaai/AdamP (v0.3.0)
AdamP
Copyright (c) 2020-present NAVER Corp.
MIT license
"""

import typing as tp
import math
import torch
from torch import jit, Tensor
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer


@jit.script
def projection_channel(p: Tensor, perturb: Tensor, eps: float) -> Tensor:
    # p, perturb: [C, length]
    p_normalized = p / p.norm(p=2, dim=1, keepdim=True).add(eps)
    perturb_projected = (p_normalized * perturb).sum(dim=1, keepdim=True)
    perturb -= p_normalized * perturb_projected
    return perturb


@jit.script
def projection_layer(p: Tensor, perturb: Tensor, eps: float) -> Tensor:
    # p, perturb: [length]
    p_normalized = p / p.norm(p=2, dim=0, keepdim=True).add(eps)
    perturb_projected = (p_normalized * perturb).sum(dim=0, keepdim=True)
    perturb -= p_normalized * perturb_projected
    return perturb


@jit.script
def projection_dim(p: Tensor, perturb: Tensor, dim: tp.List[int], eps: float) -> Tensor:
    p_normalized = p / p.norm(p=2, dim=dim, keepdim=True).add(eps)
    perturb_projected = (p_normalized * perturb).sum(dim=dim, keepdim=True)
    perturb -= p_normalized * perturb_projected
    return perturb


class AdamP(Optimizer):
    def __init__(
        self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
        weight_decay=0, delta=0.1, wd_ratio=0.1, nesterov=False,
        projection="auto"
    ):
        assert (projection in ["auto", "disabled", "channelwise", "layerwise"]) \
            or isinstance(projection, int), \
            f"`projection` should be 'auto', 'disabled', 'channelwise', 'layerwise' or int," \
            f" but got {projection}"
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            delta=delta, wd_ratio=wd_ratio, nesterov=nesterov,
            projection=projection,
        )
        super(AdamP, self).__init__(params, defaults)

    def project_channelwise(self, p: Tensor, perturb: Tensor, eps: float) -> Tensor:
        x = p.view(p.size(0), -1)
        y = perturb.view(p.size(0), -1)
        perturb = projection_channel(p.data, perturb,  eps)
        return perturb.view_as(p)

    def project_layerwise(self, p: Tensor, perturb: Tensor, eps: float) -> Tensor:
        x = p.view(-1)
        y = perturb.view(-1)
        perturb = projection_layer(p.data, perturb,  eps)
        return perturb.view_as(p)

    def _projection(self, p, grad, perturb, delta, wd_ratio, eps):
        # projection channelwise
        if len(p.shape) > 1:
            x = p.view(p.size(0), -1)
            y = perturb.view(p.size(0), -1)
            cosine_sim = F.cosine_similarity(x, y, dim=1, eps=eps).abs_()
            if cosine_sim.max() < delta / math.sqrt(x.size(1)):
                perturb = projection_channel(p.data, perturb,  eps)
                return perturb.view_as(p), wd_ratio

        # projection layerwise
        x = p.view(-1)
        y = perturb.view(-1)
        cosine_sim = F.cosine_similarity(x, y, dim=0, eps=eps).abs_()
        if cosine_sim.max() < delta / math.sqrt(x.size(0)):
            perturb = projection_layer(p.data, perturb,  eps)
            return perturb.view_as(p), wd_ratio

        # if not scale-invariant, return uncahnged perturb and wd_ratio=1.0
        return perturb, 1.0

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            projection = group['projection']
            eps = group['eps']
            weight_decay = group['weight_decay']
            lr = group['lr']
            beta1, beta2 = group['betas']
            nesterov = group['nesterov']
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                # Adam
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                step_size = lr / bias_correction1

                if nesterov:
                    perturb = (beta1 * exp_avg + (1 - beta1) * grad) / denom
                else:
                    perturb = exp_avg / denom

                # Projection
                if p.numel() == 1 or projection == "disabled":
                    wd_ratio = 1
                elif projection == "auto":
                    perturb, wd_ratio = self._projection(
                        p, grad, perturb, group['delta'],
                        group['wd_ratio'], eps
                    )
                elif projection == "layerwise":
                    perturb = self.project_layerwise(p.data, perturb, eps)
                    wd_ratio = group['wd_ratio']
                elif projection == "channelwise":
                    perturb = self.project_channelwise(p.data, perturb, eps)
                    wd_ratio = group['wd_ratio']
                elif isinstance(projection, int):
                    dim = []
                    for i in range(len(p.shape)):
                        if i != projection:
                            dim.append(i)
                    perturb = projection_dim(p.data, perturb, dim, eps)
                    wd_ratio = group['wd_ratio']
                else:
                    raise RuntimeError(projection)

                # Weight decay
                if weight_decay > 0:
                    p.data.mul_(1 - lr * weight_decay * wd_ratio)

                # Step
                p.data.add_(perturb, alpha=-step_size)

        return loss
