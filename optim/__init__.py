from typing import Dict, List, Tuple, Any
import re

import torch
from torch import optim
from torch.optim.lr_scheduler import _LRScheduler

from .adamp import AdamP
from .lr_scheduler import EmptyScheduler, CosineAnnealingWarmupRestarts, ReduceLROnPlateau, \
    CosineAnnealingWarmup
from utils import HParams, verbose as _verbose


def partition_param_group(
    group: Dict[str, Any],
    regex_list: List[str],
    kwargs: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    input:
        group = dict(
            named_parameters={...},
            **group_kwargs
        )
        regex_list: list of regex strings
        kwargs: dict of kwargs to update group
    output:
        groups containing parameters whose names don't match regex_list
        groups containing parameters whose names match regex_list
    """
    named_parameters = group.pop("named_parameters")
    group["named_parameters"] = {}
    new_group = group.copy()
    new_group["named_parameters"] = {}
    new_group.update(kwargs)

    for name, param in named_parameters.items():
        if any(re.search(regex, name) is not None for regex in regex_list):
            new_group["named_parameters"][name] = param
        else:
            group["named_parameters"][name] = param
    return group, new_group


class Colors:
    PURPLE = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def update_param_groups(
    model: torch.nn.Module,
    hps_optim_groups: List[Dict[str, Any]],
    verbose: bool = True,
):
    ''' [example]
    inputs:
        model = weight_norm(nn.Linear(5, 10))
            -> model.named_parameters = dict(
                bias=[10],
                weight_v=[10, 5],
                weight_g=[10, 1],
            )
        
        hps_optim_groups = [
            dict(
                regex_list=[".+weight_v"],
                weight_decay=0.0,
            ),
            dict(
                regex_list=["weight"],
                lr=0.1,
            )
        ]
        
    outputs:
        - dict(params=["bias"])
        - dict(params=["weight_g"], lr=0.1)
        - dict(params=[], weight_decay=0.0)
        - dict(params=["weight_v"], weight_decay=0.0, lr=0.1)
    '''
    # partition param groups according to regex_list
    groups = [dict(named_parameters=dict(model.named_parameters()))]
    for hp in hps_optim_groups:
        new_groups = []
        regex_list = hp.pop("regex_list")
        for group in groups:
            group1, group2 = partition_param_group(group, regex_list, hp)
            new_groups.append(group1)
            new_groups.append(group2)
        groups = new_groups
        hp["regex_list"] = regex_list   # restore regex_list

    # print
    if verbose and _verbose():
        for group in groups:
            if not group["named_parameters"]:
                continue
            print(f"{Colors.GREEN}[", end="")
            for key, value in group.items():
                if key == "named_parameters":
                    continue
                print(f"{key}: {value}, ", end="")
            print(f"]{Colors.ENDC} ", end="")
            for name in group["named_parameters"]:
                print(f"{name}, ", end="")
            print("")

    # convert "named_parameters" to "params"
    final_groups = []
    for group in groups:
        named_parameters = group.pop("named_parameters")
        new_group = dict(
            params=[param for param in named_parameters.values()],
            **group
        )
        final_groups.append(new_group)

    return final_groups


def get_optimizer(model: torch.nn.Module, hp: HParams) -> torch.optim.Optimizer:
    optimizer_name = hp.optimizer
    if optimizer_name == "AdamP":
        optimizer = AdamP
    else:
        optimizer = getattr(torch.optim, optimizer_name)

    if hasattr(hp, "optimizer_groups"):
        params = update_param_groups(model, hp.optimizer_groups)
    else:
        params = model.parameters()

    return optimizer(params, **hp.optimizer_kwargs)


def get_scheduler(optimizer: optim.Optimizer, hp: HParams) -> _LRScheduler:
    scheduler_name = hp.scheduler
    if scheduler_name == "EmptyScheduler" or scheduler_name is None:
        return EmptyScheduler()
    elif scheduler_name == "CosineAnnealingLR":
        if "T_max" not in hp.scheduler_kwargs:
            hp.scheduler_kwargs["T_max"] = hp.max_epochs
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, **hp.scheduler_kwargs)
    elif scheduler_name == "CosineAnnealingWarmup":
        if "T_max" not in hp.scheduler_kwargs:
            hp.scheduler_kwargs["T_max"] = hp.max_epochs
        return CosineAnnealingWarmup(optimizer, **hp.scheduler_kwargs)
    elif scheduler_name == "CosineAnnealingWarmupRestarts":
        return CosineAnnealingWarmupRestarts(optimizer, max_lr=hp.optimizer_kwargs.lr,
            **hp.scheduler_kwargs)
    elif scheduler_name == "ReduceLROnPlateau":
        return ReduceLROnPlateau(optimizer, **hp.scheduler_kwargs)
    else:
        return getattr(optim.lr_scheduler, scheduler_name)(optimizer, **hp.scheduler_kwargs)
