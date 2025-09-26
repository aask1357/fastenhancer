import typing as tp
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torch.utils.data.distributed import DistributedSampler

from .noise_suppression import NSDataset, collate
from .ns_on_the_fly import NSOnTheFlyDataset

from utils import HParams


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataset_dataloader(
    hps: HParams,
    mode: str,
    keys: tp.List[str],
    textprocessor = None,
    n_gpus: int = 1,
    rank: int = 0,
) -> tp.Tuple[TorchDataset, DataLoader]:
    _collate = None
    dataset = hps["data"]["dataset"]
    if isinstance(dataset, HParams) or isinstance(dataset, dict):
        dataset = dataset[mode]

    if dataset == "NoiseSuppression":
        _Dataset, _collate = NSDataset, collate
    elif dataset == "NoiseSuppressionOnTheFly":
        _Dataset = NSOnTheFlyDataset
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    hp = hps["train"]
    batch_size = getattr(hp, "batch_size", -1)
    num_workers = getattr(hp, "num_workers", -1)
    drop_last = getattr(hp, f"drop_last", False)
    persistent_workers = getattr(hp, "persistent_worker", False)
    if mode == "valid":
        hp_valid = getattr(hps, "valid", {})
        batch_size = getattr(hp_valid, "batch_size", batch_size)
        num_workers = getattr(hp_valid, "num_workers", num_workers)
        drop_last = getattr(hp_valid, "drop_last", drop_last)
    elif mode == "infer":
        hp_infer = getattr(hps, "infer", {})
        batch_size = getattr(hp_infer, "batch_size", 1)
        num_workers = getattr(hp_infer, "num_workers", 0)     # Since we don't infer many samples at once, set num_workers=0
        drop_last = False
        # set segment_size to None
        segment_size = getattr(hps["data"], "segment_size", None)
        hps["data"]["segment_size"] = None
    elif mode == "pesq":
        hp_pesq = getattr(hps, "pesq", {})
        batch_size = getattr(hp_pesq, "batch_size", getattr(hp, "pesq_batch_size", -1))
        num_workers = getattr(hp_pesq, "num_workers", num_workers)
        drop_last = False
        persistent_workers = getattr(hp_pesq, "persistent_worker", persistent_workers)
        # set segment_size to None
        segment_size = getattr(hps["data"], "segment_size", None)
        hps["data"]["segment_size"] = None
    elif mode != "train":
        raise ValueError(f"Unknown dataset mode: {mode}")

    dataset = _Dataset(hps["data"], keys, textprocessor=textprocessor,
                       mode=mode, batch_size=batch_size*n_gpus, verbose=(rank==0))
    sampler = DistributedSampler(
        dataset, num_replicas=n_gpus, rank=rank, shuffle=False,
        drop_last=drop_last)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, sampler=sampler,
        num_workers=num_workers, collate_fn=_collate, worker_init_fn=seed_worker,
        persistent_workers=persistent_workers,
        pin_memory=getattr(hp, "pip_memory", False)
    )

    if mode == "pesq" or mode == "infer":
        # restore segment_size, in case of using segment_size in other modes
        hps["data"]["segment_size"] = segment_size

    return dataset, dataloader
