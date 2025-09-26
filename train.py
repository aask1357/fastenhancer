import os
import time
import argparse
import random
import socket
import atexit

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from utils.data import get_dataset_dataloader
from utils import get_hparams, summarize
from wrappers import get_wrapper


def close_writer(writer : SummaryWriter):
    writer.close()


def main():
    assert torch.cuda.is_available(), "CPU training is not allowed."

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', required=True, type=str,
        help="checkpoints and logs will be saved at logs/{name}")
    parser.add_argument('-c', '--config', type=str,
                        help="path to config json file. Default: logs/{name}/config.yaml")
    parser.add_argument('-p', '--params', nargs='+', default=[])
    parser.add_argument('-f', '--force_save', action='store_true',
                        help="force to save config file even if it already exists")

    a = parser.parse_args()
    base_dir = os.path.join('logs', a.name)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    hps = get_hparams(a.config, base_dir, save=True, params=a.params,
                      force_save=a.force_save)

    # assign available port
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    os.environ['MASTER_PORT'] = str(sock.getsockname()[1])
    sock.close()

    n_gpus = torch.cuda.device_count()
    print(f'Number of GPUs: {n_gpus}\n')
    
    if n_gpus > 1:
        mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))
    else:
        run(0, n_gpus, hps)


def run(rank, n_gpus, hps):
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('medium')
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=n_gpus,
        rank=rank,
        device_id=rank
    )
    hp = hps.train

    seed = hp.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.cuda.set_device(rank)
    wrapper = get_wrapper(hps.wrapper)(hps, train=True, rank=rank)
    wrapper.load()

    train_dataset, train_loader = get_dataset_dataloader(
        hps, mode="train", keys=wrapper.keys,
        n_gpus=n_gpus, rank=rank)
    val_keys = getattr(wrapper, "val_keys", wrapper.keys)
    _, val_loader = get_dataset_dataloader(
        hps, mode="valid", keys=val_keys,
        n_gpus=n_gpus, rank=rank)

    if rank == 0:
        _, infer_loader = get_dataset_dataloader(
            hps, mode="infer", keys=wrapper.infer_keys,
            n_gpus=1, rank=0)

        writer_train = SummaryWriter(log_dir=os.path.join(hps.base_dir, "train"))
        writer_valid = SummaryWriter(log_dir=os.path.join(hps.base_dir, "valid"))
        atexit.register(close_writer, writer_train)
        atexit.register(close_writer, writer_valid)

        if wrapper.epoch == 0:
            if wrapper.plot_param_and_grad:
                hists = wrapper.plot_initial_param(train_loader)
                summarize(writer_train, epoch=0, hists=hists)
            #wrapper.save()

        start_time = time.time()

    if hasattr(hps, "infer"):
        infer_interval = hps.infer.interval
    else:
        infer_interval = hp.infer_interval
    for epoch in range(wrapper.epoch + 1, hps.train.max_epochs + 1):
        wrapper.epoch = epoch
        lr = wrapper.get_lr()

        # train
        train_dataset.shuffle(hp.seed + epoch)  # type: ignore
        summary_train = wrapper.train_epoch(train_loader)

        # valid
        summary_valid = wrapper.valid_epoch(val_loader)

        # summarize & infer
        if rank == 0:
            if epoch == 1 or epoch % infer_interval == 0:
                summary_infer = wrapper.infer_epoch(infer_loader)
                summarize(writer_valid, epoch, sampling_rate = hps.data.sampling_rate,
                    **summary_infer)

            if epoch % hp.save_interval == 0:
                wrapper.save()

            end_time = time.time()
            if "scalars" not in summary_train:
                summary_train["scalars"] = {}
            if "lr" not in summary_train["scalars"]:
                summary_train["scalars"]["lr"] = lr
            if hasattr(wrapper, "scaler"):
                scale = wrapper.scaler.get_scale()
                summary_train["scalars"]["scale"] = scale
            print(f"Epoch {epoch} - Time: {end_time - start_time:.1f} sec\tName: {hps.base_dir}")
            print("\tTrain", end="")
            summarize(writer_train, epoch, **summary_train)
            print("\tValid", end="")
            summarize(writer_valid, epoch, **summary_valid)
            start_time = end_time
    if rank == 0:
        writer_train.close()
        writer_valid.close()

    # It seems that in certain cases, this function doesn't finish, so I commented it.
    # See "https://github.com/pytorch/pytorch/issues/75097"
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
