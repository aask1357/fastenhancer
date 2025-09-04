import os, time, argparse, random, atexit

import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from utils.data import get_dataset_dataloader
from utils import get_hparams, summarize
from models import get_wrapper


def close_writer(writer : SummaryWriter):
    writer.close()


@record
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', required=True, type=str,
        help="checkpoints and logs will be saved at logs/{name}")
    parser.add_argument('-c', '--config', type=str,
                        help="path to config json file. Default: logs/{name}/config.yaml")
    parser.add_argument('-p', '--params', nargs='+', default=[])
    parser.add_argument('-f', '--force_save', action='store_true',
                        help="force to save config file even if it already exists")

    a = parser.parse_args()
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    n_gpus = torch.cuda.device_count()
    if rank == 0:
        print(f'Number of GPUs: {n_gpus}\n')
    
    base_dir = os.path.join('logs', a.name)
    if rank == 0 and not os.path.exists(base_dir):
        os.makedirs(base_dir)
    dist.barrier()
    hps = get_hparams(a.config, base_dir, save=(rank==0), params=a.params,
                      force_save=a.force_save)
    
    hp = hps.train
    seed = hp.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.cuda.set_device(rank)
    wrapper = get_wrapper(hps.model)(hps, train=True, rank=rank)
    wrapper.load()
    
    textprocessor = getattr(wrapper, "textprocessor", None)
    train_dataset, train_loader = get_dataset_dataloader(
        hps, mode="train", keys=wrapper.keys, textprocessor=textprocessor,
        n_gpus=n_gpus, rank=rank)
    val_keys = getattr(wrapper, "val_keys", wrapper.keys)
    _, val_loader = get_dataset_dataloader(
        hps, mode="valid", keys=val_keys, textprocessor=textprocessor,
        n_gpus=n_gpus, rank=rank)
    
    if rank == 0:
        _, infer_loader = get_dataset_dataloader(
            hps, mode="infer", keys=wrapper.infer_keys, textprocessor=textprocessor,
            n_gpus=1, rank=0)

        writer_train = SummaryWriter(log_dir=os.path.join(hps.base_dir, "train"))
        writer_valid = SummaryWriter(log_dir=os.path.join(hps.base_dir, "valid"))
        atexit.register(close_writer, writer_train)
        atexit.register(close_writer, writer_valid)
    
        if wrapper.epoch == 0:
            if wrapper.plot_param_and_grad:
                hists = wrapper.plot_initial_param(train_loader)
                summarize(writer_train, epoch=0, hists = hists)
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
        train_dataset.shuffle(hp.seed + epoch)
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
    # dist.barrier()
    # dist.destroy_process_group()


if __name__ == '__main__':
    assert torch.cuda.is_available(), "CPU training is not allowed."
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('medium')
    main()
