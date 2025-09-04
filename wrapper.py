import os
import math
import time
import re
import importlib
from typing import Optional, Dict, Any

import torch
from torch import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from torch.utils.data import DataLoader
try:
    from torch.amp import GradScaler
except:
    from torch.cuda.amp import GradScaler
import torch.distributed as dist
from tqdm import tqdm

from functional import stft, spec_to_mel
from optim import get_optimizer, get_scheduler
from utils import clip_grad_norm_local, plot_param_and_grad
from utils.data import get_dataset_dataloader
from utils.terminal import clear_current_line
from utils.measure_metrics import Metrics
from losses import Losses


def get_model(hps : str) -> torch.nn.Module:
    model: str = hps.model
    module = importlib.import_module(f"models.{model}.model")
    return module.Model(**hps.model_kwargs)


class ModelWrapper:
    def __init__(self, hps, train=False, rank=0, device='cpu'):
        self.base_dir: str = hps.base_dir
        self.rank: int = rank
        self.model: torch.nn.Module = get_model(hps)
        self.train_mode: bool = train
        self.device = device
        self.epoch: int = 0
        self.keys = ["clean", "noisy"]
        self.infer_keys = self.keys

        self.h = hps.data
        self.hop_size = hps.model_kwargs.hop_size
        self.sr = hps.data.sampling_rate

        self._module = self.model
        if train:
            hp = hps.train
            self.test = getattr(hp, "test", False)
            if self.test:
                hp.max_epochs = 1

            self.plot_param_and_grad = hp.plot_param_and_grad
            self.fp16 = hp.fp16
            self.autocast_dtype = torch.bfloat16 if getattr(hp, "bf16", False) else torch.float16
            self.scaler = GradScaler(enabled=hp.fp16)
            torch.cuda.set_device(f'cuda:{rank}')
            self.device = torch.device("cuda", rank)
            if hp.clip_grad is None:
                self.clip_grad = lambda params: None
            elif hp.clip_grad == "norm" or hp.clip_grad == "norm_global":
                self.clip_grad = lambda params: clip_grad_norm_(params, **hp.clip_grad_kwargs)
            elif hp.clip_grad == "norm_local":
                self.clip_grad = lambda params: clip_grad_norm_local(params, **hp.clip_grad_kwargs)
            elif hp.clip_grad == "value":
                self.clip_grad = lambda params: clip_grad_value_(params, **hp.clip_grad_kwargs)
            else:
                raise RuntimeError(hp.clip_grad)

            self.model.cuda(rank)
            self.optim = get_optimizer(self.model, hp)
            self.scheduler = get_scheduler(self.optim, hp)

            self.world_size = dist.get_world_size()
            if self.world_size > 1:
                self.model = DDP(self.model, device_ids=[rank])
                self._module = self.model.module

            if "consistency" in hp.losses:
                hp.losses.consistency["n_fft"] = self._module.stft.n_fft
                hp.losses.consistency["hop_size"] = self._module.stft.hop_size
                hp.losses.consistency["win_size"] = getattr(hps.model_kwargs, "win_size", None)
                hp.losses.consistency["win_type"] = getattr(hps.model_kwargs, "window", None)

            self.loss = Losses(hp.losses)
            self.print_interval: int = getattr(hp, "print_interval", 1)

            if hasattr(hps, "pesq"):
                self.pesq_interval: int = hps.pesq.interval
                _, self.pesq_loader = get_dataset_dataloader(
                    hps, "pesq", ["clean", "noisy", "wav_len"],
                    n_gpus=self.world_size, rank=rank
                )
                self.metrics = Metrics(
                    num_workers=hps.pesq.num_workers_executor, sr=hps.data.sampling_rate,
                    world_size=self.world_size, rank=rank, device=self.device,
                    **hps.pesq.metrics_to_calculate)
            else:
                self.pesq_interval: int = hp.max_epochs + 1
                self.pesq_loader = None
                self.metrics = None
        else:
            self.device = device
            self.model.to(device)
        self._module.flatten_parameters()

    def set_keys(self):
        '''set self.keys, self.infer_keys
        self.keys: used for train_dataset & valid_dataset
        self.infer_keys: used for infer_dataset'''
        self.keys = None
        self.infer_keys = None
    
    def plot_initial_param(self, dataloader: DataLoader) -> dict:
        hists = {}
        plot_param_and_grad(hists, self._module)
        return hists

    def get_lr(self) -> float:
        return self.optim.param_groups[0]['lr']
    
    def to(self, device):
        self.device = device
        self.model.to(device)
    
    def train(self):
        self.model.train()
    
    def eval(self):
        self.model.eval()

    def train_epoch(self, dataloader):
        self.train()
        self.loss.initialize(
            device=torch.device("cuda", index=self.rank),
            dtype=torch.float32
        )
        max_items = len(dataloader)
        padding = int(math.log10(max_items)) + 1

        summary = {"scalars": {}, "hists": {}}
        start_time = time.perf_counter()

        for idx, batch in enumerate(dataloader, start=1):
            self.optim.zero_grad(set_to_none=True)
            wav_clean = batch["clean"].cuda(self.rank, non_blocking=True)
            wav_noisy = batch["noisy"].cuda(self.rank, non_blocking=True)
            length = wav_clean.size(-1) // self.hop_size * self.hop_size
            wav_clean = wav_clean[..., :length]
            wav_noisy = wav_noisy[..., :length]

            with amp.autocast('cuda', enabled=self.fp16):
                spec_clean = self._module.stft(wav_clean)
                wav_hat, spec_hat = self.model(wav_noisy)
                loss = self.loss.calculate(
                    wav_hat, spec_hat, wav_clean, spec_clean,
                )

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optim)
            if idx == len(dataloader) and self.plot_param_and_grad:
                plot_param_and_grad(summary["hists"], self.model)
            self.clip_grad(self.model.parameters())
            self.scaler.step(self.optim)
            self.scaler.update()
            if self.rank == 0 and idx % self.print_interval == 0:
                time_ellapsed = time.perf_counter() - start_time
                print(
                    f"\rEpoch {self.epoch} - Train "
                    f"{idx:{padding}d}/{max_items} ({idx/max_items*100:>4.1f}%)"
                    f"{self.loss.print()}"
                    f"  scale {self.scaler.get_scale():.4f}"
                    f"  [{int(time_ellapsed)}/{int(time_ellapsed/idx*max_items)} sec]",
                    sep=' ', end='', flush=True
                )
            if hasattr(self.scheduler, "warmup_step"):
                self.scheduler.warmup_step()
            if self.test:
                if idx >= 10:
                    break
        if self.rank == 0:
            clear_current_line()
        self.scheduler.step()
        self.optim.zero_grad(set_to_none=True)

        summary["scalars"] = self.loss.reduce()
        return summary

    @torch.no_grad()
    def valid_epoch(self, dataloader):
        self.eval()
        summary = self._valid_epoch(dataloader)
        if self.epoch % self.pesq_interval == 0:
            summary["scalars"].update(self.calculate_metrics())
        return summary

    @torch.no_grad()
    def _valid_epoch(self, dataloader):
        self.loss.initialize(device=torch.device("cuda", index=self.rank), dtype=torch.float32)
        for batch in tqdm(
            dataloader,
            desc="Valid",
            disable=(self.rank!=0),
            leave=False,
            dynamic_ncols=True,
            smoothing=0.0,
        ):
            wav_clean = batch["clean"].cuda(self.rank, non_blocking=True)
            wav_noisy = batch["noisy"].cuda(self.rank, non_blocking=True)
            length = wav_clean.size(-1) // self.hop_size * self.hop_size
            wav_clean = wav_clean[..., :length]
            wav_noisy = wav_noisy[..., :length]

            with amp.autocast('cuda', enabled=self.fp16):
                spec_clean = self._module.stft(wav_clean)
                wav_hat, spec_hat = self.model(wav_noisy)
                self.loss.calculate(
                    wav_hat, spec_hat, wav_clean, spec_clean,
                )
        summary_scalars = self.loss.reduce()

        return {"scalars": summary_scalars}

    def calculate_metrics(self):
        self.metrics.initialize()
        for batch in tqdm(
            self.pesq_loader,
            desc="PESQ",
            disable=(self.rank!=0),
            leave=False,
            dynamic_ncols=True
        ):
            wav_clean = batch["clean"].cuda(self.rank, non_blocking=True)
            wav_noisy = batch["noisy"].cuda(self.rank, non_blocking=True)
            wav_len = [x // self.hop_size * self.hop_size for x in batch["wav_len"]]

            batch_wav_len = wav_clean.size(-1) // self.hop_size * self.hop_size
            wav_clean = wav_clean[:, :batch_wav_len]
            wav_noisy = wav_noisy[:, :batch_wav_len]
            with amp.autocast('cuda', enabled=self.fp16):
                wav_hat, *_ = self._module(wav_noisy)
            self.metrics.submit(wav_clean, wav_hat, wav_len)

        metrics = self.metrics.retrieve(verbose=(self.rank==0))
        return metrics

    @torch.no_grad()
    def infer_epoch(self, dataloader):
        self.eval()
        summary = {"audios": {}, "specs": {}}
        for idx, batch in enumerate(dataloader):
            wav_clean = batch["clean"].cuda(self.rank, non_blocking=True)
            wav_noisy = batch["noisy"].cuda(self.rank, non_blocking=True)

            batch_size = wav_clean.size(0)
            batch_wav_len = wav_clean.size(-1) // self.hop_size * self.hop_size
            wav_clean = wav_clean[:, :batch_wav_len]
            wav_noisy = wav_noisy[:, :batch_wav_len]

            if self.epoch == 1:
                for i in range(batch_size):
                    spec_clean = stft(wav_clean[i:i+1], 1024, 256, 1024)
                    spec_noisy = stft(wav_noisy[i:i+1], 1024, 256, 1024)
                    mel_clean = spec_to_mel(spec_clean, 1024, 80, self.h.sampling_rate)
                    mel_noisy = spec_to_mel(spec_noisy, 1024, 80, self.h.sampling_rate)

                    _idx = i + idx * batch_size + 1
                    summary["audios"][f"clean/wav_{_idx}"] = wav_clean[i].squeeze().cpu().numpy()
                    summary["specs"][f"clean/mel_{_idx}"] = mel_clean.squeeze().cpu().numpy()
                    summary["specs"][f"clean/spec_{_idx}"] = spec_clean.clamp_min(1e-5).log().squeeze().cpu().numpy()
                    summary["audios"][f"noisy/wav_{_idx}"] = wav_noisy[i].squeeze().cpu().numpy()
                    summary["specs"][f"noisy/mel_{_idx}"] = mel_noisy.squeeze().cpu().numpy()
                    summary["specs"][f"noisy/spec_{_idx}"] = spec_noisy.clamp_min(1e-5).log().squeeze().cpu().numpy()

            with amp.autocast('cuda', enabled=self.fp16):
                wav_hat, *_ = self.model(wav_noisy)
            wav_hat = wav_hat.float()

            for i in range(batch_size):
                spec_hat = stft(wav_hat[i:i+1], 1024, 256, 1024)
                mel_hat = spec_to_mel(spec_hat, 1024, 80, self.h.sampling_rate)

                _idx = i + idx * batch_size + 1
                summary["audios"][f"enhanced/wav_{_idx}"] = wav_hat[i].squeeze().cpu().numpy()
                summary["specs"][f"enhanced/mel_{_idx}"] = mel_hat.squeeze().cpu().numpy()
                summary["specs"][f"enhanced/spec_{_idx}"] = spec_hat.clamp_min(1e-5).log().squeeze().cpu().numpy()
        return summary
    
    def get_checkpoint(
        self,
        epoch: Optional[int]=None,
        path: Optional[str]=None
    ) -> Optional[Dict[str, Any]]:
        if path is None:
            if epoch is None:       # get lastest checkpoint
                files = [int(f[:-4]) for f in os.listdir(self.base_dir) if re.match('[0-9]{5,}.pth', f)]
                if not files:
                    if self.rank == 0:
                        print("No checkpoint exists.")
                    return None
                files.sort()
                epoch = files[-1]
            path = os.path.join(self.base_dir, f"{epoch:0>5d}.pth")
        checkpoint = torch.load(path, map_location=self.device)
        if self.rank == 0:
            print(f"Loading checkpoint file '{path}'...")
        return checkpoint

    def load(self, epoch: Optional[int] = None, path: Optional[str] = None):
        checkpoint = self.get_checkpoint(epoch, path)
        if checkpoint is None:
            return

        self._module.load_state_dict(checkpoint['model'], strict=False)
        self.epoch = checkpoint['epoch']

        if self.train_mode:
            self.optim.load_state_dict(checkpoint['optim'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.scaler.load_state_dict(checkpoint['scaler'])
            if self.metrics is not None:
                self.metrics.load_state_dict(checkpoint['metrics'])

    def save(self, path: Optional[str] = None):
        if path is None:
            path = os.path.join(self.base_dir, f"{self.epoch:0>5d}.pth")
        wrapper_dict = {
            "model": self._module.state_dict(),
            "optim": self.optim.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": self.epoch,
        }
        if self.metrics is not None:
            wrapper_dict["metrics"] = self.metrics.state_dict()

        torch.save(wrapper_dict, path)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def remove_weight_reparameterizations(self) -> None:
        self._module.remove_weight_reparameterizations()

