import math
import time
from typing import Tuple, List

import torch
from torch import Tensor
import torch.nn as nn
from torch import amp

from wrappers.ns import ModelWrapper as BaseModelWrapper
from utils.data.ns_on_the_fly import SNRMixer
from utils.terminal import clear_current_line
from utils import plot_param_and_grad


class DynamicBatchLPF(nn.Module):
    def __init__(
        self,
        sampling_rate: int = 48000,
        kernel_size: int = 127,
        p_lpf: float = 0.0,
        window: str = "hann",
        target_sr_list: List[int] = [8000, 16000, 22050, 24000, 32000, 44100],
    ):
        assert kernel_size % 2 == 1, "Kernel size must be odd for symmetric FIR filters."
        super().__init__()
        self.sr = sampling_rate
        self.p_lpf = p_lpf
        self.padding = kernel_size // 2

        nyquist_freqs = torch.tensor([sr / 2 for sr in target_sr_list], dtype=torch.float32)
        nyquist_freqs_angular = 2.0 * nyquist_freqs / self.sr
        self.register_buffer('nyquist_freqs_angular', nyquist_freqs_angular)

        n = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1).float()
        self.register_buffer("n_grid", n.view(1, -1))
        if window == "hann":
            window = torch.hann_window(kernel_size)
        else:
            raise ValueError(f"Window {window} is currently not implemented.")
        self.register_buffer("window", window.view(1, -1))

    def forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        if not self.training or self.p_lpf <= 0.0:
            return x, y
        B = x.size(0)
        num_lpf = (torch.rand(B) < self.p_lpf).sum().item()
        if num_lpf == 0:
            return x, y
        x_lpf = x[:num_lpf]
        y_lpf = y[:num_lpf]

        idx = torch.randint(0, len(self.nyquist_freqs_angular), (num_lpf,), device=x.device)
        sampled_nyq = self.nyquist_freqs_angular[idx]
        alpha = torch.empty(num_lpf, device=x.device).uniform_(0.95, 1.0)
        cutoffs = (sampled_nyq * alpha).unsqueeze(1)
        h = cutoffs * torch.sinc(cutoffs * self.n_grid)     # [n_lpf, kernel_size]
        h = h * self.window
        h = h / h.sum(dim=1, keepdim=True) # Normalize DC gain to 1
        
        # Conv
        h = h.view(num_lpf, 1, -1)   # [n_lpf, 1, kernel_size]
        x_lpf = torch.nn.functional.conv1d(
            x_lpf.unsqueeze(0), h, padding=self.padding, groups=num_lpf
        ).squeeze(0)
        y_lpf = torch.nn.functional.conv1d(
            y_lpf.unsqueeze(0), h, padding=self.padding, groups=num_lpf
        ).squeeze(0)

        x[:num_lpf] = x_lpf
        y[:num_lpf] = y_lpf

        return x, y


class ModelWrapper(BaseModelWrapper):
    def __init__(self, hps, train=False, rank=0, device='cpu'):
        super().__init__(hps, train, rank, device)
        self.snr_mixer = SNRMixer(sr=self.sr, **hps.data.snr_mixer)
        if hasattr(hps.data, "dynamic_lpf"):
            self.dynamic_lpf = DynamicBatchLPF(sampling_rate=self.sr, **hps.data.dynamic_lpf)
        else:
            self.dynamic_lpf = DynamicBatchLPF(sampling_rate=self.sr, p_lpf=0.0)   # No LPF applied
        self.dynamic_lpf.cuda(rank)

    def set_keys(self):
        self.keys = ["clean", "noise", "noisy"]
        self.infer_keys = self.keys

    def to(self, device):
        self.device = device
        self.model.to(device)
        self.dynamic_lpf.to(device)

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
            wav_noise = batch["noise"].cuda(self.rank, non_blocking=True)
            length = wav_clean.size(-1) // self.hop_size * self.hop_size
            wav_clean = wav_clean[..., :length]
            wav_noise = wav_noise[..., :length]
            wav_clean, wav_noise, wav_noisy = self.snr_mixer(wav_clean, wav_noise)   # [B, T]
            wav_clean, wav_noisy = self.dynamic_lpf(wav_clean, wav_noisy)

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
