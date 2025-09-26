import math
import time

import torch
from torch import amp

from wrappers.ns import ModelWrapper as BaseModelWrapper
from utils.data.ns_on_the_fly import SNRMixer
from utils.terminal import clear_current_line
from utils import plot_param_and_grad


class ModelWrapper(BaseModelWrapper):
    def __init__(self, hps, train=False, rank=0, device='cpu'):
        super().__init__(hps, train, rank, device)
        self.snr_mixer = SNRMixer(sr=self.sr, **hps.data.snr_mixer)

    def set_keys(self):
        self.keys = ["clean", "noise", "noisy"]
        self.infer_keys = self.keys

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
            wav_clean, wav_noise, wav_noisy = self.snr_mixer(wav_clean, wav_noise)   # [B, 1, T]

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
