import os
from typing import Optional, Dict, List
import time
import copy

import torch
from torch import nn, Tensor
from torch import amp
import torch.distributed as dist
from tqdm import tqdm

from wrappers.ns import ModelWrapper as BaseModelWrapper
from utils.summarize import plot_param_and_grad
from utils.data.shard_on_the_fly import DynamicMixer, ShardBatch
from utils.terminal import clear_current_line


class ModelWrapper(BaseModelWrapper):
    def __init__(self, hps, train=False, rank=0, device='cpu'):
        self.load_reverb = getattr(hps.data.train, "prob_speech_reverb", 0.0) > 0.0
        super().__init__(hps, train, rank, device)

        self.mixer_train = DynamicMixer(
            sampling_rate=self.sr,
            **hps.data.train.mixer,
            **hps.data.dereverberation,
        )
        self.steps_per_epoch = hps.train.steps_per_epoch

    def set_keys(self):
        self.keys = ["speech", "speech_clean", "num_samples", "noise"]
        if self.load_reverb:
            self.keys += ["rir", "rir_onset", "rir_t60"]
        self.val_keys = ["clean", "noisy"]
        self.infer_keys = ["clean", "noisy"]

    def train_epoch(self, dataloader):
        self.train()
        self.loss.initialize(
            device=torch.device("cuda", index=self.rank),
            dtype=torch.float32
        )
        summary = {"scalars": {}, "hists": {}}
        total_load_time = total_mixer_time = total_forward_time = total_backward_time = 0.0
        start_time = time.perf_counter()
        st = time.perf_counter()
        if self.rank == 0 and self.epoch == 1:
            print("Constructing data samplers", flush=True)

        for idx, batch in zip(range(1, self.steps_per_epoch + 1), dataloader):
            et = time.perf_counter(); batch_load_time = et - st; st = et
            if idx == 1:
                print(f"Initial load time: {batch_load_time:.1f} sec"); start_time = et
            self.optim.zero_grad(set_to_none=True)

            batch: ShardBatch = batch.to(self.device)
            mixed = self.mixer_train(batch)
            et = time.perf_counter(); mixer_time = et - st; st = et

            wav_clean = mixed.speech_clean.squeeze(1)  # (B, 1, T) -> (B, T)
            wav_noisy = mixed.speech.squeeze(1)
            length = wav_clean.size(-1) // self.hop_size * self.hop_size
            wav_clean = wav_clean[..., :length]
            wav_noisy = wav_noisy[..., :length]

            with amp.autocast('cuda', enabled=self.fp16):
                spec_clean = self._module.stft(wav_clean)
                wav_hat, spec_hat = self.model(wav_noisy)
                loss = self.loss.calculate(
                    wav_hat, spec_hat, wav_clean, spec_clean,
                )
            et = time.perf_counter(); forward_time = et - st; st = et

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optim)
            if idx == self.steps_per_epoch and self.plot_param_and_grad:
                plot_param_and_grad(summary["hists"], self.model)
            self.clip_grad(self.model.parameters())
            self.scaler.step(self.optim)
            self.scaler.update()
            et = time.perf_counter(); backward_time = et - st
            total_load_time += batch_load_time
            total_mixer_time += mixer_time
            total_forward_time += forward_time
            total_backward_time += backward_time

            if hasattr(self.scheduler, "warmup_step"):
                self.scheduler.warmup_step()

            if self.rank == 0 and idx % self.print_interval == 0:
                time_ellapsed = time.perf_counter() - start_time
                print(
                    f"\rEpoch {self.epoch} - Step {idx}/{self.steps_per_epoch}"
                    f"{self.loss.print()}"
                    f"  scale {self.scaler.get_scale():.4f}"
                    f"  load: {batch_load_time:.2f}s"
                    f"  mixer: {mixer_time:.2f}s"
                    f"  [{time_ellapsed/idx:.2f} sec/iter, total {int(time_ellapsed)} sec]",
                    end='', flush=True
                )

            if self.test and idx >= 50:
                break
            st = time.perf_counter()

        if self.rank == 0:
            clear_current_line()
            print(
                f"Epoch {self.epoch} timing —"
                f"  load: {total_load_time:.1f}s"
                f"  mixer: {total_mixer_time:.1f}s"
                f"  forward: {total_forward_time:.1f}s"
                f"  backward: {total_backward_time:.1f}s"
            )
        self.scheduler.step()
        self.optim.zero_grad(set_to_none=True)

        summary["scalars"] = {
            f"{k}/train": v for k, v in self.loss.reduce().items()
        }
        return summary
