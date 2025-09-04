from typing import Dict, Optional, Union, Optional, Tuple
from enum import Enum
import math

import torch
from torch import jit, Tensor, nn
from torch.nn import functional as F
from torch import distributed as dist
from torchaudio.transforms import MelSpectrogram
from librosa.filters import mel as librosa_mel_fn
from torch_pesq import PesqLoss

from utils import HParams
from functional import CompressedSTFT


class InputType(Enum):
    WAV = 0
    SPEC = 1


def product(s1: Tensor, s2: Tensor) -> Tensor:
    norm = torch.sum(s1*s2, -1, keepdim=True)
    return norm


@jit.script
def si_snr(s1: Tensor, s2: Tensor, eps: float = 1e-7) -> Tensor:
    # s1: wav_hat / s2: wav
    s1_s2_norm = product(s1, s2)
    s2_s2_norm = product(s2, s2)
    s_target =  s1_s2_norm / (s2_s2_norm + eps) * s2
    e_nosie = s1 - s_target
    target_norm = product(s_target, s_target)
    noise_norm = product(e_nosie, e_nosie)
    snr = torch.log10(target_norm / (noise_norm + eps) + eps)
    return -10.0 * torch.mean(snr)


class _Loss:
    def __init__(self, weight: float, input_type: InputType):
        self.weight = weight
        self.loss: Tensor = torch.empty(1)
        self.input_type = input_type
    
    def initialize(self, device, dtype):
        self.loss = torch.zeros(1, dtype=dtype, device=device)
    
    def calculate(
        self, wav_g: Tensor, wav_r: Tensor, spec_g: Tensor, spec_r: Tensor,
        batch_size: int
    ) -> Tensor:
        if self.input_type == InputType.WAV:
            loss = self._calculate(wav_g, wav_r)
        elif self.input_type == InputType.SPEC:
            loss = self._calculate(spec_g, spec_r)
        self.loss.add_(loss.detach(), alpha=batch_size)
        return loss * self.weight
    
    def _calculate(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError()


class SISNR(_Loss):
    def __init__(self, weight: float, eps: float = 1e-7):
        super().__init__(weight, InputType.WAV)
        self.eps = eps
    
    def _calculate(self, wav_hat: Tensor, wav: Tensor) -> Tensor:
        return si_snr(wav_hat, wav, self.eps)


class WavL1(_Loss):
    def __init__(self, weight: float):
        super().__init__(weight, InputType.WAV)
    
    def _calculate(self, wav_hat: Tensor, wav: Tensor) -> Tensor:
        return F.l1_loss(wav_hat, wav)


class ConsistencyLoss(_Loss):
    def __init__(
        self, weight: float, compression: float = 1.0,
        n_fft: int = 512, hop_size: int = 256,
        win_size: Optional[int] = None, win_type: Optional[str] = None,
    ):
        super().__init__(weight, InputType.WAV)
        self.stft = CompressedSTFT(
            n_fft=n_fft,
            hop_size=hop_size,
            win_size=win_size,
            compression=compression,
            win_type=win_type,
        )
    
    def initialize(self, device, dtype):
        self.stft.to(device=device, dtype=dtype)
        super().initialize(device, dtype)
    
    def _calculate(self, wav_hat: Tensor, wav: Tensor) -> Tensor:
        spec_hat = self.stft(wav_hat)
        spec = self.stft(wav)
        return F.mse_loss(spec_hat, spec)


def anti_wrapping_function(x: Tensor) -> Tensor:
    return torch.abs(x - torch.round(x / (2 * math.pi)) * 2 * math.pi)


def phase_losses(phase_r: Tensor, phase_g: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    ip_loss = anti_wrapping_function(phase_r - phase_g).mean()
    gd_loss = anti_wrapping_function(
        torch.diff(phase_r, dim=1) - torch.diff(phase_g, dim=1)
    ).mean()
    iaf_loss = anti_wrapping_function(
        torch.diff(phase_r, dim=2) - torch.diff(phase_g, dim=2)
    ).mean()
    return ip_loss, gd_loss, iaf_loss


class PhaseLoss(_Loss):
    def __init__(self, weight: float):
        super().__init__(weight, InputType.SPEC)
    
    def _calculate(self, spec_hat: Tensor, spec: Tensor) -> Tensor:
        phase_hat = torch.angle(spec_hat)
        phase = torch.angle(spec)
        ip_loss, gd_loss, iaf_loss = phase_losses(phase_hat, phase)
        return ip_loss + gd_loss + iaf_loss


class MagMSE(_Loss):
    def __init__(self, weight: float):
        super().__init__(weight, InputType.SPEC)
    
    def _calculate(self, s1: Tensor, s2: Tensor) -> Tensor:
        s1 = torch.linalg.norm(s1, dim=-1)
        s2 = torch.linalg.norm(s2, dim=-1)
        return F.mse_loss(s1, s2)


class ComplexMSE(_Loss):
    def __init__(self, weight: float):
        super().__init__(weight, InputType.SPEC)
    
    def _calculate(self, s1: Tensor, s2: Tensor) -> Tensor:
        return F.mse_loss(s1, s2)


class PESQ(_Loss):
    def __init__(self, weight: float, custom: bool = False):
        super().__init__(weight, InputType.WAV)
        self.initialized = False  # lazy initialization
        self.pesq_loss = PesqLoss(1.0, sample_rate=16_000)
    
    def _calculate(self, wav_hat: Tensor, wav: Tensor) -> Tensor:
        if not self.initialized:
            self.pesq_loss.to(wav.device)
            self.initialized = True
        with torch.amp.autocast("cuda", enabled=False):
            loss = self.pesq_loss(wav.float(), wav_hat.float())
        return loss.mean()


LOSS_CLASS = {
    "si_snr": SISNR,
    "wav_l1": WavL1,
    "mag_mse": MagMSE,
    "complex_mse": ComplexMSE,
    "consistency": ConsistencyLoss,
    "phase": PhaseLoss,
    "pesq": PESQ,
}


class Losses:
    def __init__(self, losses: HParams):
        self.losses: Dict[str, _Loss] = {}
        for name, kwargs in losses.items():
            self.losses[name] = LOSS_CLASS[name](**kwargs)
        self.world_size = dist.get_world_size()
        self.n_items = 0
        self.device = None
        self.dtype = torch.float32
    
    def initialize(self, device, dtype=torch.float32):
        for loss in self.losses.values():
            loss.initialize(device, dtype)
        self.n_items = 0
        self.device = device
        self.dtype = dtype

    def print(self) -> str:
        out = ""
        for name, loss_class in self.losses.items():
            out = f"{out}  {name}: {loss_class.loss.item() / self.n_items:8.2e}"
        return out

    def get(self, key: str) -> float:
        if key not in self.losses:
            return 0.0
        return self.losses[key].loss.item() / self.n_items

    def calculate(
        self, wav_hat: Tensor, spec_hat: Tensor, wav: Tensor, spec: Tensor
    ) -> Tensor:
        loss_total = wav.new_zeros(1)
        batch_size = wav_hat.size(0)
        time = spec_hat.size(2)
        self.n_items += batch_size
        for loss in self.losses.values():
            loss_total += loss.calculate(wav_hat, wav, spec_hat, spec, batch_size)
        return loss_total

    def reduce(self) -> Dict[str, float]:
        losses = {}
        if self.world_size > 1:
            keys = self.losses.keys()
            n_items_tensor = torch.full((1,), float(self.n_items),
                dtype=self.dtype, device=self.device)
            bucket = torch.cat([self.losses[key].loss for key in keys] + \
                [n_items_tensor])
            dist.reduce(bucket, dst=0, op=dist.ReduceOp.SUM)
            bucket = bucket.cpu()
            n_items = bucket[-1].item()
            for idx, key in enumerate(keys):
                losses[f"loss/{key}"] = bucket[idx].item() / n_items
        else:
            n_items = self.n_items
            for key, value in self.losses.items():
                losses[f"loss/{key}"] = value.loss.item() / n_items
        return losses
