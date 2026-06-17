from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    print(
        "utils.data.segmental_rms.py: [Warning] "
        "Triton not available. Fallback to PyTorch implementation."
    )
    _TRITON_AVAILABLE = False


def _segmental_rms_torch(
    wav:                    Tensor,
    sr:                     int             = 16_000,
    window_ms:              int             = 100,
    relative_threshold_db:  float           = -25.0,
    absolute_threshold_db:  Optional[float] = -50.0,
) -> Tensor:
    B   = wav.shape[0]
    wav = wav.reshape(B, -1)
    T   = wav.shape[-1]
    win = int(sr * window_ms / 1000)
    T_valid = (T // win) * win

    if T_valid == 0:
        return wav.pow(2).mean(dim=-1, keepdim=True).sqrt().unsqueeze(-1)

    # --- Frame mean power via avg_pool1d (fused CUDA kernel, no reshape copy) ---
    # Shape: (B, 1, T) → pool → (B, 1, N) → (B, N)
    seg_pow = F.avg_pool1d(
        wav[:, :T_valid].pow(2).unsqueeze(1),
        kernel_size=win, stride=win
    ).squeeze(1)                                    # (B, N)

    # --- Stay in power domain for all threshold logic ---
    # dB thresholds: amplitude → power means dividing dB by 10, not 20
    rel_pow_ratio = 10.0 ** (relative_threshold_db / 10.0)

    max_pow   = seg_pow.amax(dim=-1, keepdim=True)  # amax: no .values unwrap needed
    threshold = max_pow * rel_pow_ratio

    if absolute_threshold_db is not None:
        abs_floor = 10.0 ** (absolute_threshold_db / 10.0)
        threshold = threshold.clamp(min=abs_floor)   # avoids extra tensor alloc

    active   = seg_pow > threshold                  # bool, no .float() yet
    n_active = active.sum(dim=-1, keepdim=True)     # int

    # masked mean of power, then single sqrt at the end
    rms_out = (seg_pow * active).sum(dim=-1, keepdim=True) \
              / n_active.clamp(min=1).float()
    rms_out = torch.where(
        n_active > 0,
        rms_out.sqrt(),
        rms_out.new_full(rms_out.shape, float("inf"))
    )

    return rms_out.unsqueeze(-1)


# ── Kernel 1: frame mean-power ────────────────────────────────────────────────
@triton.jit
def frame_pow_kernel(
    wav_ptr, out_ptr,
    T_valid, win,
    stride_b,                       # wav stride along batch dim
    B, N,                           # N = T_valid // win
    BLOCK_WIN: tl.constexpr,        # must be >= win; tile over one frame
):
    b = tl.program_id(0)
    n = tl.program_id(1)

    offs = tl.arange(0, BLOCK_WIN)
    mask = offs < win

    base = b * stride_b + n * win
    x    = tl.load(wav_ptr + base + offs, mask=mask, other=0.0)
    acc  = tl.sum(x * x, axis=0) / win

    tl.store(out_ptr + b * N + n, acc)


# ── Kernel 2: threshold + masked mean (one CTA per batch row) ─────────────────
@triton.jit
def masked_mean_kernel(
    pow_ptr, out_ptr,
    N,
    rel_ratio,                      # scalar: power ratio for relative threshold
    abs_floor,                      # scalar: absolute power floor (or -1 if disabled)
    BLOCK_N: tl.constexpr,
):
    b    = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N

    p = tl.load(pow_ptr + b * N + offs, mask=mask, other=0.0)

    # --- max reduction ---
    max_p = tl.max(tl.where(mask, p, 0.0), axis=0)

    # --- threshold ---
    thr = max_p * rel_ratio
    if abs_floor >= 0.0:
        thr = tl.maximum(thr, abs_floor)

    active   = (p > thr) & mask
    n_active = tl.sum(active.to(tl.float32), axis=0)
    sum_p    = tl.sum(tl.where(active, p, 0.0), axis=0)

    mean_p = tl.where(n_active > 0, sum_p / n_active, float("inf"))
    tl.store(out_ptr + b, tl.sqrt(mean_p))


def _segmental_rms_triton(
    wav:                    Tensor,
    sr:                     int             = 16_000,
    window_ms:              int             = 100,
    relative_threshold_db:  float           = -25.0,
    absolute_threshold_db:  Optional[float] = -50.0,
) -> Tensor:
    B   = wav.shape[0]
    wav = wav.reshape(B, -1).contiguous()
    T   = wav.shape[-1]
    win = int(sr * window_ms / 1000)
    T_valid = (T // win) * win
    N   = T_valid // win

    if T_valid == 0:
        return wav.pow(2).mean(dim=-1, keepdim=True).sqrt().unsqueeze(-1)

    # ── Frame power ───────────────────────────────────────────────────────────
    seg_pow  = torch.empty(B, N, device=wav.device, dtype=wav.dtype)
    BLOCK_WIN = triton.next_power_of_2(win)

    frame_pow_kernel[(B, N)](
        wav, seg_pow,
        T_valid, win,
        wav.stride(0),
        B, N,
        BLOCK_WIN=BLOCK_WIN,
    )

    # ── Masked mean ───────────────────────────────────────────────────────────
    rms_out  = torch.empty(B, device=wav.device, dtype=wav.dtype)
    BLOCK_N  = triton.next_power_of_2(N)

    rel_ratio = 10.0 ** (relative_threshold_db / 10.0)
    abs_floor = (10.0 ** (absolute_threshold_db / 10.0)
                 if absolute_threshold_db is not None else -1.0)

    masked_mean_kernel[(B,)](
        seg_pow, rms_out,
        N,
        rel_ratio, abs_floor,
        BLOCK_N=BLOCK_N,
    )

    return rms_out.reshape(B, 1, 1)

def segmental_rms(
    wav:                    Tensor,
    sr:                     int             = 16_000,
    window_ms:              int             = 100,
    relative_threshold_db:  float           = -25.0,
    absolute_threshold_db:  Optional[float] = -50.0,
) -> Tensor:
    if _TRITON_AVAILABLE and wav.is_cuda:
        return _segmental_rms_triton(wav, sr, window_ms,
                                    relative_threshold_db, absolute_threshold_db)
    return _segmental_rms_torch(wav, sr, window_ms,
                               relative_threshold_db, absolute_threshold_db)
