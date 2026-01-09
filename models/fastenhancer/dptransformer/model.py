import math
import typing as tp
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm as weight_norm_fn
from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations
from torch import Tensor
from torchaudio.functional import melscale_fbanks

from functional import ONNXSTFT, CompressedSTFT


class StridedConv1d(nn.Conv1d):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int,
        stride: int = 1, padding: int = 0, dilation: int = 1,
        groups: int = 1, bias: bool = True, padding_mode: str = 'zeros',
        device=None, dtype=None
    ):
        assert kernel_size % stride == 0, (
            f'kernel_size k and stride s must satisfy k=(2n+1)s, but '
            f'got k={kernel_size}, s={stride}. Use naive Conv1d instead.'
        )
        assert groups == 1, (
            f'groups must be 1, but got {groups}. '
            f'Use naive Conv1d instead.'
        )
        assert dilation == 1, (
            f'dilation must be 1, but got {dilation}. '
            f'Use naive Conv1d instead.'
        )
        assert padding_mode == 'zeros', (
            f'Only `zeros` padding mode is supported for '
            f'StridedConv1d, but got {padding_mode}. '
            f'Use naive Conv1d instead.'
        )
        self.original_stride = stride
        self.original_padding = padding
        super().__init__(
            in_channels*stride, out_channels, kernel_size//stride,
            stride=1, padding=0, dilation=dilation,
            groups=groups, bias=bias, padding_mode=padding_mode,
            device=device, dtype=dtype
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, Ci, Ti] -> conv1d -> [B, Co, Ti // s]
        <=> x: [B, Ci, Ti] -> reshape to [B, Ci*s, Ti//s] -> conv1d -> [B, Co, Ti//S]"""
        stride = self.original_stride
        padding = self.original_padding
        x = F.pad(x, (padding, padding))
        B, C, T = x.shape
        x = x.view(B, C, T//stride, stride).permute(0, 3, 1, 2).reshape(B, C*stride, T//stride)
        return super().forward(x)


class ScaledConvTranspose1d(nn.ConvTranspose1d):
    def __init__(
        self,
        *args,
        normalize: bool = False,
        exp_scale: bool = False,
        final_scale_init: str = "1/sqrt(fan_in)",
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.normalize = normalize
        self.exp_scale = exp_scale
        self.scale = nn.Parameter(torch.ones(1))
        if normalize:
            if final_scale_init == "1/sqrt(fan_in)":
                fan_in = self.in_channels * self.kernel_size[0] // self.stride[0]
                self.scale.data.mul_(1 / math.sqrt(fan_in))
            elif final_scale_init == "||weight||":
                self.scale.data.fill_(self.weight.data.square().sum().sqrt())
            elif final_scale_init == "one":
                self.scale.data.fill_(1.0)
            elif final_scale_init == "zero":
                self.scale.data.fill_zero_()
            else:
                # mean_std
                mean, std = final_scale_init.split('_')
                mean, std = float(mean), float(std)
                self.scale.data.fill_(self.weight.data.square().sum().sqrt().mul(std))
                if self.bias is not None:
                    self.bias.data.fill_(mean)
        if exp_scale:
            self.scale.data.clamp_(min=1e-5).log_()
        self.weight_norm = True

    def remove_weight_reparameterizations(self):
        scale = self.scale.exp() if self.exp_scale else self.scale
        if self.normalize:
            weight = F.normalize(self.weight, dim=(0, 1, 2)).mul_(scale)
        else:
            weight = self.weight * scale
        self.weight.data.copy_(weight)
        self.weight_norm = False

    def forward(self, x: Tensor) -> Tensor:
        if self.weight_norm:
            scale = self.scale.exp() if self.exp_scale else self.scale
            if self.normalize:
                weight = F.normalize(self.weight, dim=(0, 1, 2)).mul_(scale)
            else:
                weight = self.weight * scale
        else:
            weight = self.weight
        return F.conv_transpose1d(
            x, weight, self.bias, stride=self.stride,
            padding=self.padding, output_padding=self.output_padding,
            groups=self.groups, dilation=self.dilation,
        )


def calculate_positional_embedding(channels: int, freq: int) -> Tensor:
    # f0: [1/F, 2/F, ..., 1] * pi
    # c: [1, ..., F-1] -> log-spaced, numel = C//2
    f = torch.arange(1, freq+1, dtype=torch.float32) * (math.pi / freq)
    c = torch.linspace(
        start=math.log(1),
        end=math.log(freq-1),
        steps=channels//2,
        dtype=torch.float32
    ).exp()
    grid = f.view(-1, 1) * c.view(1, -1)            # [F, C//2]
    pe = torch.cat((grid.sin(), grid.cos()), dim=1) # [F, C]
    return pe


class ChannelsLastBatchNorm(nn.BatchNorm1d):
    def forward(self, x: Tensor) -> Tensor:
        """input/output: [T, B, F, C]"""
        T, B, F, C = x.shape
        x = x.view(T*B*F, C, 1)
        return super().forward(x).view(T, B, F, C)


class ChannelsLastSyncBatchNorm(nn.SyncBatchNorm):
    def forward(self, x: Tensor) -> Tensor:
        """input/output: [T, B, F, C]"""
        T, B, F, C = x.shape
        x = x.view(T*B*F, C, 1)
        return super().forward(x).view(T, B, F, C)


@torch.jit.script
def expand_attn_map(x: Tensor, T: int, value: float) -> Tensor:
    '''
    input: [N, L] where L < T
    output: [L, T, T]
    ex) input: [1, 2] -> output: [1, 4, 4]
    input:
        1 2
    output:
           2 -inf -inf -inf
           1    2 -inf -inf
        -inf    1    2 -inf
        -inf -inf    1    2
    '''
    N, L = x.shape
    x = F.pad(x, (0, T), value=value)       # [N, L+T]
    x = x.view(N, 1, L+T).repeat(1, T, 1)   # [N, T, L+T]
    x = x.flatten(start_dim=1)              # [N, T*(L+T)]
    x = x[:, :-T]                           # [N, T*(L+T-1)]
    x = x.reshape(N, T, L+T-1)              # [N, T, L+T-1]
    x = x[:, :, L-1:]                       # [N1, N2, T, T]
    return x


class CausalAttention(nn.Module):

    __constants__ = ["channels", "num_heads", "freq", "lookbehind"]

    def __init__(
        self,
        channels: int,
        num_heads: int,
        attn_bias: bool,
        freq: int,
        lookbehind: int,
    ):
        super().__init__()
        self.channels = channels // num_heads
        self.num_heads = num_heads
        self.scale: float = (channels // num_heads) ** -0.5
        self.qkv = nn.Linear(channels, channels*3, bias=attn_bias)
        self.freq = freq
        self.lookbehind = lookbehind

    def initialize_cache(self, x: Tensor) -> tp.List[Tensor]:
        return [
            x.new_zeros(self.freq, self.num_heads, self.lookbehind, self.channels),
            x.new_zeros(self.freq, self.num_heads, self.lookbehind, self.channels)
        ]

    def forward(
        self,
        x: Tensor,
        pe: Tensor,
        h_k: tp.Optional[Tensor],
        h_v: tp.Optional[Tensor],
    ) -> tp.Tuple[Tensor, Tensor, Tensor]:
        '''input:
            x: [B*F, T, C]
            pe: [NH, L+1] where L = lookbehind
            h: [B*F, NH, L, C']
        output:
            out: [B*F, T, C]'''
        BF, T, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(BF, T, self.num_heads, -1).transpose(1, 2)    # [BF, NH, T, 3*C']
        q, k, v = torch.chunk(qkv, chunks=3, dim=3)                     # [BF, NH, T, C']
        if h_k is None:
            mask = expand_attn_map(pe, T, -float("inf"))        # [NH, T, T]
            out = F.scaled_dot_product_attention(q, k, v, mask) # [BF, NH, T, C'']
            # attn_map = torch.einsum('BNTC,BNtC->BNTt', (q, k))  # [BF, NH, T, T]
            # pe = expand_attn_map(pe, -float("inf"))             # [NH, T, T]
            # attn_map = pe.add(attn_map, alpha=self.scale)       # [BF, NH, T, T]
            # attn_map = F.softmax(attn_map, dim=-1)
            # out = torch.matmul(attn_map, v)                     # [BF, NH, T, C']
        else:
            # q: [F, NH, 1, C']
            k = torch.concat((h_k, k), dim=2)                   # [F, NH, L+1, C']
            v = torch.concat((h_v, v), dim=2)                   # [F, NH, L+1, C']
            attn_map = (q * k).sum(dim=3)                       # [F, NH, L+1]
            attn_map = pe.add(attn_map, alpha=self.scale)       # [F, NH, L+1]
            attn_map = F.softmax(attn_map, dim=2).unsqueeze(2)  # [F, NH, 1, L+1]
            out = attn_map @ v                                  # [F, NH, 1, C']
        h_k = k[:, :, -self.lookbehind:, :]                     # [BF, NH, L, C']
        h_v = v[:, :, -self.lookbehind:, :]                     # [BF, NH, L, C']
        out = out.transpose(1, 2).reshape(BF, T, -1)            # [BF, T, C]
        return out, h_k, h_v


class Attention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        attn_bias: bool,
    ):
        super().__init__()
        self.channels = channels // num_heads
        self.num_heads = num_heads
        self.scale: float = (channels // num_heads) ** -0.5
        self.qkv = nn.Linear(channels, channels*3, bias=attn_bias)

    def forward(self, x: Tensor) -> Tensor:
        '''input / output: [T*B, F, C]'''
        TB, Freq, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(TB, Freq, self.num_heads, -1).transpose(1, 2)     # [TB, NH, F, C']
        q = qkv[:, :, :, :self.channels]
        k = qkv[:, :, :, self.channels:self.channels*2]
        v = qkv[:, :, :, self.channels*2:]
        out = F.scaled_dot_product_attention(q, k, v, scale=None)     # [TB, NH, F, C'']
        out = out.transpose(1, 2).reshape(TB, Freq, -1)     # [TB, F, Cout]
        return out


class DPTBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        freq: int,
        num_heads: int,
        weight_norm: bool,
        activation: str,
        activation_kwargs: tp.Dict[str, tp.Any],
        positional_embedding: tp.Optional[str],
        attn_bias: bool = False,
        eps: float = 1e-8,
        post_act: bool = False,
        pre_norm: bool = False,
        lookbehind: int = 16,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.freq = freq
        self.pre_norm = pre_norm

        def Act(**kwargs):
            if post_act:
                return getattr(nn, activation)(**kwargs)
            return nn.Identity(**kwargs)

        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            BatchNorm = ChannelsLastSyncBatchNorm
        else:
            BatchNorm = ChannelsLastBatchNorm

        self.time_pre_norm = BatchNorm(channels, eps, affine=False) if pre_norm else nn.Identity()
        self.time_attn = CausalAttention(channels, num_heads, attn_bias, freq, lookbehind)
        self.time_fc = nn.Linear(channels, channels, bias=False)
        self.time_post_norm = BatchNorm(channels, eps)
        self.time_act = Act(**activation_kwargs)

        self.freq_pre_norm = BatchNorm(channels, eps, affine=False) if pre_norm else nn.Identity()
        self.freq_attn = Attention(channels, num_heads, attn_bias)
        self.freq_fc = nn.Linear(channels, channels, bias=False)
        self.freq_post_norm = BatchNorm(channels, eps)
        self.freq_act = Act(**activation_kwargs)

        self.pe = None
        if positional_embedding is not None:
            pe = calculate_positional_embedding(channels, freq) # [F, C]
            if positional_embedding == "fixed":
                self.register_buffer("pe", pe)
                self.pe: Tensor
            elif positional_embedding == "train":
                self.pe = nn.Parameter(pe)

        self.weight_norm = weight_norm
        if weight_norm:
            self.time_attn.qkv = weight_norm_fn(self.time_attn.qkv)
            self.freq_attn.qkv = weight_norm_fn(self.freq_attn.qkv)

    def remove_weight_reparameterizations(self):
        if self.weight_norm:
            remove_parametrizations(self.time_attn.qkv, "weight")
            remove_parametrizations(self.freq_attn.qkv, "weight")
            self.weight_norm = False

        for fc, norm in (
            (self.time_fc, self.time_post_norm),
            (self.freq_fc, self.freq_post_norm)
        ):
            std = norm.running_var.add(norm.eps).sqrt()
            fc.weight.data *= norm.weight.view(-1, 1) / std.view(-1, 1)
            fc.bias = nn.Parameter(norm.bias - norm.running_mean * norm.weight / std)
        self.time_post_norm = nn.Identity()
        self.freq_post_norm = nn.Identity()

        if self.pre_norm:
            # w @ (x - mean) / std + bias
            # = (w \cdot gamma) @ x + (bias + w @ beta)
            # where gamma = 1/std, beta = -mean/std
            # 1. Time Attn
            norm = self.time_pre_norm
            std = norm.running_var.add(norm.eps).sqrt()
            beta = -norm.running_mean / std
            w_matmul_beta = (self.time_attn.qkv.weight.data @ beta.view(-1, 1)).squeeze(1)

            self.time_attn.qkv.weight.data /= std
            attn_bias = torch.zeros(self.time_attn.qkv.weight.size(0))
            if self.time_attn.qkv.bias is not None:
                attn_bias = self.time_attn.qkv.bias.data
            self.time_attn.qkv.bias = nn.Parameter(attn_bias + w_matmul_beta)
            self.time_pre_norm = nn.Identity()

            # 2. Freq Attn
            norm = self.freq_pre_norm
            std = norm.running_var.add(norm.eps).sqrt()
            beta = -norm.running_mean / std
            w_matmul_beta = (self.freq_attn.qkv.weight.data @ beta.view(-1, 1)).squeeze(1)

            self.freq_attn.qkv.weight.data /= std
            attn_bias = torch.zeros(self.freq_attn.qkv.weight.size(0))
            if self.freq_attn.qkv.bias is not None:
                attn_bias = self.freq_attn.qkv.bias.data
            self.freq_attn.qkv.bias = nn.Parameter(attn_bias + w_matmul_beta)
            self.freq_pre_norm = nn.Identity()

    def initialize_cache(self, x: Tensor) -> tp.List[Tensor]:
        return self.time_attn.initialize_cache(x)

    def forward(
        self,
        x: Tensor,
        pe: Tensor,
        h_k: tp.Optional[Tensor],
        h_v: tp.Optional[Tensor],
    ) -> tp.Tuple[Tensor, Tensor, Tensor]:
        BATCH, TIME, FREQ, CH = x.shape     # [B, T, F, C]
        x_in = x
        x = self.time_pre_norm(x)           # [B, T, F, C]
        x = x.transpose(1, 2)               # [B, F, T, C]
        x = x.reshape(BATCH*FREQ, TIME, CH) # [B*F, T, C]
        x, h_k, h_v = self.time_attn(x, pe, h_k, h_v)   # [B*F, T, C]
        x = x.view(BATCH, FREQ, TIME, CH)   # [B, F, T, C]
        x = x.transpose(1, 2)               # [B, T, F, C]
        x = self.time_fc(x)                 # [B, T, F, C]
        x = self.time_post_norm(x)          # [B, T, F, C]
        x = self.time_act(x)                # [B, T, F, C]
        x = x.add_(x_in)                    # [B, T, F, C]

        if self.pe is not None:
            x = x.add_(self.pe)
        x_in = x
        x = self.freq_pre_norm(x)           # [B, T, F, C]
        x = x.view(TIME*BATCH, FREQ, CH)    # [B*T, F, C]
        x = self.freq_attn(x)               # [B*T, F, C]
        x = x.view(BATCH, TIME, FREQ, CH)   # [B, T, F, C]
        x = self.freq_fc(x)                 # [B, T, F, C]
        x = self.freq_post_norm(x)          # [B, T, F, C]
        x = self.freq_act(x)                # [B, T, F, C]
        x = x.add_(x_in)                    # [B, T, F, C]
        return x, h_k, h_v


@dataclass
class DPTConfig:
    num_blocks: int = 3
    channels: int = 32
    freq: int = 32
    num_heads: int = 4
    eps: float = 1e-8
    positional_embedding: tp.Optional[str] = "train"    # None | "fixed" | "train"
    attn_bias: bool = False
    post_act: bool = False
    pre_norm: bool = True
    lookbehind: int = 16


def dpt_pre_post_lin(
    freq: int,
    n_filter: int,
    init: tp.Optional[str],
    bias: bool,
    sr: int = 16_000
) -> tp.Tuple[nn.Module, nn.Module]:
    assert init in [None, "mel", "mel_fixed", "linear", "linear_fixed"]
    pre = nn.Linear(freq, n_filter, bias=bias)
    post = nn.Linear(n_filter, freq, bias=bias)

    if init is None:
        return pre, post

    if init in ["mel", "mel_fixed"]:
        def hz_to_mel(freq: float) -> float:
            return 2595.0 * math.log10(1 + freq / 700)

        def mel_to_hz(mel: float) -> float:
            return 700.0 * (math.e ** (mel / 1127) - 1)

        def clip(x: int, a_min: int, a_max: int) -> int:
            return max(min(x, a_max), a_min)

        f_n = sr // 2
        mel_max = hz_to_mel(f_n)
        mel_fb = melscale_fbanks(
            n_freqs=freq,
            f_min=0.0,
            f_max=f_n,
            n_mels=n_filter,
            sample_rate=sr,
            norm='slaney',
            mel_scale='htk'
        ).float().transpose(0, 1) * f_n / freq     # [n_filter, freq]
        zeros = mel_fb.new_zeros(1)
        for idx, mfb in enumerate(mel_fb, start=0):
            if mfb.sum().isclose(zeros):
                idx_f = round(mel_to_hz(idx / n_filter * mel_max) * freq / f_n)
                idx_f = clip(idx_f, 0, freq - 1)
                mfb[idx_f] = 1.0

        mel_fb_inv = torch.linalg.pinv(mel_fb)
        for idx, mfb in enumerate(mel_fb_inv, start=0):
            if mfb.sum().isclose(zeros):
                idx_mel = round(hz_to_mel(idx / freq * f_n) * n_filter / mel_max)
                idx_mel = clip(idx_mel, 0, n_filter - 1)
                mfb[idx_mel] = 1.0
        pre_weight = mel_fb
        post_weight = mel_fb_inv

    elif init in ["linear", "linear_fixed"]:
        f_filter = torch.linspace(0, sr // 2, n_filter)
        delta_f = sr // 2 / n_filter
        f_freqs = torch.linspace(0, sr // 2, freq)
        down = f_filter[1:, None] - f_freqs[None, :]    # [n_filter - 1, freq]
        down = down / delta_f
        down = F.pad(down, (0, 0, 0, 1), value=1.0)     # [n_filter, freq]
        up = f_freqs[None, :] - f_filter[:-1, None]     # [n_filter - 1, freq]
        up = up / delta_f
        up = F.pad(up, (0, 0, 1, 0), value=1.0)         # [n_filter, freq]
        pre_weight = torch.max(up.new_zeros(1), torch.min(down, up))
        post_weight = pre_weight.transpose(0, 1)
        pre_weight = pre_weight / pre_weight.sum(dim=1, keepdim=True)
        post_weight = post_weight / post_weight.sum(dim=1, keepdim=True)

    if init.endswith("_fixed"):
        delattr(pre, "weight")
        delattr(post, "weight")
        pre.register_buffer("weight", pre_weight.contiguous().clone())
        post.register_buffer("weight", post_weight.contiguous().clone())
    else:
        pre.weight.data.copy_(pre_weight)
        post.weight.data.copy_(post_weight)

    return pre, post


class ONNXModel(nn.Module):
    def __init__(
        self,
        channels: int = 64,
        kernel_size: tp.List[int] = [8, 3, 3],
        stride: int = 4,
        dpt_kwargs: tp.Dict[str, tp.Any] = dict(),
        activation: str = "ReLU",
        activation_kwargs: tp.Dict[str, tp.Any] = dict(inplace=True),
        n_fft: int = 512,
        hop_size: int = 256,
        win_size: int = 512,
        window: tp.Optional[str] = "hann",
        stft_normalized: bool = False,
        mask: tp.Optional[str] = None,
        input_compression: float = 0.3,
        weight_norm: bool = False,
        final_scale: tp.Union[bool, str] = "exp",
        final_scale_init: str = "1/sqrt(fan_in)",
        normalize_final_conv: bool = False,
        pre_post_init: tp.Optional[str] = None,
    ):
        assert final_scale in [True, False, "exp"]
        super().__init__()
        self.input_compression = input_compression
        self.stft = self.get_stft(n_fft, hop_size, win_size, window, stft_normalized)
        dpt_config = DPTConfig(**dpt_kwargs)
        self.dpt_ch = dpt_config.channels
        self.dpt_freq = dpt_config.freq
        if mask is None:
            self.mask = nn.Identity()
        elif mask == "sigmoid":
            self.mask = nn.Sigmoid()
        elif mask == "tanh":
            self.mask = nn.Tanh()
        else:
            raise RuntimeError(f"model_kwargs.mask={mask} is not supported.")
        self.weight_norm = weight_norm

        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            BatchNorm = nn.SyncBatchNorm
        else:
            BatchNorm = nn.BatchNorm1d

        def norm(module):
            if self.weight_norm:
                return weight_norm_fn(module)
            return module

        Act = getattr(nn, activation)

        # Encoder PreNet
        assert kernel_size[0] % stride == 0
        assert (kernel_size[0] - stride) % 2 == 0
        self.enc_pre = nn.Sequential(
            StridedConv1d(  # in_channels = 2 = [real, imag]
                2, channels, kernel_size[0], stride=stride,
                padding=(kernel_size[0] - stride) // 2, bias=False
            ),
            BatchNorm(channels),
            Act(**activation_kwargs),
        )

        # Encoder
        self.encoder = nn.ModuleList()
        for idx in range(1, len(kernel_size)):
            module = nn.Sequential(
                nn.Conv1d(
                    channels, channels, kernel_size[idx], 
                    padding=(kernel_size[idx] - 1) // 2, bias=False
                ),
                BatchNorm(channels),
                Act(**activation_kwargs),
            )
            self.encoder.append(module)

        # Pre DPT Net
        freq = n_fft // 2 // stride
        dpt_pre, dpt_post = dpt_pre_post_lin(freq, self.dpt_freq, pre_post_init, bias=False)
        self.dpt_pre = nn.Sequential(
            dpt_pre,
            nn.Conv1d(channels, self.dpt_ch, 1, bias=False),
            BatchNorm(self.dpt_ch),
        )

        # Dual-Path Transformer Blocks
        pe = calculate_positional_embedding(
            dpt_config.num_heads,
            dpt_config.lookbehind + 1
        )    # [L+1, NH]
        self.pe = nn.Parameter(pe.transpose(0, 1).contiguous())     # [NH, L+1]
        dpt_list = []
        for _ in range(dpt_config.num_blocks):
            block = DPTBlock(
                self.dpt_ch, self.dpt_freq,
                dpt_config.num_heads,
                eps=dpt_config.eps, weight_norm=weight_norm,
                activation=activation, activation_kwargs=activation_kwargs,
                positional_embedding=dpt_config.positional_embedding,
                attn_bias=dpt_config.attn_bias,
                post_act=dpt_config.post_act,
                pre_norm=dpt_config.pre_norm,
                lookbehind=dpt_config.lookbehind,
            )
            dpt_list.append(block)
            dpt_config.positional_embedding = None
        self.dpt_block = nn.ModuleList(dpt_list)

        # Post DPT Net
        self.dpt_post = nn.Sequential(
            dpt_post,
            nn.Conv1d(self.dpt_ch, channels, 1, bias=False),
            BatchNorm(channels),
        )

        # Decoder
        self.decoder = nn.ModuleList()
        for idx in range(len(kernel_size)-1, 0, -1):
            module = nn.Sequential(
                nn.Conv1d(channels*2, channels, 1, bias=False),
                BatchNorm(channels),
                Act(**activation_kwargs),
                nn.Conv1d(
                    channels, channels, kernel_size[idx],
                    padding=(kernel_size[idx] - 1) // 2, bias=False
                ),
                BatchNorm(channels),
                Act(**activation_kwargs),
            )
            self.decoder.append(module)

        # Decoder PostNet
        padding = (kernel_size[0] - stride) // 2
        if final_scale != False:
            # out_channels = 2 = [real, imag] of the mask
            upsample = ScaledConvTranspose1d(
                channels, 2, kernel_size[0], stride=stride,
                padding=padding, bias=True, exp_scale=(final_scale == "exp"),
                normalize=normalize_final_conv,
                final_scale_init=final_scale_init,
            )
        else:
            upsample = nn.ConvTranspose1d(
                channels, 2, kernel_size[0], stride=stride,
                padding=padding, bias=True
            )
        self.dec_post = nn.Sequential(
            nn.Conv1d(channels*2, channels, 1, bias=False),
            BatchNorm(channels),
            Act(**activation_kwargs),
            upsample
        )

    def get_stft(
        self, n_fft: int, hop_size: int, win_size: int,
        window: str, normalized: bool
    ) -> nn.Module:
        return ONNXSTFT(
            n_fft=n_fft, hop_size=hop_size, win_size=win_size,
            win_type=window, normalized=normalized
        )

    @torch.no_grad()
    def remove_weight_reparameterizations(self):
        """ 1. Remove weight_norm """
        if self.weight_norm:
            with torch.enable_grad():
                # DPT
                for block in self.dpt_block:
                    block.remove_weight_reparameterizations()

                # Decoder
                self.dec_post[3].remove_weight_reparameterizations()
            self.weight_norm = False

        """ 2. Merge BatchNorm into Conv
        y = (conv(x) - mean) / std * gamma + beta \
          = conv(x) * (gamma / std) + (beta - mean * gamma / std)
        <=> y = conv'(x) where
          W'[c, :, :] = W[c, :, :] * (gamma / std)
          b' = (beta - mean * gamma / std)
        """
        def merge_conv_bn(conv: nn.Module, norm: nn.Module, error_message: str = "") -> nn.Module:
            assert conv.bias is None, error_message
            std = norm.running_var.add(norm.eps).sqrt()
            conv.weight.data *= norm.weight.view(-1, 1, 1) / std.view(-1, 1, 1)
            conv.bias = nn.Parameter(norm.bias - norm.running_mean * norm.weight / std)
            return conv

        # Encoder PreNet
        conv = merge_conv_bn(self.enc_pre[0], self.enc_pre[1], "enc_pre")
        self.enc_pre = nn.Sequential(conv, self.enc_pre[2])

        # Encoder
        new_encoder = nn.ModuleList()
        for idx, module in enumerate(self.encoder):
            conv = merge_conv_bn(module[0], module[1], f"enc.{idx}")
            new_module = nn.Sequential(
                conv,       # Conv-BN Merged
                module[2],  # Activation
            )
            new_encoder.append(new_module)
        self.encoder = new_encoder

        # Pre DPT Net
        conv = merge_conv_bn(self.dpt_pre[1], self.dpt_pre[2], "dpt_pre")
        self.dpt_pre = nn.Sequential(
            self.dpt_pre[0],     # Linear
            conv,               # Conv-BN Merged
        )

        # Post DPT Net
        conv = merge_conv_bn(self.dpt_post[1], self.dpt_post[2], "dpt_post")
        self.dpt_post = nn.Sequential(
            self.dpt_post[0],    # Linear
            conv,               # Conv-BN Merged
        )

        # Decoder
        new_decoder = nn.ModuleList()
        for idx, module in enumerate(self.decoder):
            conv1 = merge_conv_bn(module[0], module[1], f"dec.{idx}.0")
            conv2 = merge_conv_bn(module[3], module[4], f"dec.{idx}.1")
            new_module = nn.Sequential(
                conv1,      # Conv-BN Merged
                module[2],  # Activation
                conv2,      # Conv-BN Merged
                module[5],  # Activation
            )
            new_decoder.append(new_module)
        self.decoder = new_decoder

        # Decoder PostNet
        conv = merge_conv_bn(self.dec_post[0], self.dec_post[1], "dec_post")
        self.dec_post = nn.Sequential(
            conv,               # Conv-BN Merged
            self.dec_post[2],   # Activation
            self.dec_post[3]    # Transposed Convolution
        )

    def flatten_parameters(self):
        pass

    def initialize_cache(self, x: Tensor) -> tp.List[Tensor]:
        cache_list = []
        for block in self.dpt_block:
            cache_list.extend(block.initialize_cache(x))
        return cache_list

    def model_forward(self, spec_noisy: Tensor, *args) -> tp.Tuple[Tensor, tp.List[Tensor]]:
        # spec_noisy: [B, F, T, 2]
        cache_in_list = [*args]
        cache_out_list = []
        if len(cache_in_list) == 0:
            cache_in_list = [None for _ in range(len(self.dpt_block) * 2)]
        x = spec_noisy

        B, FREQ, T, _ = x.shape
        x = x.permute(0, 2, 3, 1)       # [B, T, 2, F]
        x = x.reshape(B*T, 2, FREQ)     # [B*T, 2, F]

        # Encoder PreNet
        x = self.enc_pre(x)
        encoder_outs = [x]

        # Encoder
        for module in self.encoder:
            x = module(x)
            encoder_outs.append(x)      # [B*T, C, F']

        # DPT
        x_in = x
        x = self.dpt_pre(x)              # [B*T, C, F']
        _, C, _FREQ = x.shape
        x = x.view(B, T, C, _FREQ)      # [B, T, C, F']
        x = x.transpose(2, 3)           # [B, T, F', C]
        x = x.contiguous()
        for idx, block in enumerate(self.dpt_block):
            h_k, h_v = cache_in_list[idx*2:idx*2+2]
            x, h_k, h_v = block(x, self.pe, h_k, h_v)   # [B, T, F', C]
            cache_out_list.append(h_k)
            cache_out_list.append(h_v)
        x = x.transpose(2, 3)           # [B, T, C, F']
        x = x.reshape(B*T, C, _FREQ)    # [B*T, C, F']
        x = self.dpt_post(x)            # [B*T, C, F']

        # Decoder
        for module in self.decoder:
            x = torch.cat([x, encoder_outs.pop(-1)], dim=1)     # [B*T, 2*C, F]
            x = module(x)                                       # [B*T, C, F] or [B*T, 2, F]

        # Decoder PostNet
        x = torch.cat([x, encoder_outs.pop(-1)], dim=1)     # [B*T, 2*C, F]
        x = self.dec_post(x)                                # [B*T, 2, F]
        x = x.reshape(B, T, 2, FREQ).permute(0, 3, 1, 2)    # [B, F, T, 2]

        # Mask
        mask = self.mask(x).contiguous()
        return mask, cache_out_list

    def forward(self, spec_noisy: Tensor, *args) -> tp.Tuple[Tensor, tp.List[Tensor]]:
        # Remove n_fft//2+1 & compress
        spec_noisy = spec_noisy[:, :-1, :, :]    # [B, F, T, 2]
        mag = torch.linalg.norm(
            spec_noisy,
            dim=-1,
            keepdim=True
        ).clamp(min=1.0e-5)
        spec_noisy = spec_noisy * mag.pow(self.input_compression - 1.0)

        # Model forward
        mask, cache_out_list = self.model_forward(spec_noisy, *args)
        spec_hat = torch.stack(
            [
                spec_noisy[..., 0] * mask[..., 0] - spec_noisy[..., 1] * mask[..., 1],
                spec_noisy[..., 0] * mask[..., 1] + spec_noisy[..., 1] * mask[..., 0],
            ],
            dim=3
        )

        # Uncompress & pad n_fft//2+1
        mag_compressed = torch.linalg.norm(
            spec_hat,
            dim=-1,
            keepdim=True
        )
        spec_hat = spec_hat * mag_compressed.pow(1.0 / self.input_compression - 1.0)
        spec_hat = F.pad(spec_hat, (0, 0, 0, 0, 0, 1))    # [B, F+1, T, 2]
        return spec_hat, *cache_out_list


class Model(ONNXModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_stft(
        self, n_fft: int, hop_size: int, win_size: int,
        window: str, normalized: bool
    ) -> nn.Module:
        return CompressedSTFT(
            n_fft=n_fft, hop_size=hop_size, win_size=win_size,
            win_type=window, normalized=normalized,
            compression=self.input_compression,
            discard_last_freq_bin=True,
        )

    def forward(self, noisy: Tensor) -> tp.Tuple[Tensor, Tensor]:
        # farend/nearend_mic: [B, T_wav]
        spec_noisy = self.stft(noisy)                   # [B, F, T, 2]
        mask, _ = self.model_forward(spec_noisy)        # [B, F, T, 2]
        spec_hat = torch.view_as_complex(spec_noisy) \
            * torch.view_as_complex(mask)       # [B, F, T]
        wav_hat = self.stft.inverse(spec_hat)   # [B, T_wav]
        return wav_hat, torch.view_as_real(spec_hat)


def test():
    x = torch.randn(3, 16_000)
    from utils import get_hparams
    hparams = get_hparams("configs/rnnformer/size/dpt_b.yaml")
    # hparams["model_kwargs"]["weight_norm"] = False
    model = Model(**hparams["model_kwargs"])
    wav_out, spec_out = model(x)
    (wav_out.sum() + spec_out.sum()).backward()
    print(spec_out.shape)

    model.remove_weight_reparameterizations()
    total_params = sum(p.numel() for n, p in model.named_parameters())
    print(f"Number of total parameters: {total_params}")
    # for n, p in model.named_parameters():
    #     print(n, p.shape)


if __name__ == "__main__":
    test()
