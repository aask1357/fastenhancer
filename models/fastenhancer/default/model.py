import math
import typing as tp
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm as weight_norm_fn
from torch.nn.utils.parametrize import remove_parametrizations
from torch import Tensor

from functional import ONNXSTFT, CompressedSTFT


class StridedConv1d(nn.Conv1d):
    """Same as nn.Conv1d with stride > 1.
    We just want to show that MAC of StridedConv is not Cin x Cout x K x T (ptflops calculation),
    but Cin x Cout x K x (T / S) where K: kernel_size, T: time, S: stride."""
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
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.normalize = normalize
        self.scale = nn.Parameter(torch.ones(1))
        self.weight_norm = True

    def remove_weight_reparameterizations(self):
        if self.normalize:
            weight = F.normalize(self.weight, dim=(0, 1, 2)).mul_(self.scale)
        else:
            weight = self.weight * self.scale
        self.weight.data.copy_(weight)
        self.weight_norm = False
        self.scale = None

    def forward(self, x: Tensor) -> Tensor:
        if self.weight_norm:
            if self.normalize:
                weight = F.normalize(self.weight, dim=(0, 1, 2)).mul_(self.scale)
            else:
                weight = self.weight * self.scale
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


class RNNFormerBlock(nn.Module):
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
        p_dropout: float = 0.0,
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

        self.rnn_pre_norm = BatchNorm(channels, eps, affine=False) if pre_norm else nn.Identity()
        self.rnn = nn.GRU(channels, channels, batch_first=False)
        self.rnn_fc = nn.Linear(channels, channels, bias=False)
        self.rnn_post_norm = BatchNorm(channels, eps)
        self.rnn_act = Act(**activation_kwargs)

        self.attn_pre_norm = BatchNorm(channels, eps, affine=False) if pre_norm else nn.Identity()
        self.attn = Attention(channels, num_heads, attn_bias)
        self.attn_fc = nn.Linear(channels, channels, bias=False)
        self.attn_post_norm = BatchNorm(channels, eps)
        self.attn_act = Act(**activation_kwargs)

        self.dropout = nn.Identity() if p_dropout == 0.0 else nn.Dropout(p=p_dropout, inplace=True)

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
            self.rnn = weight_norm_fn(self.rnn, name="weight_ih_l0")
            self.rnn = weight_norm_fn(self.rnn, name="weight_hh_l0")
            self.attn.qkv = weight_norm_fn(self.attn.qkv)

    def remove_weight_reparameterizations(self):
        if self.weight_norm:
            remove_parametrizations(self.rnn, "weight_ih_l0")
            remove_parametrizations(self.rnn, "weight_hh_l0")
            remove_parametrizations(self.attn.qkv, "weight")
            self.flatten_parameters()
            self.weight_norm = False

        for fc, norm in (
            (self.rnn_fc, self.rnn_post_norm),
            (self.attn_fc, self.attn_post_norm)
        ):
            std = norm.running_var.add(norm.eps).sqrt()
            fc.weight.data *= norm.weight.view(-1, 1) / std.view(-1, 1)
            fc.bias = nn.Parameter(norm.bias - norm.running_mean * norm.weight / std)
        self.rnn_post_norm = nn.Identity()
        self.attn_post_norm = nn.Identity()

        if self.pre_norm:
            # w @ (x - mean) / std + bias
            # = (w \cdot gamma) @ x + (bias + w @ beta)
            # where gamma = 1/std, beta = -mean/std
            # 1. Attn
            norm = self.attn_pre_norm
            std = norm.running_var.add(norm.eps).sqrt()
            beta = -norm.running_mean / std
            w_matmul_beta = (self.attn.qkv.weight.data @ beta.view(-1, 1)).squeeze(1)

            self.attn.qkv.weight.data /= std
            attn_bias = torch.zeros(self.attn.qkv.weight.size(0))
            if self.attn.qkv.bias is not None:
                attn_bias = self.attn.qkv.bias.data
            self.attn.qkv.bias = nn.Parameter(attn_bias + w_matmul_beta)
            self.attn_pre_norm = nn.Identity()

            # 2. GRU
            norm = self.rnn_pre_norm
            std = norm.running_var.add(norm.eps).sqrt()
            beta = -norm.running_mean / std
            w_matmul_beta = (self.rnn.weight_ih_l0.data @ beta.view(-1, 1)).squeeze(1)

            self.rnn.weight_ih_l0.data /= std
            self.rnn.bias_ih_l0.data.add_(w_matmul_beta)
            self.rnn_pre_norm = nn.Identity()

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def initialize_cache(self, x: Tensor) -> Tensor:
        return x.new_zeros(1, self.freq, self.rnn.hidden_size)

    def forward(self, x: Tensor, h: tp.Optional[Tensor]) -> tp.Tuple[Tensor, Tensor]:
        TIME, BATCH, FREQ, CH = x.shape     # [T, B, F, C]
        x_in = x
        x = self.rnn_pre_norm(x)            # [T, B, F, C]
        x = x.view(TIME, FREQ*BATCH, CH)    # [T, B*F, C]
        x, h = self.rnn(x, h)               # [T, B*F, C]
        x = x.view(TIME, BATCH, FREQ, CH)   # [T, B, F, C]
        x = self.rnn_fc(x)                  # [T, B, F, C]
        x = self.dropout(x)
        x = self.rnn_post_norm(x)           # [T, B, F, C]
        x = self.rnn_act(x)                 # [T, B, F, C]
        x = x.add_(x_in)                    # [T, B, F, C]

        if self.pe is not None:
            x = x.add_(self.pe)
        x_in = x
        x = self.attn_pre_norm(x)           # [T, B, F, C]
        x = x.view(TIME*BATCH, FREQ, CH)    # [T*B, F, C]
        x = self.attn(x)                    # [T*B, F, C]
        x = x.view(TIME, BATCH, FREQ, CH)   # [T, B, F, C]
        x = self.attn_fc(x)                 # [T, B, F, C]
        x = self.dropout(x)
        x = self.attn_post_norm(x)          # [T, B, F, C]
        x = self.attn_act(x)                # [T, B, F, C]
        x = x.add_(x_in)                    # [T, B, F, C
        return x, h


@dataclass
class RNNFormerConfig:
    num_blocks: int = 3
    channels: int = 32
    freq: int = 32
    num_heads: int = 4
    eps: float = 1e-8
    positional_embedding: tp.Optional[str] = "train"    # None | "fixed" | "train"
    attn_bias: bool = False
    post_act: bool = False
    pre_norm: bool = False
    p_dropout: float = 0.0


def rf_pre_post_lin(
    freq: int,
    n_filter: int,
    init: tp.Optional[str],
    bias: bool,
    sr: int = 16_000
) -> tp.Tuple[nn.Module, nn.Module]:
    assert init in [None, "linear", "linear_fixed"]
    pre = nn.Linear(freq, n_filter, bias=bias)
    post = nn.Linear(n_filter, freq, bias=bias)

    if init is None:
        return pre, post

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
        rnnformer_kwargs: tp.Dict[str, tp.Any] = dict(),
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
        normalize_final_conv: bool = False,
        pre_post_init: tp.Optional[str] = None,
        resnet: bool = False,
    ):
        super().__init__()
        self.input_compression = input_compression
        self.stft = self.get_stft(n_fft, hop_size, win_size, window, stft_normalized)
        rnnformer_config = RNNFormerConfig(**rnnformer_kwargs)
        self.rf_ch = rnnformer_config.channels
        self.rf_freq = rnnformer_config.freq
        if mask is None:
            self.mask = nn.Identity()
        elif mask == "sigmoid":
            self.mask = nn.Sigmoid()
        elif mask == "tanh":
            self.mask = nn.Tanh()
        else:
            raise RuntimeError(f"model_kwargs.mask={mask} is not supported.")
        self.weight_norm = weight_norm
        self.resnet = resnet

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

        # RNNFormer PreNet
        freq = n_fft // 2 // stride
        rf_pre, rf_post = rf_pre_post_lin(freq, self.rf_freq, pre_post_init, bias=False)
        self.rf_pre = nn.Sequential(
            rf_pre,
            nn.Conv1d(channels, self.rf_ch, 1, bias=False),
            BatchNorm(self.rf_ch),
        )

        # RNNFormer Blocks
        rf_list = []
        for _ in range(rnnformer_config.num_blocks):
            block = RNNFormerBlock(
                self.rf_ch, self.rf_freq,
                rnnformer_config.num_heads,
                eps=rnnformer_config.eps, weight_norm=weight_norm,
                activation=activation, activation_kwargs=activation_kwargs,
                positional_embedding=rnnformer_config.positional_embedding,
                attn_bias=rnnformer_config.attn_bias,
                post_act=rnnformer_config.post_act,
                pre_norm=rnnformer_config.pre_norm,
                p_dropout=rnnformer_config.p_dropout,
            )
            rf_list.append(block)
            rnnformer_config.positional_embedding = None
        self.rf_block = nn.ModuleList(rf_list)

        # RNNFormer PostNet
        self.rf_post = nn.Sequential(
            rf_post,
            nn.Conv1d(self.rf_ch, channels, 1, bias=False),
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
        # out_channels = 2 = [real, imag] of the mask
        upsample = ScaledConvTranspose1d(
            channels, 2, kernel_size[0], stride=stride,
            bias=True,
            padding=(kernel_size[0] - stride) // 2,
            normalize=normalize_final_conv,
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
                # RNNFormer
                for block in self.rf_block:
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

        # RNNFormer PreNet
        conv = merge_conv_bn(self.rf_pre[1], self.rf_pre[2], "rf_pre")
        self.rf_pre = nn.Sequential(
            self.rf_pre[0],     # Linear
            conv,               # Conv-BN Merged
        )

        # RNNFormer PostNet
        conv = merge_conv_bn(self.rf_post[1], self.rf_post[2], "rf_post")
        self.rf_post = nn.Sequential(
            self.rf_post[0],    # Linear
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
        for rf in self.rf_block:
            rf.flatten_parameters()

    def initialize_cache(self, x: Tensor) -> tp.List[Tensor]:
        cache_list = []
        for block in self.rf_block:
            cache_list.append(block.initialize_cache(x))
        return cache_list

    def model_forward(self, spec_noisy: Tensor, *args) -> tp.Tuple[Tensor, tp.List[Tensor]]:
        # spec_noisy: [B, F, T, 2]
        cache_in_list = [*args]
        cache_out_list = []
        if len(cache_in_list) == 0:
            cache_in_list = [None for _ in range(len(self.rf_block))]
        x = spec_noisy

        B, FREQ, T, _ = x.shape
        x = x.permute(0, 2, 3, 1)       # [B, T, 2, F]
        x = x.reshape(B*T, 2, FREQ)     # [B*T, 2, F]

        # Encoder PreNet
        x = self.enc_pre(x)
        encoder_outs = [x]

        # Encoder
        for module in self.encoder:
            x_in = x
            x = module(x)
            encoder_outs.append(x)      # [B*T, C, F']
            if self.resnet:
                x = x.add_(x_in)

        # RNNFormer
        x_in = x
        x = self.rf_pre(x)              # [B*T, C, F']
        _, C, _FREQ = x.shape
        x = x.view(B, T, C, _FREQ)      # [B, T, C, F']
        x = x.permute(1, 0, 3, 2)       # [T, B, F', C]
        x = x.contiguous()
        for block, cache_in in zip(self.rf_block, cache_in_list):
            x, cache_out = block(x, cache_in)   # [T, B, F', C]
            cache_out_list.append(cache_out)
        x = x.permute(1, 0, 3, 2)       # [B, T, C, F']
        x = x.reshape(B*T, C, _FREQ)    # [B*T, C, F']
        x = self.rf_post(x)             # [B*T, C, F']
        if self.resnet:
            x = x.add_(x_in)

        # Decoder
        for module in self.decoder:
            x_in = x
            x = torch.cat([x, encoder_outs.pop(-1)], dim=1)     # [B*T, 2*C, F]
            x = module(x)                                       # [B*T, C, F] or [B*T, 2, F]
            if self.resnet:
                x = x.add_(x_in)

        # Decoder PostNet
        x = torch.cat([x, encoder_outs.pop(-1)], dim=1)     # [B*T, 2*C, F]
        x = self.dec_post(x)                                # [B*T, 2, F]
        x = x.reshape(B, T, 2, FREQ).permute(0, 3, 1, 2)    # [B, F, T, 2]

        # Mask
        mask = self.mask(x).contiguous()
        return mask, cache_out_list

    def forward(
        self,
        spec_noisy: Tensor,
        *args
    ) -> tp.Tuple[Tensor, Tensor, ...]:
        """ input/output: [B, n_fft//2+1, T_spec, 2]"""
        # Compress
        spec_noisy = spec_noisy[:, :-1, :, :]   # [B, n_fft//2, T_spec, 2]
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
        )   # [B, F, T, 2]

        # Uncompress
        mag_compressed = torch.linalg.norm(
            spec_hat,
            dim=3,
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
        """ input/output: [B, T_wav]"""
        spec_noisy = self.stft(noisy)                   # [B, F, T, 2]
        mask, _ = self.model_forward(spec_noisy)        # [B, F, T, 2]
        spec_hat = torch.view_as_complex(spec_noisy) \
            * torch.view_as_complex(mask)       # [B, F, T]
        wav_hat = self.stft.inverse(spec_hat)   # [B, T_wav]
        return wav_hat, torch.view_as_real(spec_hat)


def test():
    x = torch.randn(3, 16_000)
    from utils import get_hparams
    hparams = get_hparams("configs/fastenhancer/t.yaml")
    model = Model(**hparams["model_kwargs"])
    wav_out, spec_out = model(x)
    (wav_out.sum() + spec_out.sum()).backward()
    print(spec_out.shape)

    model.remove_weight_reparameterizations()
    model.flatten_parameters()
    total_params = sum(p.numel() for n, p in model.named_parameters())
    print(f"Number of total parameters: {total_params}")
    # for n, p in model.named_parameters():
    #     print(n, p.shape)


if __name__ == "__main__":
    test()
