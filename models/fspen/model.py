from typing import List, Any
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor

from functional import ONNXSTFT, CompressedSTFT


class LayerNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1.0e-5, affine: bool = True):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(channels))
            self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: Tensor) -> Tensor:
        """input/output: [..., F, C]
        1. Normalize across [F, C] dimension
        2. Each weight & bias has a shape of [c]
        """
        mean = x.mean(dim=(-2, -1), keepdim=True)
        diff = x.sub(mean)
        variance = diff.square().mean(dim=(-2, -1), keepdim=True)
        inv_std = variance.add_(self.eps).rsqrt()
        if self.affine:
            w = inv_std.mul(self.weight)
            x = diff.addcmul(w, self.bias)
        else:
            x = diff.mul(inv_std)
        return x


class SubbandEncoder(nn.Module):
    def __init__(
        self,
        out_ch=32,
        kernel_size=[4, 7, 11, 20, 40],
        stride=[2, 3, 5, 10, 20],
    ) -> None:
        super().__init__()
        def build_conv(idx):
            return nn.Sequential(
                nn.Conv1d(1, out_ch, kernel_size[idx], stride[idx]),
                nn.ReLU(inplace=True)
            )
        self.conv1 = build_conv(0)
        self.conv2 = build_conv(1)
        self.conv3 = build_conv(2)
        self.conv4 = build_conv(3)
        self.conv5 = build_conv(4)

    def forward(self, x):
        # x: [B*T, 1, F=257]
        x1 = self.conv1(F.pad(x[...,:17], (1, 0)))  # Fin=18  -> conv(k=4, s=2)   -> Fout=8
        x2 = self.conv2(x[...,13:35])               # Fin=22  -> conv(k=7, s=3)   -> Fout=6
        x3 = self.conv3(x[...,30:66])               # Fin=36  -> conv(k=11, s=5)  -> Fout=6
        x4 = self.conv4(x[...,61:131])              # Fin=70  -> conv(k=20, s=10) -> Fout=6
        x5 = self.conv5(F.pad(x[...,122:], (0, 5))) # Fin=140 -> conv(k=40, s=20) -> Fout=6

        return torch.cat([x1, x2, x3, x4, x5], dim=2)   # [B*T, 1, 32]


class SubbandDecoder(nn.Module):
    def __init__(self, in_ch=32, out_ch=[2, 3, 5, 10, 20]) -> None:
        super().__init__()
        def build_lin(idx):
            return nn.Sequential(
                nn.Linear(in_ch*2, out_ch[idx]),
                nn.ReLU(inplace=True)
            )
        self.lin1 = build_lin(0)
        self.lin2 = build_lin(1)
        self.lin3 = build_lin(2)
        self.lin4 = build_lin(3)
        self.lin5 = build_lin(4)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B*T, C, F]
        BT = x.size(0)
        x = x.transpose(1, 2)   # [B*T, F, C]
        x1 = self.lin1(x[:, :8, :]).view(BT, -1)                # 8*2 = 16
        x2 = self.lin2(x[:, 8:14, :]).view(BT, -1)[:, 1:17]     # 6*3 = 18 -> 16
        x3 = self.lin3(x[:, 13:21, :]).view(BT, -1)[:, 4:36]    # 8*5 = 40 -> 32
        x4 = self.lin4(x[:, 19:27, :]).view(BT, -1)[:, 8:72]    # 8*10 = 80 -> 64
        x5 = self.lin5(
            F.pad(x[:, 25:32, :], (0, 0, 0, 1))                 # 7 -> pad -> 8
        ).view(BT, -1)[:, 16:145]                               # 8*20 = 160 -> 129

        return torch.cat([x1, x2, x3, x4, x5], 1)   # [B*T, 257]


class InterRNNPathExtension(nn.Module):
    def __init__(self, channels, freq, groups=1) -> None:
        super().__init__()
        self.channels = channels
        self.freq = freq
        self.groups = groups
        self.inter_rnn = nn.ModuleList()
        self.inter_fc = nn.ModuleList()
        assert freq % groups == 0, f"freq {freq} must be divided by groups {groups}"
        for _ in range(0, groups):
            self.inter_rnn.append(nn.GRU(channels, channels, batch_first=False))
            self.inter_fc.append(nn.Linear(channels, channels))

    def initialize_cache(self, x: Tensor) -> List[Tensor]:
        cache_list = []
        for i in range(0, self.groups):
            h = x.new_zeros(1, self.freq // 8, self.channels)
            cache_list.append(h)
        return cache_list

    def flatten_parameters(self):
        for rnn in self.inter_rnn:
            rnn.flatten_parameters()

    def forward(self, x: Tensor, *h_in_list):
        TIME, BATCH, FREQ, CH = x.shape     # [T, B, F, C]

        x_in = x
        x_chunked = torch.chunk(x, chunks=self.groups, dim=2)   # List of [T, B, F//groups, C]
        rnn_out = []
        h_out_list = []
        for i in range(self.groups):
            x = x_chunked[i].reshape(TIME, -1, CH)      # [T, B*F//G, C]
            x, h = self.inter_rnn[i](x, h_in_list[i])   # [T, B*F//G, C]
            h_out_list.append(h)
            x = self.inter_fc[i](x)             # [T, B*F//G, C]
            x = x.view(TIME, BATCH, -1, CH)     # [T, B, F//G, C]
            rnn_out.append(x)
        x = torch.cat(rnn_out, 2)   # [T, B, F, C]
        x.add_(x_in)
        return x, *h_out_list


class DPE(nn.Module):
    def __init__(self, channels: int, freq: int, groups: int, norm: str, **kwargs) -> None:
        super().__init__()
        self.freq = freq
        self.channels = channels

        self.intra_rnn = nn.GRU(
            input_size=channels,
            hidden_size=channels,
            batch_first=True,
            bidirectional=True
        )
        self.intra_fc = nn.Linear(channels*2, channels)
        if norm == "LayerNorm-FreqChannels":
            self.intra_ln = nn.LayerNorm([freq, channels])
        elif norm == "LayerNorm-Channels":
            self.intra_ln = nn.LayerNorm([channels])
        elif norm == "CustomLayerNorm":
            self.intra_ln = LayerNorm(channels)
        else:
            raise RuntimeError(f"dpe_kwargs.norm {norm} is not supported")
 
        self.inter_rnn = InterRNNPathExtension(channels, freq, groups=groups)

    def flatten_parameters(self):
        for rnn in [self.intra_rnn, self.inter_rnn]:
            rnn.flatten_parameters()

    def initialize_cache(self, x: Tensor) -> List[Tensor]:
        return self.inter_rnn.initialize_cache(x)

    def forward(self, x, *cache_in_list):
        TIME, BATCH, FREQ, CH = x.shape     # [T, B, F, C]

        # Intra RNN (Time Bi-GRU)
        x_in = x
        x = x.view(TIME*BATCH, FREQ, CH)    # [T*B, F, C]
        x = self.intra_rnn(x)[0]
        x = self.intra_fc(x)
        x = self.intra_ln(x)
        x = x.view(TIME, BATCH, FREQ, CH)
        x = x.add_(x_in)

        # Inter RNN (Frequency Uni-GRU)
        x_in = x
        x, *cache_out_list = self.inter_rnn(x, *cache_in_list)
        x = x.add_(x_in)
        return x, *cache_out_list


@dataclass
class DPEConfig:
    num_blocks: int = 3
    channels: int = 16
    freq: int = 32
    groups: int = 8
    norm: str = "LayerNorm-FreqChannels"


class ONNXModel(nn.Module):
    def __init__(
        self,
        channels=[4, 16, 32],
        kernel_size=[6, 8, 6],
        stride=[2, 2, 2],
        dpe_kwargs: dict[str, Any] = dict(),
        n_fft: int = 512,
        hop_size: int = 256,
        win_size: int = 512,
        window: str = 'hann',
        input_compression: float = 0.3,
    ) -> None:
        assert n_fft == 512, f"Only n_fft == 512 is allowed, but given {n_fft}."
        super().__init__()
        dpe_config = DPEConfig(**dpe_kwargs)
        self.stft = self.get_stft(
            n_fft=n_fft,
            hop_size=hop_size,
            win_size=win_size,
            window=window,
            normalized=False,
            input_compression=input_compression,
        )
        self.input_compression = input_compression

        self.subband_encoder = SubbandEncoder(channels[-1])
        self.subband_decoder = SubbandDecoder(channels[-1])

        self.fullband_encoder = nn.ModuleList()
        for i in range(len(channels)):
            cin = 2 if i == 0 else channels[i-1]
            cout = channels[i]
            k = kernel_size[i]
            s = stride[i]
            encoder = nn.Sequential(
                nn.Conv1d(
                    cin, cout, k, s, padding=(k - s) // 2, bias=False
                ),
                nn.BatchNorm1d(cout),
                nn.ELU(inplace=True),
            )
            self.fullband_encoder.append(encoder)
        self.fullband_encoder_post = nn.Conv1d(channels[-1], channels[-1], 1, bias=False)

        self.feature_merge = nn.Sequential(
            nn.Linear(64, dpe_config.freq, bias=False),
            nn.ELU(inplace=True),
            nn.Conv1d(channels[-1], dpe_config.channels, 1, bias=True),
        )

        self.dpe_blocks = nn.ModuleList([
            DPE(**asdict(dpe_config)) for _ in range(dpe_config.num_blocks)
        ])

        self.feature_split = nn.Sequential(
            nn.Conv1d(dpe_config.channels, channels[-1], 1, bias=True),
            nn.Linear(dpe_config.freq, 64, bias=False),
            nn.ELU(inplace=True) 
        )

        self.fullband_decoder = nn.ModuleList()
        for i in range(len(channels)-1, -1, -1):
            cin = channels[i]
            cout = 2 if i == 0 else channels[i-1]
            k = kernel_size[i]
            s = stride[i]
            decoder = nn.Sequential(
                nn.Conv1d(cin*2, cin, 1, bias=False),
                nn.ConvTranspose1d(
                    cin, cout, k, s, padding=(k-s)//2,
                    output_padding=1 if i == 0 else 0,
                    bias=True if i == 0 else 0,
                ),
                nn.Identity() if i == 0 else nn.BatchNorm1d(cout),
                nn.Identity() if i == 0 else nn.ELU(inplace=True),
            )
            self.fullband_decoder.append(decoder)

    def get_stft(
        self, n_fft: int, hop_size: int, win_size: int,
        window: str, normalized: bool, input_compression: float,
    ) -> nn.Module:
        return ONNXSTFT(
            n_fft=n_fft, hop_size=hop_size, win_size=win_size,
            win_type=window, normalized=normalized
        )

    def flatten_parameters(self):
        for dpe in self.dpe_blocks:
            dpe.flatten_parameters()


    def initialize_cache(self, x: Tensor) -> List[Tensor]:
        cache_list = []
        for dpe in self.dpe_blocks:
            cache_list.extend(dpe.initialize_cache(x))
        return cache_list

    def remove_weight_reparameterizations(self):
        def merge_conv_bn(
            conv: nn.Conv1d,
            norm: nn.BatchNorm1d,
            error_message: str = ""
        ) -> nn.Conv1d:
            assert conv.bias is None, error_message
            std = norm.running_var.add(norm.eps).sqrt()
            conv.weight.data *= norm.weight.view(-1, 1, 1) / std.view(-1, 1, 1)
            conv.bias = nn.Parameter(norm.bias - norm.running_mean * norm.weight / std)
            return conv
        new_fullband_encoder = nn.ModuleList()
        for idx, module in enumerate(self.fullband_encoder):
            conv = merge_conv_bn(module[0], module[1], f"fullband_encoder.{idx}")
            new_module = nn.Sequential(
                conv,       # Conv-BN Merged
                module[2]   # Activation
            )
            new_fullband_encoder.append(new_module)
        self.fullband_encoder = new_fullband_encoder

        def merge_convt_bn(
            convt: nn.ConvTranspose1d,
            norm: nn.BatchNorm1d,
            error_message: str = ""
        ) -> nn.ConvTranspose1d:
            assert convt.bias is None, error_message
            std = norm.running_var.add(norm.eps).sqrt()
            convt.weight.data *= norm.weight.view(1, -1, 1) / std.view(1, -1, 1)
            convt.bias = nn.Parameter(norm.bias - norm.running_mean * norm.weight / std)
            return convt
        new_fullband_decoder = nn.ModuleList()
        for idx, module in enumerate(self.fullband_decoder[:-1]):
            conv = merge_convt_bn(module[1], module[2], f"fullband_decoder.{idx}")
            new_module = nn.Sequential(
                module[0],  # Conv
                conv,       # ConvT-BN Merged
                module[3]  # Activation
            )
            new_fullband_decoder.append(new_module)
        new_fullband_decoder.append(self.fullband_decoder[-1])
        self.fullband_decoder = new_fullband_decoder

    def model_forward(self, spec_noisy: Tensor, *args):
        # [B, F=257, T, 2]
        cache_in_list= [*args]
        if len(cache_in_list) == 0:
            num_dpe = len(self.dpe_blocks)
            num_groups = self.dpe_blocks[0].inter_rnn.groups
            cache_in_list = [None for _ in range(num_dpe *num_groups)]

        B, F0, T, _ = spec_noisy.shape
        x = spec_noisy                                      # [B, F, T, 2]
        x = x.permute(0, 2, 3, 1)                           # [B, T, 2, F]
        x = x.reshape(B*T, 2, F0)                           # [B*T, 2, F]
        mag = torch.linalg.norm(x, dim=1, keepdim=True)     # [B*T, 1, F]

        # Sub-band encoder
        x_sub1 = self.subband_encoder(mag)  # [B*T, C=32, F=32]

        # Full-band encoder
        enc_out = []
        for module in self.fullband_encoder:
            x = module(x)
            enc_out.append(x)
        x = self.fullband_encoder_post(x)   # [B*T, C=32, F=32]

        # Feature merge
        x = torch.cat([x, x_sub1], -1)  # [B*T, C=32, F=64]
        x = self.feature_merge(x)       # [B*T, C=16, F=32]

        # Dual Path RNN with Path Extension (DPE)
        _, C, F1 = x.shape
        x = x.reshape(B, T, C, F1).permute(1, 0, 3, 2)  # [T, B, F1, C]
        x = x.contiguous()
        cache_out_list = []
        for idx, dpe in enumerate(self.dpe_blocks):
            x, *cache_out = dpe(x, *cache_in_list[idx*8:(idx+1)*8])
            cache_out_list.extend(cache_out)
        # cache_out_list = cache_in_list

        x = x.permute(1, 0, 3, 2)   # [B, T, C, F1]
        x = x.reshape(B*T, C, F1)   # [B*T, C=16, F=32]

        # Feature split
        x = self.feature_split(x)   # [B*T, C=32, F=64]
        x_full = x[:, :, :32]       # [B*T, C=32, F=32]
        x_sub2 = x[:, :, 32:]       # [B*T, C=32, F=32]

        # Sub-band decoder
        x = torch.cat((x_sub1, x_sub2), dim=1)  # [B*T, C=64, F=32]
        x = self.subband_decoder(x)             # [B*T, F=257]
        mask_sub = x.reshape(B, T, F0).transpose(1, 2).unsqueeze(3)     # [B, F=257, T, 1]

        # Full-band decoder
        x = x_full
        for module in self.fullband_decoder:
            x = torch.cat([x, enc_out.pop(-1)], 1)  # [B*T, Cin*2, F]
            x = module(x)                           # [B*T, Cout, F]
        mask_full = x.reshape(B, T, 2, F0).permute(0, 3, 1, 2)  # [B, F=257, T, 2]
        out_full_r = spec_noisy[..., 0] * mask_full[..., 0] - spec_noisy[..., 1] * mask_full[..., 1]
        out_full_i = spec_noisy[..., 0] * mask_full[..., 1] + spec_noisy[..., 1] * mask_full[..., 0]
        out_full = torch.stack((out_full_r, out_full_i), dim=3)

        mask_full_mag = torch.linalg.norm(mask_full, dim=-1, keepdim=True)
        mask_mag = (mask_sub + mask_full_mag) * 0.5

        spec_out = out_full / mask_full_mag * mask_mag   # [B, F, T, 2]
        return spec_out, *cache_out_list

    def forward(self, spec_noisy: Tensor, *args):
        # spec_noisy: [B, F, T, 2] where F = n_fft//2+1
        # Input compression
        mag = torch.linalg.norm(
            spec_noisy,
            dim=-1,
            keepdim=True
        ).clamp(min=1.0e-5)
        spec_noisy = spec_noisy * mag.pow(self.input_compression - 1.0)

        # Model forward
        spec_hat, *cache_out_list = self.model_forward(spec_noisy, *args)

        # Uncompress
        mag_compressed = torch.linalg.norm(
            spec_hat,
            dim=3,
            keepdim=True
        )
        spec_hat = spec_hat * mag_compressed.pow(1.0 / self.input_compression - 1.0)
        return spec_hat, *cache_out_list


class Model(ONNXModel):
    def get_stft(
        self, n_fft: int, hop_size: int, win_size: int,
        window: str, normalized: bool, input_compression: float,
    ) -> nn.Module:
        return CompressedSTFT(
            n_fft=n_fft, hop_size=hop_size, win_size=win_size,
            win_type=window, normalized=normalized,
            compression=input_compression
        )

    def forward(self, noisy: Tensor):
        # noisy: [B, T_wav]
        spec_noisy = self.stft(noisy)   # [B, F, T, 2] where F = n_fft//2+1
        spec_out, *_ = self.model_forward(spec_noisy)
        x = torch.view_as_complex(spec_out.contiguous().float())  # [B, F, T]
        wav_out = self.stft.inverse(x)    # [B, T_wav]
        return wav_out, spec_out


if __name__ == "__main__":
    x = torch.randn(2, 16_000)
    model = FSPEN()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of total parameters: {total_params}")
    wav_out, spec_out = model(x)
    print(wav_out.shape, spec_out.shape)
