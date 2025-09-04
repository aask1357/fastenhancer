"""
From https://github.com/Xiaobin-Rong/gtcrn/
GTCRN: ShuffleNetV2 + SFE + TRA + 2 DPGRNN
Ultra tiny, 33.0 MMACs, 23.67 K params
"""
import typing as tp

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from .stft import CompressedSTFT


class ERB(nn.Module):
    def __init__(self, erb_subband_1, erb_subband_2, nfft=512, high_lim=8000, fs=16000):
        super().__init__()
        erb_filters = self.erb_filter_banks(erb_subband_1, erb_subband_2, nfft, high_lim, fs)
        nfreqs = nfft//2 + 1
        self.erb_subband_1 = erb_subband_1
        self.erb_fc = nn.Linear(nfreqs-erb_subband_1, erb_subband_2, bias=False)
        self.ierb_fc = nn.Linear(erb_subband_2, nfreqs-erb_subband_1, bias=False)
        delattr(self.erb_fc, "weight")
        delattr(self.ierb_fc, "weight")
        self.erb_fc.register_buffer("weight", erb_filters)
        self.ierb_fc.register_buffer("weight", erb_filters.T.contiguous().clone())

    def hz2erb(self, freq_hz):
        erb_f = 21.4*np.log10(0.00437*freq_hz + 1)
        return erb_f

    def erb2hz(self, erb_f):
        freq_hz = (10**(erb_f/21.4) - 1)/0.00437
        return freq_hz

    def erb_filter_banks(self, erb_subband_1, erb_subband_2, nfft=512, high_lim=8000, fs=16000):
        low_lim = erb_subband_1/nfft * fs
        erb_low = self.hz2erb(low_lim)
        erb_high = self.hz2erb(high_lim)
        erb_points = np.linspace(erb_low, erb_high, erb_subband_2)
        bins = np.round(self.erb2hz(erb_points)/fs*nfft).astype(np.int32)
        erb_filters = np.zeros([erb_subband_2, nfft // 2 + 1], dtype=np.float32)

        erb_filters[0, bins[0]:bins[1]] = (bins[1] - np.arange(bins[0], bins[1]) + 1e-12) \
                                                / (bins[1] - bins[0] + 1e-12)
        for i in range(erb_subband_2-2):
            erb_filters[i + 1, bins[i]:bins[i+1]] = (np.arange(bins[i], bins[i+1]) - bins[i] + 1e-12)\
                                                    / (bins[i+1] - bins[i] + 1e-12)
            erb_filters[i + 1, bins[i+1]:bins[i+2]] = (bins[i+2] - np.arange(bins[i+1], bins[i + 2])  + 1e-12) \
                                                    / (bins[i + 2] - bins[i+1] + 1e-12)

        erb_filters[-1, bins[-2]:bins[-1]+1] = 1- erb_filters[-2, bins[-2]:bins[-1]+1]
        
        erb_filters = erb_filters[:, erb_subband_1:]
        return torch.from_numpy(np.abs(erb_filters))
    
    def bm(self, x):
        """x: (B,C,T,F)"""
        x_low = x[..., :self.erb_subband_1]
        x_high = self.erb_fc(x[..., self.erb_subband_1:])
        return torch.cat([x_low, x_high], dim=-1)
    
    def bs(self, x_erb):
        """x: (B,C,T,F_erb)"""
        x_erb_low = x_erb[..., :self.erb_subband_1]
        x_erb_high = self.ierb_fc(x_erb[..., self.erb_subband_1:])
        return torch.cat([x_erb_low, x_erb_high], dim=-1)


class SFE(nn.Module):
    """Subband Feature Extraction"""
    def __init__(self, kernel_size=3, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size=(1,kernel_size), stride=(1, stride), padding=(0, (kernel_size-1)//2))
        
    def forward(self, x):
        """x: (B,C,T,F)"""
        xs = self.unfold(x).reshape(x.shape[0], x.shape[1]*self.kernel_size, x.shape[2], x.shape[3])
        return xs


class TRA(nn.Module):
    """Temporal Recurrent Attention"""
    def __init__(self, channels):
        super().__init__()
        self.att_gru = nn.GRU(channels, channels*2, 1, batch_first=True)
        self.att_fc = nn.Linear(channels*2, channels)
        self.att_act = nn.Sigmoid()

    def flatten_parameters(self):
        self.att_gru.flatten_parameters()

    def forward(self, x: Tensor, h: tp.Optional[Tensor]) -> tp.Tuple[Tensor, Tensor]:
        """x: [B, C, T, F]"""
        zt = torch.mean(x.pow(2), dim=-1)           # [B, C, T]
        at, h = self.att_gru(zt.transpose(1,2), h)  # [B, T, 2*C]
        at = self.att_fc(at).transpose(1,2)         # [B, C, T]
        at = self.att_act(at)
        At = at[..., None]  # [B, C, T, 1]
        return x * At, h


def merge_conv_bn(conv: nn.Module, norm: nn.Module, error_message: str = "") -> nn.Module:
    std = norm.running_var.add(norm.eps).sqrt()
    conv.weight.data *= norm.weight.view(-1, 1, 1, 1) / std.view(-1, 1, 1, 1)
    new_bias = norm.bias - norm.running_mean * norm.weight / std
    if conv.bias is None:
        conv.bias = nn.Parameter(new_bias)
    else:
        conv.bias.data += new_bias
    return conv


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, use_deconv=False, is_last=False):
        super().__init__()
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
        self.conv = conv_module(in_channels, out_channels, kernel_size, stride, padding,
                                groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.Tanh() if is_last else nn.PReLU()

    def remove_weight_reparametrizations(self):
        """ 2. Merge BatchNorm into Conv
        y = (conv(x) - mean) / std * gamma + beta \
          = conv(x) * (gamma / std) + (beta - mean * gamma / std)
        <=> y = conv'(x) where
          W'[c, :, :] = W[c, :, :] * (gamma / std)
          b' = (beta - mean * gamma / std)
        """
        self.conv = merge_conv_bn(self.conv, self.bn)
        self.bn = nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class GTConvBlock(nn.Module):
    """Group Temporal Convolution"""
    def __init__(self, in_channels, hidden_channels, kernel_size, stride,
                 padding, dilation, freq, use_deconv=False):
        super().__init__()
        self.use_deconv = use_deconv
        self.pad_size = (kernel_size[0]-1) * dilation[0]
        self.freq = freq
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d

        self.sfe = SFE(kernel_size=3, stride=1)

        self.point_conv1 = conv_module(in_channels//2*3, hidden_channels, 1)
        self.point_bn1 = nn.BatchNorm2d(hidden_channels)
        self.point_act = nn.PReLU()

        self.depth_conv = conv_module(hidden_channels, hidden_channels, kernel_size,
                                      stride=stride, padding=padding,
                                      dilation=dilation, groups=hidden_channels)
        self.depth_bn = nn.BatchNorm2d(hidden_channels)
        self.depth_act = nn.PReLU()

        self.point_conv2 = conv_module(hidden_channels, in_channels//2, 1)
        self.point_bn2 = nn.BatchNorm2d(in_channels//2)

        self.tra = TRA(in_channels//2)

    def flatten_parameters(self):
        self.tra.flatten_parameters()

    def remove_weight_reparametrizations(self):
        """ 2. Merge BatchNorm into Conv
        y = (conv(x) - mean) / std * gamma + beta \
          = conv(x) * (gamma / std) + (beta - mean * gamma / std)
        <=> y = conv'(x) where
          W'[c, :, :] = W[c, :, :] * (gamma / std)
          b' = (beta - mean * gamma / std)
        """
        self.point_conv1 = merge_conv_bn(self.point_conv1, self.point_bn1)
        self.point_bn1 = nn.Identity()
        self.depth_conv = merge_conv_bn(self.depth_conv, self.depth_bn)
        self.depth_bn = nn.Identity()
        self.point_conv2 = merge_conv_bn(self.point_conv2, self.point_bn2)
        self.point_bn2 = nn.Identity()

    def initialize_cache(self, x: Tensor) -> tp.List[Tensor]:
        cache_conv = x.new_zeros(1, self.depth_conv.in_channels, self.pad_size, self.freq)
        cache_tra = x.new_zeros(1, 1, self.tra.att_gru.hidden_size)
        return [cache_conv, cache_tra]

    def shuffle(self, x1, x2):
        """x1, x2: [B, C, T, F]"""
        B, C, T, F = x1.shape
        x = torch.stack([x1, x2], dim=2)    # [B, C, 2, T, F]
        x = x.view(B, 2*C, T, F)            # [B, 2*C, T, F]
        # x = torch.stack([x1, x2], dim=1)
        # x = x.transpose(1, 2).contiguous()  # (B,C,2,T,F)
        # x = x.view(x.shape[0], -1, x.shape[3], x.shape[4])  # (B,2C,T,F)
        return x

    def forward(
        self,
        x: Tensor,
        cache_conv: tp.Optional[Tensor],
        cache_tra: tp.Optional[Tensor]
    ) -> tp.Tuple[Tensor, Tensor, Tensor]:
        """x: [B, C, T, F]"""
        x1, x2 = torch.chunk(x, chunks=2, dim=1)

        x = self.sfe(x1)
        x = self.point_act(self.point_bn1(self.point_conv1(x)))
        if cache_conv is None:
            x = nn.functional.pad(x, [0, 0, self.pad_size, 0])
        else:
            x = torch.cat((cache_conv, x), dim=2)
        cache_conv = x[:, :, -self.pad_size:, :]
        x = self.depth_conv(x)
        x = self.depth_act(self.depth_bn(x))
        x = self.point_bn2(self.point_conv2(x))
        x, cache_tra = self.tra(x, cache_tra)
        x = self.shuffle(x, x2)
        
        return x, cache_conv, cache_tra


class GRNN(nn.Module):
    """Grouped RNN"""
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn1 = nn.GRU(input_size//2, hidden_size//2, num_layers, batch_first=batch_first, bidirectional=bidirectional)
        self.rnn2 = nn.GRU(input_size//2, hidden_size//2, num_layers, batch_first=batch_first, bidirectional=bidirectional)

    def flatten_parameters(self):
        self.rnn1.flatten_parameters()
        self.rnn2.flatten_parameters()

    def forward(
        self,
        x: Tensor,
        h1: tp.Optional[Tensor] = None,
        h2: tp.Optional[Tensor] = None
    ) -> tp.Tuple[Tensor, Tensor, Tensor]:
        """
        x: (B, seq_length, input_size)
        h1: (num_layers, B, hidden_size//2)
        h2: (num_layers, B, hidden_size//2)
        """
        x1, x2 = torch.chunk(x, chunks=2, dim=-1)
        y1, h1 = self.rnn1(x1, h1)
        y2, h2 = self.rnn2(x2, h2)
        y = torch.cat([y1, y2], dim=-1)
        return y, h1, h2
    
    
class DPGRNN(nn.Module):
    """Grouped Dual-path RNN"""
    def __init__(self, input_size, width, hidden_size, **kwargs):
        super(DPGRNN, self).__init__(**kwargs)
        self.input_size = input_size
        self.width = width
        self.hidden_size = hidden_size

        self.intra_rnn = GRNN(input_size=input_size, hidden_size=hidden_size//2, bidirectional=True)
        self.intra_fc = nn.Linear(hidden_size, hidden_size)
        self.intra_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)

        self.inter_rnn = GRNN(input_size=input_size, hidden_size=hidden_size, bidirectional=False)
        self.inter_fc = nn.Linear(hidden_size, hidden_size)
        self.inter_ln = nn.LayerNorm(((width, hidden_size)), eps=1e-8)

    def flatten_parameters(self):
        self.intra_rnn.flatten_parameters()
        self.inter_rnn.flatten_parameters()

    def initialize_cache(self, x: Tensor) -> tp.List[Tensor]:
        return [x.new_zeros(1, self.width, self.inter_rnn.rnn1.hidden_size) for _ in range(2)]

    def forward(
        self,
        x: Tensor,
        h1: tp.Optional[Tensor],
        h2: tp.Optional[Tensor]
    ) -> tp.Tuple[Tensor, Tensor, Tensor]:
        """x: (B, C, T, F)"""
        ## Intra RNN
        x = x.permute(0, 2, 3, 1)  # (B,T,F,C)
        intra_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])  # (B*T,F,C)
        intra_x = self.intra_rnn(intra_x)[0]  # (B*T,F,C)
        intra_x = self.intra_fc(intra_x)      # (B*T,F,C)
        intra_x = intra_x.reshape(x.shape[0], -1, self.width, self.hidden_size) # (B,T,F,C)
        intra_x = self.intra_ln(intra_x)
        intra_out = torch.add(x, intra_x)

        ## Inter RNN
        x = intra_out.permute(0,2,1,3)  # (B,F,T,C)
        inter_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3]) 
        inter_x, h1, h2 = self.inter_rnn(inter_x, h1, h2)  # (B*F,T,C)
        inter_x = self.inter_fc(inter_x)      # (B*F,T,C)
        inter_x = inter_x.reshape(x.shape[0], self.width, -1, self.hidden_size) # (B,F,T,C)
        inter_x = inter_x.permute(0,2,1,3)   # (B,T,F,C)
        inter_x = self.inter_ln(inter_x) 
        inter_out = torch.add(intra_out, inter_x)
        
        dual_out = inter_out.permute(0,3,1,2)  # (B,C,T,F)
        
        return dual_out, h1, h2
    

class Mask(nn.Module):
    """Complex Ratio Mask"""
    def __init__(self):
        super().__init__()

    def forward(self, mask, spec):
        s_real = spec[..., 0] * mask[..., 0] - spec[..., 1] * mask[..., 1]
        s_imag = spec[..., 1] * mask[..., 0] + spec[..., 0] * mask[..., 1]
        s = torch.stack([s_real, s_imag], dim=3)  # [B, F, T, 2]
        return s


class ONNXModel(nn.Module):
    def __init__(
        self,
        n_fft: int = 512,
        hop_size: int = 256,
        win_size: int = 512,
        input_compression: float = 0.3,
        n_dpgrnn: int = 2,
    ):
        super().__init__()
        self.input_compression = input_compression
        self.stft = CompressedSTFT(
            n_fft=n_fft,
            hop_size=hop_size,
            win_size=n_fft,
            win_type="hann",
            normalized=False,
            compression=input_compression,
        )
        self.erb = ERB(65, 64)
        self.sfe = SFE(3, 1)

        self.enc_pre = nn.ModuleList([
            ConvBlock(9, 16, (1,5), stride=(1,2), padding=(0,2),),
            ConvBlock(16, 16, (1,5), stride=(1,2), padding=(0,2), groups=2,)
        ])
        self.encoder = nn.ModuleList([
            GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(d,1), freq=33)
            for d in (1, 2, 5)
        ])

        self.dpgrnn = nn.ModuleList()
        for _ in range(n_dpgrnn):
            self.dpgrnn.append(DPGRNN(16, 33, 16))

        self.decoder = nn.ModuleList([
            GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(d,1), freq=33)
            for d in (5, 2, 1)
        ])
        self.dec_post = nn.ModuleList([
            ConvBlock(16, 16, (1,5), stride=(1,2), padding=(0,2), groups=2, use_deconv=True, is_last=False),
            ConvBlock(16, 2, (1,5), stride=(1,2), padding=(0,2), use_deconv=True, is_last=True)
        ])

        self.mask = Mask()
    
    @torch.no_grad()
    def remove_weight_reparameterizations(self):
        pass

    def flatten_parameters(self):
        for module in self.encoder:
            module.flatten_parameters()
        for module in self.dpgrnn:
            module.flatten_parameters()
        for module in self.decoder:
            module.flatten_parameters()

    def initialize_cache(self, x: Tensor) -> tp.List[Tensor]:
        cache_list = []
        for module in self.encoder:
            cache_list.extend(module.initialize_cache(x))
        for module in self.dpgrnn:
            cache_list.extend(module.initialize_cache(x))
        for module in self.decoder:
            cache_list.extend(module.initialize_cache(x))
        return cache_list

    def model_forward(self, spec_noisy: Tensor, *args) -> tp.Tuple[Tensor, tp.List[Tensor]]:
        # spec_noisy: [B, F, T, 2] where F = n_fft // 2 + 1
        cache_in_list = [*args]
        cache_out_list = []
        if len(cache_in_list) == 0:
            cache_in_list = [None for _ in range(len(self.encoder) * 4 + len(self.dpgrnn) * 2)]
        x = spec_noisy

        x = x.permute(0, 3, 2, 1)           # [B, 2, T, F]
        mag = torch.linalg.norm(x, dim=1, keepdim=True)     # [B, 1, T, F]
        x = torch.cat((mag, x), dim=1)      # [B, 3, T, F]

        # Band Merging & SFE
        x = self.erb.bm(x)  # [B, 3, T, 129]
        x = self.sfe(x)     # [B, 9, T, 129]

        # Encoder
        encoder_outs = []
        for module in self.enc_pre:
            x = module(x)
            encoder_outs.append(x)
        for module in self.encoder:
            x, *cache_out = module(x, cache_in_list.pop(0), cache_in_list.pop(0))
            encoder_outs.append(x)
            cache_out_list.extend([*cache_out])

        # Dual Path Grouped RNN
        for dpgrnn in self.dpgrnn:
            x, *cache_out = dpgrnn(x, cache_in_list.pop(0), cache_in_list.pop(0))
            cache_out_list.extend([*cache_out])

        # Decoder
        for module in self.decoder:
            x = x + encoder_outs.pop(-1)
            x, *cache_out = module(x, cache_in_list.pop(0), cache_in_list.pop(0))
            cache_out_list.extend([*cache_out])
        for module in self.dec_post:
            x = x + encoder_outs.pop(-1)
            x = module(x)

        # Band Split & mask
        mask = self.erb.bs(x)                   # [B, 2, T, F]
        mask = mask.permute(0, 3, 2, 1)         # [B, F, T, 2]
        spec_hat = self.mask(mask, spec_noisy)  # [B, F, T, 2]
        return spec_hat, cache_out_list

    def forward(self, spec_noisy: Tensor, *args) -> tp.Tuple[Tensor, tp.List[Tensor]]:
        # Compress Input
        mag = torch.linalg.norm(
            spec_noisy,
            dim=-1,
            keepdim=True
        ).clamp(min=1.0e-5)
        spec_noisy = spec_noisy * mag.pow(self.input_compression - 1.0)

        # Model forward
        spec_hat, cache_out_list = self.model_forward(spec_noisy, *args)

        # Uncompress Output
        mag_compressed = torch.linalg.norm(
            spec_hat,
            dim=-1,
            keepdim=True
        )
        spec_hat = spec_hat * mag_compressed.pow(1.0 / self.input_compression - 1.0)
        return spec_hat, *cache_out_list


class Model(ONNXModel):
    def forward(self, noisy: Tensor) -> tp.Tuple[Tensor, Tensor]:
        # farend/nearend_mic: [B, T_wav]
        spec_noisy = self.stft(noisy)                   # [B, F, T, 2]
        spec_hat, _ = self.model_forward(spec_noisy)    # [B, F, T, 2]
        spec_hat = torch.view_as_complex(spec_hat)      # [B, F, T]
        wav_hat = self.stft.inverse(spec_hat)           # [B, T_wav]
        return wav_hat, torch.view_as_real(spec_hat)


def test():
    x = torch.randn(3, 16_000)
    from utils import get_hparams
    hparams = get_hparams("configs/se/gtcrn.yaml")
    model = GTCRN(**hparams["model_kwargs"])
    model.flatten_parameters()
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
