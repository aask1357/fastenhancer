import typing as tp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from torch import Tensor

from functional import CompressedSTFT, ONNXSTFT


class CustomLayerNorm(nn.Module):
    def __init__(self, input_dims, stat_dims=(1,), num_dims=4, eps=1e-5):
        super().__init__()
        assert isinstance(input_dims, tuple) and isinstance(stat_dims, tuple)
        assert len(input_dims) == len(stat_dims)
        param_size = [1] * num_dims
        for input_dim, stat_dim in zip(input_dims, stat_dims):
            param_size[stat_dim] = input_dim
        self.gamma = Parameter(torch.ones(*param_size).to(torch.float32))
        self.beta = Parameter(torch.zeros(*param_size).to(torch.float32))
        self.eps = eps
        self.stat_dims = stat_dims
        self.num_dims = num_dims

    def forward(self, x):
        assert x.ndim == self.num_dims, print(
            "Expect x to have {} dimensions, but got {}".format(self.num_dims, x.ndim))

        mu_ = x.mean(dim=self.stat_dims, keepdim=True)  # [B,1,T,F]
        std_ = torch.sqrt(
            x.var(dim=self.stat_dims, unbiased=False, keepdim=True) + self.eps
        )  # [B,1,T,F]
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat


class RNN(nn.Module):
    def __init__(
            self,
            emb_dim,
            hidden_dim,
            dropout_p=0.1,
            bidirectional=False,
    ):
        super().__init__()
        self.rnn = nn.GRU(emb_dim, hidden_dim, 1, batch_first=True, bidirectional=bidirectional)
        if bidirectional:
            self.dense = nn.Linear(hidden_dim * 2, emb_dim)
        else:
            self.dense = nn.Linear(hidden_dim, emb_dim)
    
    def forward(self, x: Tensor, h: tp.Optional[Tensor]) -> tp.Tuple[Tensor, Tensor]:
        # x:(b,t,d)
        x, h = self.rnn(x, h)
        x = self.dense(x)
        return x, h


class DualPathRNN(nn.Module):
    def __init__(
            self,
            emb_dim,
            hidden_dim,
            n_freqs=32,
            dropout_p=0.1,
    ):
        super().__init__()
        self.n_freqs = n_freqs
        self.intra_norm = nn.LayerNorm((n_freqs, emb_dim))
        self.intra_rnn_attn = RNN(emb_dim, hidden_dim // 2, dropout_p, bidirectional=True)

        self.inter_norm = nn.LayerNorm((n_freqs, emb_dim))
        self.inter_rnn_attn = RNN(emb_dim, hidden_dim, dropout_p, bidirectional=False)

    def initialize_cache(self, x: Tensor) -> Tensor:
        return x.new_zeros(1, self.n_freqs, self.inter_rnn_attn.rnn.hidden_size)

    def forward(self, x: Tensor, h: tp.Optional[Tensor]) -> tp.Tuple[Tensor, Tensor]:
        # x:(b,d,t,f)
        B, D, T, F = x.size()
        x = x.permute(0, 2, 3, 1)  # (b,t,f,d)

        x_res = x
        x = self.intra_norm(x)
        x = x.reshape(B * T, F, D)  # (b*t,f,d)
        x, _ = self.intra_rnn_attn(x, None)
        x = x.reshape(B, T, F, D)
        x = x + x_res

        x_res = x
        x = self.inter_norm(x)
        x = x.permute(0, 2, 1, 3)  # (b,f,t,d)
        x = x.reshape(B * F, T, D)
        x, h = self.inter_rnn_attn(x, h)
        x = x.reshape(B, F, T, D).permute(0, 2, 1, 3) # (b,t,f,d)
        x = x + x_res

        x = x.permute(0, 3, 1, 2)
        return x, h


class ConvolutionalGLU(nn.Module):
    def __init__(self, emb_dim, n_freqs=32, expansion_factor=2, dropout_p=0.1):
        super().__init__()
        hidden_dim = int(emb_dim * expansion_factor)
        self.n_freqs = n_freqs
        self.hidden_dim = hidden_dim
        self.norm = CustomLayerNorm((emb_dim, n_freqs), stat_dims=(1, 3))
        self.fc1 = nn.Conv2d(emb_dim, hidden_dim * 2, 1)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, 3, groups=hidden_dim, padding=(0, 1))
        self.act = nn.Mish()
        self.fc2 = nn.Conv2d(hidden_dim, emb_dim, 1)
        self.dropout = nn.Dropout(dropout_p)

    def initialize_cache(self, x: Tensor) -> Tensor:
        return x.new_zeros(1, self.hidden_dim, 2, self.n_freqs)

    def forward(self, x: Tensor, cache_in: tp.Optional[Tensor] = None) -> tp.Tuple[Tensor, Tensor]:
        # x:(b,d,t,f)
        res = x
        x = self.norm(x)
        x, v = self.fc1(x).chunk(2, dim=1)
        
        if cache_in is not None:
            x = torch.cat([cache_in, x], dim=2)
        else:
            x = F.pad(x, (0, 0, 2, 0), mode='constant', value=0.0)
        cache_out = x[:, :, -2:, :]  # (b,d,2,f)

        x = self.act(self.dwconv(x)) * v
        x = self.dropout(x)
        x = self.fc2(x)
        x = x + res
        return x, cache_out


class DPR(nn.Module):
    def __init__(
            self,
            emb_dim=16,
            hidden_dim=24,
            n_freqs=32,
            dropout_p=0.1,
    ):
        super().__init__()
        self.dp_rnn_attn = DualPathRNN(emb_dim, hidden_dim, n_freqs, dropout_p)
        self.conv_glu = ConvolutionalGLU(emb_dim, n_freqs=n_freqs, expansion_factor=2, dropout_p=dropout_p)

    def forward(
        self,
        x: Tensor,
        cache_rnn: tp.Optional[Tensor],
        cache_conv: tp.Optional[Tensor],
    ) -> tp.Tuple[Tensor, Tensor, Tensor]:
        x, cache_rnn = self.dp_rnn_attn(x, cache_rnn)
        x, cache_conv = self.conv_glu(x, cache_conv)
        return x, cache_rnn, cache_conv


class LearnableSigmoid2d(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1, 1))
        self.slope.requires_grad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


class DSConv(nn.Module):
    def __init__(self, in_channels, out_channels, n_freqs):
        super().__init__()
        self.n_freqs = n_freqs
        self.low_freqs = n_freqs // 4
        self.low_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=(2, 3), padding=(0, 1),
        )
        self.high_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=(2, 5), stride=(1, 3), padding=(0, 1),
        )
        self.norm = CustomLayerNorm((1, n_freqs // 2), stat_dims=(1, 3))
        self.act = nn.PReLU(out_channels)

    def initialize_cache(self, x: Tensor) -> Tensor:
        return x.new_zeros(1, self.low_conv.in_channels, 1, self.n_freqs)
    
    def forward(self, x: Tensor, cache_in: tp.Optional[Tensor]) -> tp.Tuple[Tensor, Tensor]:
        # (b,d,t,f)
        if cache_in is not None:
            x = torch.cat([cache_in, x], dim=2)
        else:
            x = F.pad(x, (0, 0, 1, 0), mode='constant', value=0.0)
        cache_out = x[:, :, -1:, :]
        
        x_low = x[..., :self.low_freqs]
        x_high = x[..., self.low_freqs:]
        
        x_low = self.low_conv(x_low)
        x_high = self.high_conv(x_high)

        x = torch.cat([x_low, x_high], dim=-1)
        x = self.norm(x)
        x = self.act(x)
        return x, cache_out


class USConv(nn.Module):
    def __init__(self, in_channels, out_channels, n_freqs):
        super().__init__()
        self.n_freqs = n_freqs
        self.low_freqs = n_freqs // 2
        self.low_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), padding=(0, 1))
        self.high_conv = SPConvTranspose2d(in_channels, out_channels, kernel_size=(1, 3), r=3)

    def forward(self, x: Tensor) -> Tensor:
        # (b,d,t,f)
        x_low = x[..., :self.low_freqs]
        x_high = x[..., self.low_freqs:]

        x_low = self.low_conv(x_low)
        x_high = self.high_conv(x_high)
        x = torch.cat([x_low, x_high], dim=-1)
        return x


class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        super(SPConvTranspose2d, self).__init__()
        # self.pad_time = kernel_size[0] - 1
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1),
            padding=(0, kernel_size[1]//2)
        )
        self.r = r

    def forward(self, x):
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out


class Encoder(nn.Module):
    def __init__(self, in_channels, num_channels=16):
        super(Encoder, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, num_channels//4, (1, 1), (1, 1)),
            CustomLayerNorm((1, 257), stat_dims=(1, 3)),
            nn.PReLU(num_channels // 4),
        )
        
        self.conv_2 = DSConv(num_channels//4, num_channels//2, n_freqs=257)
        self.conv_3 = DSConv(num_channels//2, num_channels//4*3, n_freqs=128)
        self.conv_4 = DSConv(num_channels//4*3, num_channels, n_freqs=64)

    def initialize_cache(self, x: Tensor) -> tp.List[Tensor]:
        # Initialize caches for each DSConv layer
        cache2 = self.conv_2.initialize_cache(x)
        cache3 = self.conv_3.initialize_cache(x)
        cache4 = self.conv_4.initialize_cache(x)
        return [cache2, cache3, cache4]

    def forward(self, x, cache2, cache3, cache4):
        x1 = self.conv_1(x)
        x2, cache2 = self.conv_2(x1, cache2)    # 128
        x3, cache3 = self.conv_3(x2, cache3)    # 64
        x4, cache4 = self.conv_4(x3, cache4)    # 32
        return x2, x3, x4, cache2, cache3, cache4


class MaskDecoder(nn.Module):
    def __init__(self, num_features, num_channels=64, out_channel=2, beta=1):
        super(MaskDecoder, self).__init__()
        self.up1 = USConv(num_channels * 2, num_channels // 4 * 3, n_freqs=32)
        self.up2 = USConv(num_channels // 4 * 3 * 2, num_channels // 2, n_freqs=64)  # 128
        self.up3 = USConv(num_channels // 2 * 2, num_channels // 4, n_freqs=128)  # 256
        self.mask_conv = nn.Sequential(
            nn.Conv2d(num_channels // 4, out_channel, (2, 2), padding=(0, 1)), # 257
            CustomLayerNorm((1, 257), stat_dims=(1, 3)),
            nn.PReLU(out_channel),
            nn.Conv2d(out_channel, out_channel, (1, 1)),
        )
        self.lsigmoid = LearnableSigmoid2d(num_features, beta=beta)

    def initialize_cache(self, x: Tensor) -> tp.List[Tensor]:
        cache = x.new_zeros(1, self.up3.low_conv.out_channels, 1, 256)
        return [cache]

    def forward(self, x, enc1, enc2, enc3, cache):
        x = self.up1(torch.cat([x, enc3], dim=1))  # 64
        x = self.up2(torch.cat([x, enc2], dim=1))  # 128
        x = self.up3(torch.cat([x, enc1], dim=1))  # 256

        if cache is not None:
            x = torch.cat([cache, x], dim=2)
        else:
            x = F.pad(x, (0, 0, 1, 0), mode='constant', value=0.0)
        cache = x[:, :, -1:, :]  # (B, out_channel, 1, F)

        x = self.mask_conv(x)  # (B,out_channel,T,F)
        x = x.permute(0, 3, 2, 1)  # (B,F,T,out_channel)
        x = self.lsigmoid(x).permute(0, 3, 2, 1)
        return x, cache


class ONNXModel(nn.Module):
    def __init__(
        self,
        num_channels: int = 16,
        n_blocks: int = 2,
        n_fft: int = 512,
        hop_size: int = 256,
        win_size: int = 512,
        window: tp.Optional[str] = "hann",
        input_compression: float = 0.3,
        normalized: bool = False,
    ):
        super().__init__()
        self.input_compression = input_compression
        self.stft = self.get_stft(n_fft, hop_size, win_size, window, normalized)
        self.n_freqs = n_fft // 2 + 1

        self.encoder = Encoder(in_channels=3, num_channels=num_channels)

        self.blocks = nn.Sequential(
            *[DPR(
                emb_dim=num_channels,
                hidden_dim=num_channels // 2 * 3,
                n_freqs=self.n_freqs // (2 ** 3),
                dropout_p=0.1,
            ) for _ in range(n_blocks)]
        )

        self.decoder = MaskDecoder(self.n_freqs, num_channels=num_channels, out_channel=2, beta=1)

    def get_stft(
        self, n_fft: int, hop_size: int, win_size: int,
        window: str, normalized: bool
    ) -> nn.Module:
        return ONNXSTFT(
            n_fft=n_fft, hop_size=hop_size, win_size=win_size,
            win_type=window, normalized=normalized
        )

    def flatten_parameters(self):
        for block in self.blocks:
            block.dp_rnn_attn.intra_rnn_attn.rnn.flatten_parameters()
            block.dp_rnn_attn.inter_rnn_attn.rnn.flatten_parameters()

    @staticmethod
    def cal_gd(x):
        # x: (B,T,F), return (-pi, pi]
        b, t, f = x.size()
        x_padded = F.pad(x[:, :, :-1], (1, 0), mode='constant', value=0.0)  # pad at the front
        x_gd = x_padded - x  # (-2pi, 2pi]
        return torch.atan2(x_gd.sin(), x_gd.cos())

    @staticmethod
    def cal_if(x):
        # x:(B,T,F), return (-pi, pi]
        b, t, f = x.size()
        x_padded = F.pad(x[:, :-1, :], (1, 0), mode='constant', value=0.0)  # pad at the front
        x_if = x_padded - x  # (-2pi, 2pi]
        return torch.atan2(x_if.sin(), x_if.cos())
    
    def cal_ifd(self, x: Tensor, cache_in: Tensor) -> Tensor:
        # x:(B,T,F), return (-pi, pi]
        b, t, f = x.size()
        x_padded = torch.cat([cache_in, x[:, :-1, :]], dim=1)  # pad at the front
        x_if = x_padded - x  # (-2pi, 2pi]
        x_ifd = x_if - 2 * torch.pi * (self.stft.hop_size / self.stft.n_fft) * torch.arange(f, device=x.device, dtype=torch.float32)[None, None, :]
        return torch.atan2(x_ifd.sin(), x_ifd.cos())
    
    def initialize_cache(self, x: Tensor) -> tp.List[Tensor]:
        # ifd
        cache_list = [torch.zeros(1, 1, self.n_freqs, device=x.device)]
        
        # encoder
        cache_list.extend(self.encoder.initialize_cache(x))

        # blocks
        for block in self.blocks:
            # DPR
            cache_list.append(block.dp_rnn_attn.initialize_cache(x))
            cache_list.append(block.conv_glu.initialize_cache(x))

        # decoder
        cache_list.extend(self.decoder.initialize_cache(x))

        return cache_list

    def model_forward(self, x: Tensor, *args) -> tp.Tuple[Tensor, ...]:
        # x: [B, 3, T, F]
        cache_in_list = [*args]
        if len(cache_in_list) == 0:
            cache_in_list = [None for _ in range(3+len(self.blocks)*2+1)]
        cache_out_list = []

        # Encoder
        enc1, enc2, enc3, cache0, cache1, cache2 = self.encoder(
            x,
            cache_in_list.pop(0),
            cache_in_list.pop(0),
            cache_in_list.pop(0),
        )
        cache_out_list.extend([cache0, cache1, cache2])

        # Bottleneck
        x = enc3
        for block in self.blocks:
            x, cache_out1, cache_out2 = block(x, cache_in_list.pop(0), cache_in_list.pop(0))
            cache_out_list.append(cache_out1)
            cache_out_list.append(cache_out2)

        # Decoder
        x, cache = self.decoder(
            x,
            enc1,
            enc2,
            enc3,
            cache_in_list.pop(0),
        )       # [B, 2, T, F]
        cache_out_list.append(cache)
        mask = x.permute(0, 3, 2, 1).contiguous()   # [B, F, T, 2]
        
        return mask, *cache_out_list

    def forward(self, spec_noisy: Tensor, *args) -> tp.Tuple[Tensor, Tensor]:
        # spec_noisy: [B, F, T, 2] where F = n_fft // 2 + 1
        cache_in_list = [*args]
        cache_out_list = []

        # input processing
        mag = torch.linalg.norm(
            spec_noisy,
            dim=-1,
            keepdim=True
        ).clamp(min=1.0e-5)
        spec_noisy = spec_noisy * mag.pow(self.input_compression - 1.0)
        x = spec_noisy.transpose(1, 2)          # [B, T, F, 2]
        mag = torch.linalg.norm(x, dim=-1)      # [B, T, F]
        pha = torch.atan2(x[..., 1], x[..., 0]) # [B, T, F]
        gd = self.cal_gd(pha)
        ifd = self.cal_ifd(pha, cache_in_list.pop(0))
        cache_out_list.append(pha[:, -1:, :])
        x = torch.stack([mag, gd / torch.pi, ifd / torch.pi], dim=1)    # [B, 3, T, F]

        # model
        mask, *cache_out = self.model_forward(x, *cache_in_list)    # [B, F, T, 2]
        cache_out_list.extend(cache_out)

        # output processing
        spec_hat = torch.stack(
            [
                spec_noisy[..., 0] * mask[..., 0] - spec_noisy[..., 1] * mask[..., 1],
                spec_noisy[..., 0] * mask[..., 1] + spec_noisy[..., 1] * mask[..., 0],
            ],
            dim=3
        )
        return spec_hat, *cache_out_list

    def remove_weight_reparameterizations(self):
        pass


class Model(ONNXModel):
    def get_stft(
        self, n_fft: int, hop_size: int, win_size: int,
        window: str, normalized: bool
    ) -> nn.Module:
        return CompressedSTFT(
            n_fft=n_fft, hop_size=hop_size, win_size=win_size,
            win_type=window, normalized=normalized,
            compression=self.input_compression
        )

    @staticmethod
    def cal_gd(x):
        # x: (B,T,F), return (-pi, pi]
        b, t, f = x.size()
        x_gd = torch.diff(x, dim=2, prepend=torch.zeros(b, t, 1, device=x.device))  # (-2pi, 2pi]
        return torch.atan2(x_gd.sin(), x_gd.cos())

    @staticmethod
    def cal_if(x):
        # x:(B,T,F), return (-pi, pi]
        b, t, f = x.size()
        x_if = torch.diff(x, dim=1, prepend=torch.zeros(b, 1, f, device=x.device))  # (-2pi, 2pi]
        return torch.atan2(x_if.sin(), x_if.cos())
    
    def cal_ifd(self, x):
        # x:(B,T,F), return (-pi, pi]
        b, t, f = x.size()
        x_if = torch.diff(x, dim=1, prepend=x.new_zeros(b, 1, f))  # (-2pi, 2pi]
        x_ifd = x_if - 2 * torch.pi * (self.stft.hop_size / self.stft.n_fft) * torch.arange(f, device=x.device)[None, None, :]
        return torch.atan2(x_ifd.sin(), x_ifd.cos())

    def forward(self, x: Tensor) -> tp.Tuple[Tensor, Tensor]:
        # x: [B, T]
        x = self.stft(x)                # [B, F, T, 2]
        spec_noisy = torch.view_as_complex(x)   # [B, F, T]

        # input processing
        x = spec_noisy.transpose(1, 2)  # [B, T, F]
        mag = x.abs()                   # [B, T, F]
        pha = x.angle()                 # [B, T, F]
        gd = self.cal_gd(pha)
        ifd = self.cal_ifd(pha)
        x = torch.stack([mag, gd / torch.pi, ifd / torch.pi], dim=1)    # [B, 3, T, F]

        # model
        mask, *_ = self.model_forward(x)    # [B, F, T, 2]

        # output processing
        spec_hat = torch.view_as_complex(mask) * spec_noisy
        wav_hat = self.stft.inverse(spec_hat)
        return wav_hat, torch.view_as_real(spec_hat)


def test():
    x = torch.randn(3, 16_000)
    from utils import get_hparams
    hparams = get_hparams("configs/se/lisennet.yaml")
    # hparams["model_kwargs"]["weight_norm"] = False
    model = LiSenNet(**hparams["model_kwargs"])
    # model.flatten_parameters()
    total_params = sum(p.numel() for n, p in model.named_parameters())
    print(f"Number of total parameters: {total_params}")
    wav_out, spec_out = model(x)
    (wav_out.sum() + spec_out.sum()).backward()
    print(spec_out.shape)
    # for n, p in model.named_parameters():
    #     print(n, p.shape)


if __name__ == "__main__":
    test()
