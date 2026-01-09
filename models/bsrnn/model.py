"""Code from https://github.com/sungwon23/BSRNN"""
import re
import typing as tp
from itertools import accumulate

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor

from functional import ONNXSTFT, CompressedSTFT


def fuse_bn_conv1d(conv: nn.Module, norm: nn.Module, error: str):
    """ Fuse "BN -> Conv1d" into a Conv1d
    conv:
        weight: [Co, Ci, k=1]
        bias: [Co]
    norm:
        mean, var, weight, bias: [Ci]
    => conv.weight @ {(x - mean) / std * norm.weight + norm.bias} + conv.bias
        = (conv.weight * w') @ x + conv.bias + b'
        where w' = norm.weight / std, b' = conv.weight @ (-mean/std*norm.weight + norm.bias)
    """
    assert conv.weight.size(2) == 1, f"{error}: {conv.weight.shape}"
    try:
        std = norm.running_var.add(norm.eps).sqrt()
        w = 1 / std
        b = -norm.running_mean / std
        if norm.affine:
            w = norm.weight * w
            b = b * norm.weight + norm.bias
        b = (conv.weight.data * b.view(1, -1, 1)).sum(dim=(1, 2))
        conv.weight.data *= w.view(1, -1, 1)
        if conv.bias is None:
            conv.bias = nn.Parameter(b)
        else:
            conv.bias.data.add_(b)
        return conv
    except Exception as e:
        print(error)
        raise RuntimeError(e)


def fuse_bn_rnn(rnn: nn.Module, norm: nn.Module, error: str) -> nn.Module:
    """ Fuse "BN->RNN" into an RNN
    rnn:
        weight_ih_l0(_reverse): [N*H, Cin] where N=4 for LSTM, N=3 for GRU
        bias_ih_l0(_reverse): [N*H]
    norm:
        mean, var, weight, bias: [Cin]
    """
    try:
        if hasattr(rnn, "weight_ih_l0"): # LSTM or GRU
            std = norm.running_var.add(norm.eps).sqrt()
            w = 1 / std 
            b = -norm.running_mean / std
            if norm.affine:
                w = w * norm.weight
                b = b * norm.weight + norm.bias
            b_new = (rnn.weight_ih_l0.data @ b.view(-1, 1)).squeeze(1)
            rnn.weight_ih_l0.data.mul_(w)
            rnn.bias_ih_l0.data.add_(b_new)
            if rnn.bidirectional:
                b_new = (rnn.weight_ih_l0_reverse.data @ b.view(-1, 1)).squeeze(1)
                rnn.weight_ih_l0_reverse.data.mul_(w)
                rnn.bias_ih_l0_reverse.data.add_(b_new)
            return rnn
        else: # LSTMCell or GRUCell
            std = norm.running_var.add(norm.eps).sqrt()
            w = 1 / std 
            b = -norm.running_mean / std
            if norm.affine:
                w = w * norm.weight
                b = b * norm.weight + norm.bias
            b_new = (rnn.weight_ih.data @ b.view(-1, 1)).squeeze(1)
            rnn.weight_ih.data.mul_(w)
            rnn.bias_ih.data.add_(b_new)
            return rnn
    except Exception as e:
        print(error)
        raise RuntimeError(e)


class ChannelsLastBatchNorm(nn.BatchNorm1d):
    def forward(self, x: Tensor) -> Tensor:
        """input/output: [..., C]"""
        orig_shape = x.shape
        C = orig_shape[-1]
        x = x.view(-1, C, 1)
        return super().forward(x).view(*orig_shape)


class ChannelsLastSyncBatchNorm(nn.SyncBatchNorm):
    def forward(self, x: Tensor) -> Tensor:
        """input/output: [..., C]"""
        orig_shape = x.shape
        C = orig_shape[-1]
        x = x.view(-1, C, 1)
        return super().forward(x).view(*orig_shape)


class BandSplit(nn.Module):
    def __init__(self, n_fft: int, channels: int, bias: bool, affine: bool):
        super().__init__()
        if n_fft == 512:
            self.subbands = [
                2,    3,    3,    3,    3,   3,   3,    3,    3,    3,   3,
                8,    8,    8,    8,    8,   8,   8,    8,    8,    8,   8,   8,
                16,   16,   16,   16,   16,  16,  16,   17
            ]   # sum(self.subbands) == 257
        else:
            raise RuntimeError(f"Only n_fft=512 is supported, but given {n_fft}")

        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            BatchNorm = nn.SyncBatchNorm
        else:
            BatchNorm = nn.BatchNorm1d

        self.norm = nn.ModuleList()
        self.fc = nn.ModuleList()
        for subband in self.subbands:
            self.norm.append(BatchNorm(subband * 2, affine=affine))
            self.fc.append(nn.Conv1d(subband * 2, channels, 1, bias=bias))

    def remove_weight_reparameterizations(self):
        new_fc = nn.ModuleList()
        new_norm = nn.ModuleList()
        for i in range(len(self.subbands)):
            fc = fuse_bn_conv1d(self.fc[i], self.norm[i], f"bansplit.{i}")
            new_fc.append(fc)
            new_norm.append(nn.Identity())
        self.fc = new_fc
        self.norm = new_norm

    def forward(self, spec_noisy: Tensor) -> Tensor:
        """
        Args:
            x: [B, F, 2, T]
        Out:
            [B, F', C, T] where F' = len(self.band)"""
        band_start = 0
        B, _, _, T = spec_noisy.shape
        out = []
        for norm, fc, subband in zip(self.norm, self.fc, self.subbands):
            band_end = band_start + subband
            x = spec_noisy[:, band_start:band_end, :, :]
            x = x.reshape(B, -1, T) # [B, Subband * 2, T]
            x = norm(x)
            x = fc(x)               # [B, C, T]
            out.append(x)
            band_start = band_end
        return torch.stack(out, dim=1)  # [B, F', C, T]


class MaskDecoder(nn.Module):
    def __init__(
        self,
        freq_dim: int,
        subbands: tp.List[int],
        channels: int,
        affine: bool
    ):
        super().__init__()
        assert freq_dim == sum(subbands), (freq_dim, subbands)
        self.subbands = subbands
        self.freq_dim = freq_dim

        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            BatchNorm = nn.SyncBatchNorm
        else:
            BatchNorm = nn.BatchNorm1d

        self.mlp_mask = nn.ModuleList()
        self.mlp_residual = nn.ModuleList()
        for subband in self.subbands:
            self.mlp_mask.append(
                nn.Sequential(
                    BatchNorm(channels, affine=affine),
                    nn.Conv1d(channels, 4 * channels, 1),
                    nn.Tanh(),
                    nn.Conv1d(4 * channels, subband * 4, 1),
                    nn.GLU(dim=1),
                )
            )
            self.mlp_residual.append(
                nn.Sequential(
                    BatchNorm(channels, affine=affine),
                    nn.Conv1d(channels, 4 * channels, 1),
                    nn.Tanh(),
                    nn.Conv1d(4 * channels, subband * 4, 1),
                    nn.GLU(dim=1),
                )
            )

    def remove_weight_reparameterizations(self):
        new_mask = nn.ModuleList()
        new_residual = nn.ModuleList()
        for i in range(len(self.subbands)):
            fc = fuse_bn_conv1d(
                self.mlp_mask[i][1],
                self.mlp_mask[i][0],
                f"mlp_mask.{i}"
            )
            new_mask.append(nn.Sequential(
                fc,
                self.mlp_mask[i][2],
                self.mlp_mask[i][3],
                self.mlp_mask[i][4],
            ))
            fc = fuse_bn_conv1d(
                self.mlp_residual[i][1],
                self.mlp_residual[i][0],
                f"mlp_residual.{i}"
            )
            new_residual.append(nn.Sequential(
                fc,
                self.mlp_residual[i][2],
                self.mlp_residual[i][3],
                self.mlp_residual[i][4],
            ))
        self.mlp_mask = new_mask
        self.mlp_residual = new_residual

    def forward(self, x: Tensor) -> tp.Tuple[Tensor, Tensor]:
        """
        Args:
            x (Tensor): input tensor of shape [B, F', C, T]
        Returns:
            m (Tensor): output mask of shape [B, F, T, 2]
            r (Tensor): output residual of shape [B, F, T, 2]
        """
        B, _, _, T = x.shape
        mask, residual = [], []
        for i in range(len(self.subbands)):
            x_band = x[:, i, :, :]
            out = self.mlp_mask[i](x_band)  # [B, Subband*2, T]
            out = out.view(B, -1, 2, T)     # [B, Subband, 2, T]
            mask.append(out)

            out = self.mlp_residual[i](x_band)  
            out = out.view(B, -1, 2, T)
            residual.append(out)
        mask = torch.cat(mask, dim=1)   # [B, F, 2, T]
        residual = torch.cat(residual, dim=1)
        return mask.transpose(2, 3), residual.transpose(2, 3)


class ONNXLSTM(nn.LSTMCell):
    """In torch2.7, nn.LSTM(bidirectional=False) fails to be exported to ONNX.
    Therefore, we implement LSTM with nn.LSTMCell.
    """
    def forward(self, x, hidden):
        """x: [T=1, F, C]
        output: [F, C]"""
        h, c = super().forward(x.squeeze(0), hidden)
        return h, (h, c)


class ONNXModel(nn.Module):
    def __init__(
        self,
        num_channels: int = 16,
        num_layers: int = 6,
        bias: bool = True,
        affine: bool = True,
        n_fft: int = 512,
        hop_size: int = 256,
        win_size: int = 512,
        window: str = "hann",
        input_compression: float = 0.3,
        onnx: bool = True,
    ):
        """Band-Split RNN (BSRNN).

        References:
            J. Yu, H. Chen, Y. Luo, R. Gu, and C. Weng, “High fidelity speech
            enhancement with band-split RNN,” in Proc. ISCA Interspeech, 2023.
            https://isca-speech.org/archive/interspeech_2023/yu23b_interspeech.html
        """
        super().__init__()
        self.onnx = onnx
        self.num_layers = num_layers
        self.input_compression = input_compression

        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            BatchNorm = ChannelsLastSyncBatchNorm
        else:
            BatchNorm = ChannelsLastBatchNorm

        self.stft = self.get_stft(n_fft, hop_size, win_size, window)
        self.band_split = BandSplit(n_fft, num_channels, bias, affine)

        self.norm_time = nn.ModuleList()
        self.rnn_time = nn.ModuleList()
        self.fc_time = nn.ModuleList()
        self.norm_freq = nn.ModuleList()
        self.rnn_freq = nn.ModuleList()
        self.fc_freq = nn.ModuleList()
        hdim = 2 * num_channels
        TimeLSTM = ONNXLSTM if onnx else nn.LSTM
        for i in range(self.num_layers):
            self.norm_time.append(BatchNorm(num_channels, affine=affine))
            self.rnn_time.append(
                TimeLSTM(
                    num_channels,
                    hdim,
                    # batch_first=False,
                    # bidirectional=False,
                )
            )
            self.fc_time.append(nn.Linear(hdim, num_channels, bias=bias))
            self.norm_freq.append(BatchNorm(num_channels, affine=affine))
            self.rnn_freq.append(
                nn.LSTM(num_channels, hdim, batch_first=True, bidirectional=True)
            )
            self.fc_freq.append(nn.Linear(2 * hdim, num_channels, bias=bias))

        self.mask_decoder = MaskDecoder(
            n_fft//2+1, self.band_split.subbands, num_channels, bias
        )

    def get_stft(
        self, n_fft: int, hop_size: int, win_size: int,
        window: str, normalized: bool = False
    ) -> nn.Module:
        return ONNXSTFT(
            n_fft=n_fft, hop_size=hop_size, win_size=win_size,
            win_type=window, normalized=normalized
        )

    @torch.no_grad()
    def remove_weight_reparameterizations(self):
        self.band_split.remove_weight_reparameterizations()
        self.mask_decoder.remove_weight_reparameterizations()

        norm_time = nn.ModuleList()
        rnn_time = nn.ModuleList()
        norm_freq = nn.ModuleList()
        rnn_freq = nn.ModuleList()
        for i in range(self.num_layers):
            rnn = fuse_bn_rnn(self.rnn_time[i], self.norm_time[i], f"rnn_time.{i}")
            rnn_time.append(rnn)
            norm_time.append(nn.Identity())
            rnn = fuse_bn_rnn(self.rnn_freq[i], self.norm_freq[i], f"rnn_freq.{i}")
            rnn_freq.append(rnn)
            norm_freq.append(nn.Identity())
        self.norm_time = norm_time
        self.rnn_time = rnn_time
        self.norm_freq = norm_freq
        self.rnn_freq = rnn_freq

    def model_forward(self, spec_noisy: Tensor, *args) -> tp.Tuple[Tensor, tp.List[Tensor]]:
        """
        Args:
            spec_noisy (Tensor): [B, F, T, 2] where F = n_fft//2+1
        Returns:
            out (Tensor): [B, F, T, 2]
        """
        cache_in_list = [*args]
        cache_out_list = []

        x = spec_noisy.transpose(2, 3)  # [B, F, 2, T]
        x = self.band_split(x)          # [B, F', C, T] where F' = len(band)
        x = x.permute(3, 0, 1, 2).contiguous()  # [T, B, F', C]
        T, B, F, C = x.shape
        for i in range(self.num_layers):
            skip = x
            x = self.norm_time[i](x)
            x = x.view(T, B*F, C)
            if len(cache_in_list) == 0:
                cache_in = None
            else:
                cache_in = (cache_in_list.pop(0), cache_in_list.pop(0))
            try:
                x, (h, c) = self.rnn_time[i](x, cache_in)
            except:
                breakpoint()
            cache_out_list.extend([h, c])
            x = self.fc_time[i](x)
            x = x.view(T, B, F, C)
            x = x.add_(skip)

            skip = x
            x = self.norm_freq[i](x)
            x = x.view(T*B, F, C)
            x, _ = self.rnn_freq[i](x)
            x = self.fc_freq[i](x)
            x = x.view(T, B, F, C)
            x = x.add_(skip)

        x = x.permute(1, 2, 3, 0)           # [B, F', C, T]
        mask, res = self.mask_decoder(x)    # [B, F, T, 2]
        spec_hat = torch.stack(
            [
                spec_noisy[..., 0] * mask[..., 0] - spec_noisy[..., 1] * mask[..., 1],
                spec_noisy[..., 0] * mask[..., 1] + spec_noisy[..., 1] * mask[..., 0],
            ],
            dim=3
        )
        spec_hat = spec_hat + res   # [B, F, T, 2]
        return spec_hat, cache_out_list

    def flatten_parameters(self):
        for i in range(self.num_layers):
            self.rnn_time[i].flatten_parameters()
            self.rnn_freq[i].flatten_parameters()

    def initialize_cache(self, x: Tensor) -> tp.List[Tensor]:
        freq = len(self.band_split.subbands)
        hidden_size = self.rnn_time[0].hidden_size
        if self.onnx:
            cache_list = [
                x.new_zeros(freq, hidden_size)
                for _ in range(self.num_layers * 2)
            ]
        else:
            cache_list = [
                x.new_zeros(1, freq, hidden_size)
                for _ in range(self.num_layers * 2)
            ]
        return cache_list

    def forward(
        self,
        spec_noisy: Tensor,
        *args
    ) -> tp.Tuple[Tensor, Tensor, ...]:
        """ input/output: [B, n_fft//2+1, T_spec, 2]"""
        # Compress
        mag = torch.linalg.norm(
            spec_noisy,
            dim=-1,
            keepdim=True
        ).clamp(min=1.0e-5)
        spec_noisy = spec_noisy * mag.pow(self.input_compression - 1.0)

        # Model forward
        spec_hat, cache_out_list = self.model_forward(spec_noisy, *args)

        # Uncompress
        mag_compressed = torch.linalg.norm(
            spec_hat,
            dim=-1,
            keepdim=True
        )
        spec_hat = spec_hat * mag_compressed.pow(1.0 / self.input_compression - 1.0)
        return spec_hat, *cache_out_list

    def load_state_dict(self, state_dict, strict: bool = True) -> None:
        if not self.onnx:
            super().load_state_dict(state_dict, strict=False)
            return

        new_state_dict = {}
        for name, param in state_dict.items():
            if re.match(r"rnn_time.+_l0$", name) is not None:
                name = name.replace("_l0", "")
            new_state_dict[name] = param
        super().load_state_dict(new_state_dict, strict=strict)


class Model(ONNXModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, onnx=False, **kwargs)

    def get_stft(
        self, n_fft: int, hop_size: int, win_size: int,
        window: str, normalized: bool = False
    ) -> nn.Module:
        return CompressedSTFT(
            n_fft=n_fft, hop_size=hop_size, win_size=win_size,
            win_type=window, normalized=normalized,
            compression=self.input_compression
        )

    def forward(self, noisy: Tensor) -> tp.Tuple[Tensor, Tensor]:
        """ input/output: [B, T_wav]"""
        spec_noisy = self.stft(noisy)                   # [B, F, T, 2]
        spec_hat, _ = self.model_forward(spec_noisy)    # [B, F, T, 2]
        spec_hat = torch.view_as_complex(spec_hat)
        wav_hat = self.stft.inverse(spec_hat)   # [B, T_wav]
        return wav_hat, torch.view_as_real(spec_hat)


def test():
    x = torch.randn(3, 16_000)
    from utils import get_hparams
    hparams = get_hparams("configs/others/bsrnn_xt.yaml")
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
