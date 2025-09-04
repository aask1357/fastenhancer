import time
import argparse
import os
import importlib
from typing import List, Tuple

import numpy as np
import torch
import torch.onnx
from torch import Tensor
import onnx
import onnxruntime
from onnxsim import simplify
import librosa
from tqdm import tqdm
import scipy.io.wavfile

from utils import get_hparams
from wrapper import ModelWrapper
from functional import ONNXSTFT, ONNXiSTFT


def onnx_simplify(onnx_model):
    onnx_model, check = simplify(onnx_model)
    assert check, "Simplify failed."
    return onnx_model


def get_onnx_model(hps : str) -> torch.nn.Module:
    model: str = hps.model
    module = importlib.import_module(f"models.{model}.model")
    return module.ONNXModel(**hps.model_kwargs)


class Model(torch.nn.Module):
    def __init__(self, hps):
        super().__init__()
        self.model = get_onnx_model(hps)

    def initialize_cache(self, x: Tensor) -> List[Tensor]:
        cache_list = [self.model.stft.initialize_cache(x)]
        cache_list.extend(self.model.initialize_cache(x))
        return cache_list

    def forward(
        self,
        wav_in: Tensor,
        cache_istft: Tensor,
        *cache_model
    ) -> Tuple[Tensor, Tensor, ...]:
        spec_in = self.model.stft(wav_in)
        spec_out, *cache_model = self.model(spec_in, *cache_model)
        wav_out, cache_istft = self.model.stft.inverse(spec_out, cache_istft)
        return wav_out, cache_istft, *cache_model


def main(args):
    # Load a model
    if args.name is not None:
        args.name = os.path.join("logs", args.name)
    hps = get_hparams(args.config, args.name)
    if args.onnx_path is None:
        args.onnx_path = os.path.join("onnx", f"{hps.model}.onnx")
    n_fft = hps.model_kwargs.n_fft
    hop_size = hps.model_kwargs.hop_size
    win_size = hps.model_kwargs.win_size
    wrapper = ModelWrapper(hps)
    wrapper.load()
    wrapper.eval()

    model = Model(hps)
    model.eval()
    model.model.load_state_dict(wrapper.model.state_dict(), strict=True)
    model.model.remove_weight_reparameterizations()
    total_params = sum(p.numel() for n, p in model.model.named_parameters())
    print(f"Number of total parameters: {total_params}")

    # Load input
    print("Preparing input...", end=" ")
    wav, _ = librosa.load(args.audio_path, sr=wrapper.sr)
    wav = torch.from_numpy(wav).view(1, -1).clamp(min=-1, max=1)
    if not args.test_streaming and not args.test_remove_weight_reparam:
        wav = torch.cat([wav] * 8, dim=1)
    length = wav.size(-1) // wrapper.hop_size * wrapper.hop_size
    wav = wav[:, :length]
    wav = torch.nn.functional.pad(wav, (n_fft-hop_size, 0))     # pad left
    window = wrapper.model.stft.window
    spec = wav.stft(
        n_fft=n_fft, hop_length=hop_size, win_length=win_size, onesided=True,
        window=window, normalized=False, return_complex=True
    )           # [B, F+1, T]
    spec = torch.view_as_real(spec) # [B, F+1, T, 2]

    # Prepare cache
    cache_list = model.initialize_cache(torch.randn(1))

    # Test
    if args.test_remove_weight_reparam:
        print("✅\nTesing remove_weight_reparameterizations...", end=" ")
        with torch.no_grad():
            wav_out1, *_ = wrapper.model(wav)
        scipy.io.wavfile.write(
            "onnx/delete_it_original.wav",
            16_000,
            wav_out1.clamp(min=-1, max=1).squeeze().numpy()
        )
        wrapper.model.remove_weight_reparameterizations()
        with torch.no_grad():
            wav_out2, *_ = wrapper.model(wav)
        scipy.io.wavfile.write(
            "onnx/delete_it_remove_weight_reparams.wav",
            16_000,
            wav_out2.clamp(min=-1, max=1).squeeze().numpy()
        )
        scipy.io.wavfile.write(
            "onnx/delete_it_diff.wav",
            16_000,
            (wav_out1 - wav_out2).clamp(min=-1, max=1).squeeze().numpy()
        )
        print("✅")
        exit()

    if args.test_streaming:
        print("✅\nTesing streaming inference...")
        with torch.no_grad():
            wav_out, *_ = wrapper.model(wav)
            wav_out = wav_out[:, :-(n_fft - hop_size)]
        scipy.io.wavfile.write(
            "onnx/delete_it_original.wav",
            16_000,
            wav_out.clamp(min=-1, max=1).squeeze().numpy()
        )
        with torch.no_grad():
            out = []
            for i in tqdm(range(0, length, hop_size)):
                # NOTE) wav.shape = n_fft - hop_size + length where length = k * hop_size
                # wav is left-zero-padded
                # wav_out.shape = length
                wav_i = wav[:, i:i+n_fft]
                wav_hat, *cache_list = model(spec_in, *cache_list)
                out.append(wav_hat)         # [B, hop_size]
            wav_hat = torch.cat(out, dim=1) # [B, T_wav = hop_size * T_spec = length - (n_fft - hop_size)]
        scipy.io.wavfile.write(
            "onnx/delete_it_streaming.wav",
            16_000,
            wav_hat.clamp(min=-1, max=1).squeeze().numpy()
        )
        scipy.io.wavfile.write(
            "onnx/delete_it_diff.wav",
            16_000,
            (wav_out - wav_hat).clamp(min=-1, max=1).squeeze().numpy()
        )
        exit()

    # Export the model to ONNX
    print("Exporting the model to ONNX...")
    torch.onnx.export(
        model,
        args=(wav[:, :n_fft], *cache_list),
        f=args.onnx_path,
        input_names=['wav_in'] + [f'cache_in_{i}' for i in range(len(cache_list))],
        output_names=['wav_out'] + [f'cache_out_{i}' for i in range(len(cache_list))],
        dynamo=True
    )
    onnx_model = onnx.load(args.onnx_path)
    onnx_model = onnx_simplify(onnx_model)
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, args.onnx_path)

    # Create an ONNXRuntime session
    print("✅\nCreating a ONNXRuntime session...", end=" ")
    sess_options = onnxruntime.SessionOptions()
    sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    sess = onnxruntime.InferenceSession(
        args.onnx_path,
        sess_options=sess_options,
        providers=['CPUExecutionProvider']
    )

    # Inference
    print("✅\nInferencing...")
    wav_out = []
    wav = wav.numpy()
    cache_list = [x.numpy() for x in cache_list]
    onnx_input = {f"cache_in_{j}": x for j, x in enumerate(cache_list)}
    tic = time.perf_counter()
    # mapping = {
    #     output.name: idx for idx, output in enumerate(sess.get_outputs())
    # }
    for idx in tqdm(range(0, length, hop_size)):
        onnx_input["wav_in"] = wav[:, idx:idx+n_fft]
        out = sess.run(None, onnx_input)
        # wav_out_i = out[mapping["istft_wav_out"]]   # [batch_size=1, hop_size]
        wav_out.append(out[0][0])
        for j in range(len(out[1:])):
            onnx_input[f"cache_in_{j}"] = out[j+1]
    toc = time.perf_counter()
    print(f">>> RTF: {(toc - tic) * 16_000 / length}")

    if args.save_output:
        print("Saving the output audio...", end=" ")
        wav_out = np.concatenate(wav_out, axis=0)
        wav_out = np.clip(wav_out, a_min=-1.0, a_max=1.0)
        scipy.io.wavfile.write("onnx/delete_it_onnx.wav", 16_000, wav_out)
        print("✅")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-n', '--name', type=str,
        help=(
            "Checkpoint directory name at logs/{name}. "
            "Either args.name or args.config should be given."
        )
    )
    parser.add_argument(
        '-c', '--config', type=str,
        help=(
            "Path to config json file. Default: logs/{name}/config.yaml. "
            "Either args.name or args.config should be given."
        )
    )
    parser.add_argument(
        '--audio-path', type=str,
        default="/home/shahn/Datasets/voicebank-demand/16k/noisy_testset_wav/p232_013.wav",
        help="Path to audio."
    )
    parser.add_argument(
        '--onnx-path', type=str,
        help="Path to save exported onnx file."
    )
    parser.add_argument('--test-streaming', action='store_true')
    parser.add_argument('--test-remove-weight-reparam', action='store_true')
    parser.add_argument('--save-output', action='store_true')
    parser.add_argument('--dynamo')
    
    args = parser.parse_args()
    main(args)
