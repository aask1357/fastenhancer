import time
import argparse

import numpy as np
import torch
import onnxruntime
import librosa
from tqdm import tqdm
import scipy.io.wavfile


def main(args):
    # Load input
    print("Preparing input...", end=" ")
    wav, _ = librosa.load(args.audio_path, sr=16_000)
    wav = torch.from_numpy(wav).view(1, -1).clamp(min=-1, max=1)
    wav = torch.cat([wav] * 8, dim=1)
    length = wav.size(-1) // args.hop_size * args.hop_size
    wav = wav[:, :length]
    if args.win_type == "hann":
        window = torch.hann_window(args.win_size)
    elif args.win_type == "hann-sqrt":
        window = torch.hann_window(args.win_size).sqrt()
    else:
        raise ValueError(f"Unsupported window type: {args.win_type}")
    spec = wav.stft(
        n_fft=args.n_fft, hop_length=args.hop_size, win_length=args.win_size,
        onesided=True, window=window, normalized=False, return_complex=True
    )           # [B, F+1, T]
    spec = torch.view_as_real(spec) # [B, F+1, T, 2]

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

    # Prepare cache
    onnx_input = {
        x.name: np.zeros(x.shape, dtype=np.float32)
        for x in sess.get_inputs()
        if x.name.startswith("cache_in_")
    }

    # Inference
    print("✅\nInferencing...")
    spec_out = []
    spec = spec.numpy()
    tic = time.perf_counter()
    for idx in tqdm(range(0, spec.shape[2])):
        onnx_input["spec_in"] = spec[:, :, idx:idx+1, :]
        out = sess.run(None, onnx_input)
        spec_out.append(out[0])
        for j in range(len(out)-1):
            onnx_input[f"cache_in_{j}"] = out[j+1]
    toc = time.perf_counter()
    print(f">>> RTF: \n{(toc - tic) * 16_000 / length}")

    if args.save_output:
        print("Saving the output audio...", end=" ")
        spec_out = torch.from_numpy(np.concatenate(spec_out, axis=2))
        spec_out = torch.view_as_complex(spec_out)
        wav_out = spec_out.istft(
            n_fft=args.n_fft, hop_length=args.hop_size, win_length=args.win_size,
            window=window, return_complex=False
        ).clamp(min=-1.0, max=1.0).squeeze()
        scipy.io.wavfile.write("onnx/delete_it_onnx.wav", 16_000, wav_out.numpy())
        print("✅")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--audio-path', type=str,
        default="onnx/p232_013.wav",
        help="Path to audio."
    )
    parser.add_argument(
        '--onnx-path', type=str,
        help="Path to save exported onnx file."
    )
    parser.add_argument('--save-output', action='store_true')
    parser.add_argument(
        '--n-fft', type=int, default=512,
        help="FFT size."
    )
    parser.add_argument(
        '--hop-size', type=int, default=256,
        help="Hop size."
    )
    parser.add_argument(
        '--win-size', type=int, default=512,
        help="Window size."
    )
    parser.add_argument(
        '--win-type', type=str, default="hann",
        help="Window type."
    )
    
    args = parser.parse_args()
    main(args)
