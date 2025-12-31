import time
import argparse

import numpy as np
import onnxruntime
import librosa
from tqdm import tqdm
import scipy.io.wavfile


def main(args):
    # Load input
    print("Preparing input...", end=" ")
    wav, _ = librosa.load(args.audio_path, sr=16_000)
    wav = wav.reshape(1, -1)
    wav = np.clip(wav, a_min=-1, a_max=1)
    T = wav.shape[1]
    wav = np.concatenate([wav] * 8, axis=1)
    length = wav.shape[-1]
    wav = np.pad(wav, ((0, 0), (0, args.n_fft)), mode='constant')  # pad right

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
    wav_out = []
    tic = time.perf_counter()
    for idx in tqdm(range(0, length+args.n_fft-args.hop_size, args.hop_size)):
        onnx_input["wav_in"] = wav[:, idx:idx+args.hop_size]
        out = sess.run(None, onnx_input)
        wav_out.append(out[0][0])
        for j in range(len(out[1:])):
            onnx_input[f"cache_in_{j}"] = out[j+1]
    toc = time.perf_counter()
    print(f">>> RTF: {(toc - tic) * 16_000 / length}")

    if args.save_output:
        print("Saving the output audio...", end=" ")
        wav_out = np.concatenate(wav_out, axis=0)        
        start_idx = args.n_fft - args.hop_size
        wav_out = wav_out[start_idx:start_idx+length]
        wav_out = np.clip(wav_out, a_min=-1.0, a_max=1.0)
        wav_out = wav_out[:T]
        scipy.io.wavfile.write("onnx/delete_it_onnx.wav", 16_000, wav_out)
        print("✅")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--onnx-path', type=str, required=True,
        help="Path to save exported onnx file."
    )
    parser.add_argument(
        '--audio-path', type=str,
        default="onnx/p232_013.wav",
        help="Path to audio."
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

    args = parser.parse_args()
    main(args)
