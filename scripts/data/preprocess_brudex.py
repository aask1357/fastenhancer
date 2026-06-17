import os
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
import torchaudio
import soundfile as sf
import numpy as np
import mat73
from tqdm import tqdm


def resample_rir(
    from_dir: Path,
    to_dir: Path,
    from_file: Path,
    sr: int,
    lowpass_filter_width: int,
    rolloff: float,
) -> int:
    try:
        mat = mat73.loadmat(str(from_dir / from_file))
        data = mat["data"]  # (samples, channels)
        orig_sr = int(float(mat["fs"]))

        wav = torch.from_numpy(data.T).float()  # (channels, samples)
        if orig_sr != sr:
            wav = torchaudio.functional.resample(
                wav,
                orig_freq=orig_sr,
                new_freq=sr,
                lowpass_filter_width=lowpass_filter_width,
                rolloff=rolloff,
            )

        wav_np = wav.numpy().T  # (samples, channels)
        wav_max = np.max(np.abs(wav_np))
        if wav_max > 1e-8:
            wav_np = wav_np / wav_max * 0.99

        to_file = to_dir / from_file.with_suffix(".wav")
        to_file.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(to_file), wav_np, sr)
        return wav_np.shape[0]
    except Exception as e:
        print(e)
        return -1


def get_time_string(seconds: float) -> str:
    seconds = int(seconds)
    second = seconds % 60
    minute = seconds // 60 % 60
    hour = seconds // 3600
    return f"{hour}:{minute:02d}:{second:02d}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--to-sr", type=int, default=16_000)
    parser.add_argument(
        "--from-dir",
        type=str,
        default="/home/shahn/Datasets/brudex/rir",
    )
    parser.add_argument(
        "--to-dir",
        type=str,
        default="/home/shahn/Datasets/brudex/rir_16khz",
    )
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--lowpass-filter-width", type=int, default=512)
    parser.add_argument("--rolloff", type=float, default=0.9999)
    args = parser.parse_args()

    from_dir = Path(args.from_dir)
    to_dir = Path(args.to_dir)

    num_workers = min(args.num_workers, os.cpu_count())
    print(f"Resample with {num_workers} workers (os.cpu_count()={os.cpu_count()})")
    print(f"VHQ params: lowpass_filter_width={args.lowpass_filter_width}, rolloff={args.rolloff}")

    print("Searching for files...")
    filelists = []
    for root, dirs, files in os.walk(str(from_dir)):
        for file in files:
            if not file.endswith(".mat"):
                continue
            filelists.append((Path(root) / file).relative_to(from_dir))
    print(f"Total files: {len(filelists)}")

    time_total = 0.0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                resample_rir,
                from_dir,
                to_dir,
                f,
                args.to_sr,
                args.lowpass_filter_width,
                args.rolloff,
            ): f
            for f in filelists
        }
        for future in tqdm(
            as_completed(futures),
            desc="Resampling",
            dynamic_ncols=True,
            smoothing=0,
            total=len(filelists),
        ):
            result = future.result()
            if result == -1:
                print(f"Error processing {futures[future]}")
            else:
                time_total += result / args.to_sr

    print(f"Done. Total audio: {get_time_string(time_total)}")
