import os
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm


def resample(
    from_dir: Path,
    to_dir: Path,
    from_file: Path,
    sr: int,
    to_extension: str,
    res_type: str,
) -> int:
    try:
        wav, _ = librosa.load(str(from_file), sr=sr, res_type=res_type)
        wav_max = np.max(np.abs(wav))
        if wav_max > 1.0:
            wav = wav / wav_max * 0.99
        to_file = to_dir / from_file.relative_to(from_dir).with_suffix(f".{to_extension}")
        to_file.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(to_file), wav, sr)
        return len(wav)
    except Exception as e:
        print(e)
        return -1


def get_time_string(seconds: float) -> str:
    seconds = int(seconds)
    second = seconds % 60
    minute = seconds // 60 % 60
    hour = seconds // 3600
    return f"{hour}:{minute:02d}:{second:02d}"


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--to-sr",
        type=int,
        default=16_000
    )
    parser.add_argument(
        "--from-extension",
        type=str,
        default="wav"
    )
    parser.add_argument(
        "--to-extension",
        type=str,
        default="wav"
    )
    parser.add_argument(
        "--from-dir",
        type=str,
        default="/home/shahn/Datasets/voicebank-demand/48k/"
    )
    parser.add_argument(
        "--to-dir",
        type=str,
        default="/home/shahn/Datasets/voicebank-demand/16k/"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=32
    )
    parser.add_argument(
        "--res-type",
        type=str,
        default="soxr_hq",
        choices=["soxr_hq", "soxr_vhq", "sinc_best", "kaiser_best", "polyphase"],
        help="Resample type. Default: soxr_hq"
    )
    args = parser.parse_args()
    
    from_dir = Path(args.from_dir)
    to_dir = Path(args.to_dir)
    
    futures = []
    num_workers = min(args.num_workers, os.cpu_count())
    print(f"Resample with {num_workers} workers (os.cpu_count()={os.cpu_count()})")
    
    filelist = list(from_dir.rglob(f"*.{args.from_extension}"))
    time_total = 0.0
    with ProcessPoolExecutor(max_workers=num_workers) as e:
        for file in tqdm(filelist, desc="Submitting", dynamic_ncols=True, smoothing=0):
            futures.append(
                e.submit(resample, from_dir, to_dir, file, args.to_sr,
                         args.to_extension, args.res_type)
            )
        for idx, future in tqdm(
            enumerate(as_completed(futures), start=1),
            desc="Retrieving",
            dynamic_ncols=True,
            smoothing=0,
            total=len(filelist)
        ):
            if future.result() == -1:
                exit()
            time_total += future.result() / args.to_sr
        print(
            f"Number of utterances: {len(filelist)}, "
            f"total length: {get_time_string(time_total)}"
        )
