from pathlib import Path
import argparse
import os
import functools
import traceback
import multiprocessing as mp
import shutil
import contextlib
import io

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import numpy as np
import librosa
from tqdm import tqdm
from lhotse import CutSet
from lhotse.shar.writers import SharWriter

from scripts.audiolib import active_rms_relative
from utils import get_time_string
from .utils import numpy_to_rec


TRAIN_SCENES = {
    "DKITCHEN", "DWASHING", "NFIELD", "NPARK", "NRIVER",
    "OHALLWAY", "OMEETING", "PCAFETER", "PRESTO", "PSTATION",
    "STRAFFIC", "TCAR", "TMETRO",
}


def stop_on_init_error(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"\n[FATAL ERROR] {func.__name__} 초기화 실패!")
            traceback.print_exc()
            raise e
    return wrapper


worker_fs = 0
worker_chunk = 0
worker_min_chunk = 0
worker_res_type = ""


@stop_on_init_error
def init_worker(fs: int, chunk_sec: float, min_sec: float, res_type: str):
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    import torch
    torch.set_num_threads(1)

    global worker_fs, worker_chunk, worker_min_chunk, worker_res_type
    worker_fs = fs
    worker_chunk = int(fs * chunk_sec)
    worker_min_chunk = int(fs * min_sec)
    worker_res_type = res_type


@stop_on_init_error
def process_single_line(args):
    idx, file = args
    try:
        audio = librosa.load(
            file, sr=worker_fs, mono=True,
            res_type=worker_res_type
        )[0]
    except Exception as e:
        return f"Error processing {file}: {e}"
    num_samples = len(audio)
    cuts = []
    chunk_idx = 0
    start = 0
    while start < num_samples:
        end = min(start + worker_chunk, num_samples)
        if end - start < worker_min_chunk:
            break
        chunk_data = audio[start:end]
        rec_id = f"{idx}_{chunk_idx}"
        cut = numpy_to_rec(chunk_data, worker_fs, rec_id).to_cut()
        rms = active_rms_relative(chunk_data, worker_fs)
        cut.custom = dict(rms=rms)
        cuts.append(cut)
        start += worker_chunk
        chunk_idx += 1
    return cuts


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-n", "--name",
        type=str,
        required=True,
        help="Name of the dataset. Used as a prefix for the manifest files."
    )
    parser.add_argument(
        "-d", "--data-dir",
        type=Path,
        default="/home/shahn/Datasets/demand/16khz",
        help="DEMAND data directory containing per-scene subdirectories."
    )
    parser.add_argument(
        "-s", "--shar-dir",
        default="/home/shahn/Datasets/asr/noise/shard/",
        type=Path,
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test"],
        default="train",
        help=(
            "Which split to process. "
            "train: DKITCHEN, DWASHING, NFIELD, NPARK, NRIVER, OHALLWAY, OMEETING, "
            "PCAFETER, PRESTO, PSTATION, STRAFFIC, TCAR, TMETRO. "
            "test: all other scenes."
        )
    )
    parser.add_argument(
        "-r", "--resample-type",
        type=str,
        default="polyphase",
        help="Resampling type. See librosa.load() documentation for more details."
    )
    parser.add_argument(
        "--fs",
        type=int,
        default=16000,
        help="Sampling rate to resample the audio to."
    )
    parser.add_argument(
        "--chunk-sec",
        type=float,
        default=10.0,
        help="Duration of each audio chunk in seconds."
    )
    parser.add_argument(
        "--min-sec",
        type=float,
        default=9.0,
        help=(
            "Minimum allowed duration for a chunk in seconds. "
            "Chunks shorter than this will be discarded."
        )
    )
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        help="Force overwrite existing files."
    )
    parser.add_argument(
        "-j", "--num-jobs",
        type=int,
        default=16,
        help="Number of parallel jobs to run for processing the data."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1280,
        help=(
            "Number of audio files to process before writing to SHAR. "
            "If shar writing speed is slower than audio processing speed, cuts will "
            "be accumulated in memory, resulting in OOM. Therefore, we split the "
            "audio files into batches, process each batch, write to SHAR, and "
            "repeat until all files are processed. If set too small, the mp.Pool "
            "overhead may cause slowdown. If set too large, it may cause OOM."
        )
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode (process only 10 files)."
    )
    return parser.parse_args()


def main():
    args = get_args()
    args.num_jobs = min(args.num_jobs, mp.cpu_count() - 1)

    if (args.shar_dir / f".{args.name}.done").exists():
        if not args.force:
            print(
                f"Previous manifests computed for {args.name} found. "
                f"If you want to overwrite, use --force option."
            )
            return
        print(f"Previous manifests computed for {args.name} found. Overwriting.")
        shutil.rmtree(args.shar_dir / args.name)
    (args.shar_dir / args.name).mkdir(parents=True, exist_ok=True)

    print("Collecting audio files...", end=" ", flush=True)
    files = []
    scenes_used = []
    for scene_dir in sorted(args.data_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        # Directory names have a suffix like "_16k"; strip it to get the scene name.
        scene = scene_dir.name.split("_")[0]
        is_train = scene in TRAIN_SCENES
        if args.split == "train" and not is_train:
            continue
        if args.split == "test" and is_train:
            continue
        scenes_used.append(scene)
        files.extend(sorted(scene_dir.rglob("*.wav")))

    if args.debug:
        args.num_jobs = 1
        files = files[:10]
    print("✅")
    print(
        f"Found {len(files)} channel files from {len(scenes_used)} scenes "
        f"(split={args.split}): {', '.join(scenes_used)}."
    )

    num_processed = 0
    dataset_length = 0

    with SharWriter(
        args.shar_dir / args.name,
        fields={"recording": "flac"},
        shard_size=1000,
    ) as writer:
        with tqdm(
            total=len(files),
            desc=f"Processing to {args.shar_dir / args.name}",
            dynamic_ncols=True,
            smoothing=0,
        ) as pbar:
            for i in range(0, len(files), args.batch_size):
                with mp.Pool(
                    processes=args.num_jobs,
                    initializer=init_worker,
                    initargs=(args.fs, args.chunk_sec, args.min_sec, args.resample_type),
                ) as pool:
                    for cuts in pool.imap_unordered(
                        process_single_line,
                        enumerate(files[i:i + args.batch_size], start=i),
                        chunksize=10,
                    ):
                        if isinstance(cuts, str):
                            tqdm.write(cuts)
                            pbar.update(1)
                            continue
                        for cut in cuts:
                            try:
                                writer.write(cut)
                                num_processed += 1
                                dataset_length += cut.duration
                            except Exception as e:
                                tqdm.write(f"Error writing cut {cut.id}: {e}")
                        pbar.update(1)

    print(
        f"{num_processed} chunks processed, "
        f"total duration: {get_time_string(dataset_length)}."
    )
    cuts = CutSet.from_shar(in_dir=args.shar_dir / args.name)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cuts.describe()
    description = buf.getvalue()
    print(description, end="")
    (args.shar_dir / f".{args.name}.done").write_text(description)


if __name__ == "__main__":
    main()
