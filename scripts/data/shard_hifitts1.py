from pathlib import Path
import argparse
import contextlib
import io
import os
import functools
import traceback
import multiprocessing as mp
import shutil

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


worker_data_dir = Path("")
worker_fs = 0
worker_min_sec = 0.0
worker_res_type = ""
worker_mono = False


@stop_on_init_error
def init_worker(data_dir: Path, fs: int, min_sec: float,
                res_type: str, mono: bool):
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    import torch
    torch.set_num_threads(1)

    global worker_data_dir, worker_fs, worker_res_type, worker_mono
    worker_data_dir = data_dir
    worker_min_sec = min_sec
    worker_fs = fs
    worker_res_type = res_type
    worker_mono = mono


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
    cut = numpy_to_rec(audio, worker_fs, str(idx)).to_cut()
    rms = active_rms_relative(audio, worker_fs)
    if rms == 0.0:
        return f"File {file} has zero active RMS, skipping."
    if cut.duration < worker_min_sec:
        return f"File {file} is too short ({cut.duration:.2f}s), skipping."
    cut.custom = dict(rms=rms)
    return cut


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-n", "--name",
        type=str,
        required=True,
        help="Name of the dataset. It will be used as a prefix for the manifest files."
    )
    parser.add_argument(
        "-d", "--data-dir",
        type=Path,
        default="/home/shahn/Datasets/hifitts/audio",
        help="Data directory containing HiFi-TTS speaker folders."
    )
    parser.add_argument(
        "-s", "--shar-dir",
        default="/home/shahn/Datasets/asr/speech/shard/",
        type=Path,
    )
    parser.add_argument(
        "-r", "--resample-type",
        type=str,
        default="polyphase",
        help=(
            "Resampling type. See librosa.load() documentation for more details. "
        )
    )
    parser.add_argument(
        "--fs",
        type=int,
        default=16000,
        help="Sampling rate to resample the audio to."
    )
    parser.add_argument(
        "--min-sec",
        type=float,
        default=0.5,
        help=(
            "Minimum allowed duration for a chunk in seconds. "
            "Chunks shorter than this will be discarded."
        )
    )
    parser.add_argument(
        "-e", "--extension",
        type=str,
        default="flac",
        help="File extension of the audio files to process."
    )
    parser.add_argument(
        "-f", "--force",
        action='store_true',
        help='Force overwrite existing files.'
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
    parser.add_argument(
        "--mono",
        action="store_true",
        help="Convert audio to mono."
    )
    return parser.parse_args()


def main():
    args = get_args()
    args.num_jobs = min(args.num_jobs, mp.cpu_count()-1)
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
    clean_dirs = sorted(args.data_dir.glob("*_clean"))
    files = []
    for clean_dir in clean_dirs:
        for root, dirs, filenames in os.walk(clean_dir, followlinks=True):
            for filename in filenames:
                if filename.endswith(f".{args.extension}"):
                    files.append(Path(root) / filename)
    if args.debug:
        args.num_jobs = 1
        files = files[:10]
    print("✅")
    print(f"Found {len(files)} files in {len(clean_dirs)} *_clean directories.")

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
                    initargs=(args.data_dir, args.fs,
                              args.min_sec, args.resample_type, args.mono),
                ) as pool:
                    for cut in pool.imap_unordered(
                        process_single_line,
                        enumerate(files[i:i+args.batch_size], start=i),
                        chunksize=10
                    ):
                        if isinstance(cut, str):
                            tqdm.write(cut)
                            pbar.update(1)
                            continue
                        try:
                            writer.write(cut)
                            num_processed += 1
                            dataset_length += cut.duration
                        except Exception as e:
                            tqdm.write(f"Error writing cut {cut.id}: {e}")
                        pbar.update(1)
    print(
        f"{num_processed} audio files processed, "
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