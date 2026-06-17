from pathlib import Path, PureWindowsPath
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

import librosa
from tqdm import tqdm
from lhotse import CutSet
from lhotse.shar.writers import SharWriter

from utils import get_time_string
from scripts.audiolib import find_rir_onset_spectral, get_rir_start_sample
from .utils import numpy_to_rec


def stop_on_init_error(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"\n[FATAL ERROR] {func.__name__} 초기화 실패!")
            traceback.print_exc()  # print detailed traceback for debugging
            # Exit the entire process immediately to prevent hanging workers
            os._exit(1)
    return wrapper


worker_base_dir = None
worker_fs = None


@stop_on_init_error
def init_worker(base_dir: Path, fs: int):
    global worker_base_dir, worker_fs
    worker_base_dir = base_dir
    worker_fs = fs


@stop_on_init_error
def process_single_line(args):
    idx, line = args
    path, ch, t60, c50, is_real = line.strip().split(",")
    ch = int(ch)
    full_path = worker_base_dir / PureWindowsPath(path).as_posix()
    rir, rir_fs = librosa.load(full_path, sr=None, mono=False)
    if rir.ndim > 1:
        rir = rir[ch-1]
    elif ch != 1:
        raise ValueError(f"File {full_path} is mono but channel {ch} was requested.")
    assert rir_fs == worker_fs, (
        f"Expected sampling rate of {worker_fs}, but got {rir_fs} for file {full_path}"
    )
    cut = numpy_to_rec(rir, rir_fs, str(idx)).to_cut()
    onset_sample, _ = find_rir_onset_spectral(rir, rir_fs)
    onset_heuristic = get_rir_start_sample(rir).item()
    if abs(onset_sample - onset_heuristic) > 0.001 * worker_fs:
        print(f"Onset mismatch for file {full_path}: spectral={onset_sample} vs heuristic={onset_heuristic}")

    cut.custom = dict(
        onset_sample=onset_heuristic,
        t60=float(t60),
        c50=float(c50),
        is_real=bool(int(is_real)),
    )
    return cut


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-n", "--name",
        type=str,
        default="dns-rir",
        help="Name of the dataset. It will be used as a prefix for the manifest files."
    )
    parser.add_argument(
        "-b", "--base-dir",
        type=Path,
        default="/home/shahn/Datasets/DNS-Challenge/16khz/impulse_responses",
        help="Base directory containing the RIR files."
    )
    parser.add_argument(
        "-c", "--csv-file",
        type=str,
        default="/home/shahn/Datasets/DNS-Challenge/16khz/acoustic_params/RIR_table_simple.csv",
        help="Path to the CSV file."
    )
    parser.add_argument(
        "-s", "--shar-dir",
        default="/home/shahn/Datasets/asr/rir/shard/",
        type=Path,
    )
    parser.add_argument(
        "--fs",
        type=int,
        default=16000,
        help="Sampling rate to resample the audio to."
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
        "--debug",
        action="store_true",
        help="Run in debug mode (process only 100 lines)."
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
        

    with open(args.csv_file) as f:
        lines = f.readlines()[1:]  # Skip header
    if args.debug:
        args.num_jobs = 1
        lines = lines[:100]

    num_processed = 0
    dataset_length = 0

    with SharWriter(
        args.shar_dir / args.name,
        fields={"recording": "flac"},
        shard_size=1000,
    ) as writer:
        with mp.Pool(
            processes=args.num_jobs,
            initializer=init_worker,
            initargs=(args.base_dir, args.fs),
        ) as pool:
            for cut in tqdm(
                pool.imap_unordered(process_single_line, enumerate(lines)),
                desc=f"Processing and saving to {args.shar_dir / args.name}",
                dynamic_ncols=True,
                smoothing=0,
                total=len(lines),
            ):
                if isinstance(cut, str):
                    tqdm.write(cut)  # cut contains the error message in this case
                    continue
                try:
                    writer.write(cut)
                    num_processed += 1
                    dataset_length += cut.duration
                except Exception as e:
                    tqdm.write(f"Error writing cut {cut.id}: {e}")
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
