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

import librosa
from tqdm import tqdm
from lhotse import CutSet
from lhotse.shar.writers import SharWriter

from utils import get_time_string
from scripts.audiolib import get_rir_start_sample
from .utils import numpy_to_rec


T60_MS = {
    "rev_low": 310.0,
    "rev_medium": 510.0,
    "rev_high": 1300.0,
}


def stop_on_init_error(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"\n[FATAL ERROR] {func.__name__} 초기화 실패!")
            traceback.print_exc()
            os._exit(1)
    return wrapper


worker_fs = None


@stop_on_init_error
def init_worker(fs: int):
    global worker_fs
    worker_fs = fs


@stop_on_init_error
def process_file(args):
    file_idx, wav_path, t60_ms = args
    rirs, rir_fs = librosa.load(wav_path, sr=None, mono=False)
    assert rir_fs == worker_fs, (
        f"Expected sampling rate of {worker_fs}, but got {rir_fs} for file {wav_path}"
    )
    if rirs.ndim == 1:
        rirs = rirs[None, :]  # (1, T)
    stem = f"{Path(wav_path).parent.name}_{Path(wav_path).stem}"
    cuts = []
    for ch_idx, rir in enumerate(rirs):
        cut_id = f"{stem}_ch{ch_idx:02d}_{file_idx}"
        cut = numpy_to_rec(rir, rir_fs, cut_id).to_cut()
        onset_sample = get_rir_start_sample(rir).item()
        cut.custom = dict(
            onset_sample=onset_sample,
            t60=t60_ms,
            is_real=True,
        )
        cuts.append(cut)
    return cuts


def build_file_entries(base_dir: Path) -> list:
    entries = []
    file_idx = 0
    for subdir_name, t60_ms in T60_MS.items():
        subdir = base_dir / subdir_name
        for wav_path in sorted(subdir.glob("*.wav")):
            entries.append((file_idx, str(wav_path), t60_ms))
            file_idx += 1
    return entries


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-n", "--name",
        type=str,
        default="brudex",
        help="Name of the dataset. It will be used as a prefix for the manifest files."
    )
    parser.add_argument(
        "-b", "--base-dir",
        type=Path,
        default="/home/shahn/Datasets/brudex/rir_16khz",
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
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode (process only 100 entries)."
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

    print("Scanning wav files and building entry list...")
    entries = build_file_entries(args.base_dir)
    if args.debug:
        args.num_jobs = 1
        entries = entries[:10]
    print(f"Total files: {len(entries)}")

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
            initargs=(args.fs,),
        ) as pool:
            for cuts in tqdm(
                pool.imap_unordered(process_file, entries),
                desc=f"Processing and saving to {args.shar_dir / args.name}",
                dynamic_ncols=True,
                smoothing=0,
                total=len(entries),
            ):
                if isinstance(cuts, str):
                    tqdm.write(cuts)
                    continue
                for cut in cuts:
                    try:
                        writer.write(cut)
                        num_processed += 1
                        dataset_length += cut.duration
                    except Exception as e:
                        tqdm.write(f"Error writing cut {cut.id}: {e}")

    print(
        f"{num_processed} RIR channels processed, "
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