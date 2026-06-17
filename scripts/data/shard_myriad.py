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


T60_S = {
    "SAL": 2.1,
    "AIL": 0.5,
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
    idx, wav_path, t60_s = args
    rir, rir_fs = librosa.load(wav_path, sr=None, mono=True)
    assert rir_fs == worker_fs, (
        f"Expected sampling rate of {worker_fs}, but got {rir_fs} for file {wav_path}"
    )
    p = Path(wav_path)
    cut_id = f"{p.parts[-3]}_{p.parts[-2]}_{p.stem}_{idx}"
    cut = numpy_to_rec(rir, rir_fs, cut_id).to_cut()
    onset_sample = get_rir_start_sample(rir).item()
    cut.custom = dict(
        onset_sample=onset_sample,
        t60=t60_s,
        is_real=True,
    )
    return cut


def build_file_entries(base_dir: Path) -> list:
    entries = []
    for idx, wav_path in enumerate(
        sorted(wav for subdir in T60_S for wav in (base_dir / subdir).rglob("*.wav"))
    ):
        t60_s = T60_S[wav_path.relative_to(base_dir).parts[0]]
        entries.append((idx, str(wav_path), t60_s))
    return entries


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-n", "--name",
        type=str,
        default="myriad",
        help="Name of the dataset. It will be used as a prefix for the manifest files."
    )
    parser.add_argument(
        "-b", "--base-dir",
        type=Path,
        default="/home/shahn/Datasets/MYRiAD/MYRiAD_V2_econ/16khz",
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
        help="Run in debug mode (process only 100 files)."
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
        entries = entries[:100]
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
            for cut in tqdm(
                pool.imap_unordered(process_file, entries),
                desc=f"Processing and saving to {args.shar_dir / args.name}",
                dynamic_ncols=True,
                smoothing=0,
                total=len(entries),
            ):
                if isinstance(cut, str):
                    tqdm.write(cut)
                    continue
                try:
                    writer.write(cut)
                    num_processed += 1
                    dataset_length += cut.duration
                except Exception as e:
                    tqdm.write(f"Error writing cut {cut.id}: {e}")

    print(
        f"{num_processed} RIR files processed, "
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
