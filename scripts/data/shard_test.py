from pathlib import Path
import argparse
import os
import functools
import traceback
import multiprocessing as mp
import random
from typing import List, Any, Optional
from dataclasses import dataclass
import shutil
import contextlib
import io

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
from tqdm import tqdm
from lhotse import (
    Recording,
    SupervisionSegment,
    CutSet,
)
from lhotse.array import Array
from lhotse.features.io import MemoryRawWriter
from lhotse.shar.writers import SharWriter

from utils import get_time_string, get_hparams
from scripts.audiolib import active_rms_relative
from .utils import (
    DirectoriesDataset,
    ReverbDataset,
    numpy_to_rec,
    Farend,
)


@dataclass
class WorkerState:
    fs: Optional[int] = None
    activity_threshold_relative: Optional[float] = None
    activity_threshold_absolute: Optional[float] = None
    data_dir: Optional[Path] = None
    farend: Optional[Farend] = None
    nearend_rir: Optional[ReverbDataset] = None
    nearend_rir_prob: Optional[float] = None
    nearend_noise: Optional[DirectoriesDataset] = None
    nearend_snr: Optional[List[float | int]] = None
    nearend_speech_dbFS: Optional[List[float]] = None


def stop_on_init_error(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"\n[FATAL ERROR] {func.__name__} 초기화 실패!")
            traceback.print_exc()  # Print the full traceback for debugging
            raise e
    return wrapper


# @stop_on_init_error
def init_worker(hps):
    # Set random seed differently for each worker
    pid = os.getpid()
    random.seed(pid)
    np.random.seed(pid)

    # Initialize global variables for the worker processes
    global state
    state = WorkerState(
        fs=hps["fs"],
        activity_threshold_relative=hps["activity_threshold_relative"],
        activity_threshold_absolute=hps["activity_threshold_absolute"],
        data_dir=Path(hps["input"]["nearend_speech"]["base_dir"]),
        nearend_snr=list(range(
            hps["nearend"]["noise"]["snr"][0],
            hps["nearend"]["noise"]["snr"][1] + 1,
            hps["nearend"]["noise"]["snr"][2]
        )),
        nearend_speech_dbFS=hps["nearend"]["speech_dbFS"],
    )
    state.nearend_rir_prob = 0.0
    if hasattr(hps["nearend"], "rir"):
        state.nearend_rir_prob = hps["nearend"]["rir"]["prob"]
    assert state.nearend_rir_prob <= 1.0, \
        "Nearend RIR probability must be between 0 and 1."

    if hasattr(hps["input"], "nearend_noise"):
        state.nearend_noise = DirectoriesDataset(
            directories=hps["input"]["nearend_noise"],
            fs=hps["fs"],
            silence_length=hps["silence_length"],
            activity_threshold_relative=hps["activity_threshold_relative"],
            normalize_output=False,
        )

    state.farend = Farend(hps)

    # nearend RIR
    if hasattr(hps["nearend"], "rir"):
        state.nearend_rir = ReverbDataset(
            hps["fs"],
            hps["input"]["rir"],
        )


def match_length(x: np.ndarray, target_length: int) -> np.ndarray:
    if len(x) < target_length:
        x = np.pad(x, (0, target_length - len(x)), mode="constant")
    else:
        x = x[:target_length]
    return x


# @stop_on_init_error
def process_single_line(args):
    idx, path = args

    # Create a cut
    rec = Recording.from_file(state.data_dir / path, recording_id=str(idx))
    cut = rec.resample(state.fs).to_cut()
    speech = cut.load_audio().squeeze()
    rms = active_rms_relative(
        speech, state.fs, state.activity_threshold_relative,
        state.activity_threshold_absolute
    )
    dbFS = random.uniform(*state.nearend_speech_dbFS)
    cut.custom = dict(rms=rms)
    sup = SupervisionSegment(
        id=str(idx),
        recording_id=str(idx),
        start=0,
        duration=cut.duration,
        custom=dict(
            dbFS=dbFS,
        ),
    )
    cut.supervisions.append(sup)

    # Apply nearend rir if needed
    sampled_r = random.random()
    if state.nearend_rir is not None:
        rir, t60, onset = state.nearend_rir()
        _mem_writer = MemoryRawWriter()
        cut.rir = Array(
            storage_type="memory_raw",
            storage_path=None,
            storage_key=_mem_writer.write(str(idx), rir),
            shape=list(rir.shape),
        )
        cut.custom["rir_t60"] = t60
        cut.custom["rir_onset"] = onset
        sup.custom["return_rir"] = (sampled_r < state.nearend_rir_prob)

    # Load noise
    if state.nearend_noise is not None:
        noise, _ = state.nearend_noise(cut.duration)
        noise = match_length(noise, len(speech))
        rms = active_rms_relative(
            noise, state.fs, state.activity_threshold_relative,
            state.activity_threshold_absolute
        )
        cut.noise = numpy_to_rec(noise, state.fs, str(idx))
        cut.custom["rms_noise"] = rms
        snr = random.choice(state.nearend_snr)
        sup.custom["snr"] = snr

    # Load farend and generate echo
    if state.farend is not None:
        farend, echo, ser, farend_type, farend_exists, echo_exists = state.farend(cut.duration)
        farend = match_length(farend, len(speech))
        echo = match_length(echo, len(speech))
        cut.farend = numpy_to_rec(farend, state.fs, f"{idx}_farend")
        cut.echo = numpy_to_rec(echo, state.fs, f"{idx}_echo")
        rms_echo = active_rms_relative(
            echo, state.fs, state.activity_threshold_relative,
            state.activity_threshold_absolute
        )
        cut.custom["rms_echo"] = rms_echo

        sup.custom["farend_type"] = farend_type
        sup.custom["return_farend"] = farend_exists
        if not echo_exists:
            ser = float("inf")
        sup.custom["ser"] = ser

    return cut


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-n", "--name",
        type=str,
        default="vctk-test",
        help="Name of the dataset. It will be used as a prefix for the manifest files."
    )
    parser.add_argument(
        "-s", "--shar-dir",
        default="/home/shahn/Datasets/asr/speech/shard/",
        type=Path,
    )
    parser.add_argument(
        "-c", "--config",
        default="scripts/configs/se_test.yaml",
        type=str,
        help="Path to the configuration file."
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


def rglob_with_symlinks(base_path, extension: str):
    base_path = Path(base_path)
    for root, dirs, files in os.walk(base_path, followlinks=True):
        root_path = Path(root)
        for file in files:
            if file.endswith(extension):
                yield root_path / file


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

    hps = get_hparams(args.config)

    data_dir = Path(hps["input"]["nearend_speech"]["base_dir"])
    ext = hps["input"]["nearend_speech"]["extension"]
    try:
        # Python 3.13+ supports rglob with symlinks.
        files = list(data_dir.rglob(f"*{ext}", recurse_symlinks=True))
    except Exception as e:
        # For older Python versions.
        files = list(rglob_with_symlinks(data_dir, ext))
    if args.debug:
        args.num_jobs = 1
        files = files[:100]
    fields = {
        "recording": "flac",
        "farend": "flac",
        "echo": "flac",
    }
    num_processed = 0
    dataset_length = 0

    if hasattr(hps["nearend"], "rir"):
        fields["rir"] = "numpy"
    if hasattr(hps["input"], "nearend_noise"):
        fields["noise"] = "flac"
    print(fields)
    with SharWriter(
        args.shar_dir / args.name,
        fields=fields,
        shard_size=1000,
    ) as writer:
        with mp.Pool(
            processes=args.num_jobs,
            initializer=init_worker,
            initargs=(hps,),
        ) as pool:
            for cut in tqdm(
                pool.imap_unordered(process_single_line, enumerate(files)),
                desc=f"Processing and saving to {args.shar_dir / args.name}",
                dynamic_ncols=True,
                smoothing=0,
                total=len(files),
            ):
                if isinstance(cut, str):
                    tqdm.write(cut)  # cut contains the error message in this case
                    continue
                writer.write(cut)
                num_processed += 1
                dataset_length += cut.duration
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
