#!/usr/bin/env python3
"""
Segment DAPS produced WAV files using energy-based VAD.

For each file, 100 ms frames are classified as speech/silence by RMS energy.
Leading silence at the start of each chunk is skipped.  The segmenter then
waits until --min-dur of speech is accumulated, cuts at the FIRST silence
frame found up to --max-dur (force-cuts if none), and pads each chunk with
--pad-ms of silence at both ends for natural transitions.
Segments shorter than --tail-min at the end of a file are discarded.
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from utils import get_time_string


def segment_audio(
    audio: np.ndarray,
    sr: int,
    min_dur: float,
    max_dur: float,
    hop_ms: int,
    tail_min: float,
    pad_ms: int,
) -> list[np.ndarray]:
    """
    Cut strategy: skip leading silence, then after min_dur elapses scan
    forward for the first silence frame up to max_dur.  If none found,
    force-cut at max_dur.  Each chunk is padded with pad_ms of silence
    at both ends for natural transitions.
    """
    hop_size    = int(hop_ms / 1000 * sr)
    pad_samples = int(pad_ms / 1000 * sr)
    n_frames    = max(1, len(audio) // hop_size)

    rms = np.sqrt(np.array([
        np.mean(audio[i * hop_size : (i + 1) * hop_size] ** 2)
        for i in range(n_frames)
    ], dtype=np.float64))

    peak       = rms.max() if rms.max() > 0 else 1.0
    threshold  = peak * 0.01          # –40 dB relative to file peak
    is_silence = rms < threshold

    min_samples  = int(min_dur  * sr)
    max_samples  = int(max_dur  * sr)
    tail_samples = int(tail_min * sr)

    chunks: list[np.ndarray] = []
    start = 0

    while start < len(audio):
        # Skip leading silence frames to find speech onset
        frame = start // hop_size
        while frame < n_frames and is_silence[frame]:
            frame += 1
        speech_start = frame * hop_size

        remaining = len(audio) - speech_start
        if remaining < tail_samples:
            break                    # discard short tail

        # Head: all frames between start and speech_start are silence (skipped above),
        # so it's safe to extend back up to pad_samples without hitting active frames.
        chunk_start = max(start, speech_start - pad_samples)

        if remaining <= max_samples:
            chunks.append(audio[chunk_start:])
            break

        # Forward search: first silence frame in [min_dur, max_dur] window
        search_s = (speech_start + min_samples) // hop_size
        search_e = min((speech_start + max_samples) // hop_size, n_frames - 1)

        cut_frame = next(
            (fi for fi in range(search_s, search_e + 1) if is_silence[fi]),
            None,
        )
        cut_sample = cut_frame * hop_size if cut_frame is not None else speech_start + max_samples

        # Tail: extend into silence after cut, stopping at the first active frame
        # within the pad window (so no active audio bleeds into the padding).
        pad_end_frame = min(n_frames, (cut_sample + pad_samples) // hop_size + 1)
        first_active  = next(
            (fi for fi in range(cut_sample // hop_size, pad_end_frame) if not is_silence[fi]),
            None,
        )
        chunk_end = first_active * hop_size if first_active is not None \
            else min(len(audio), cut_sample + pad_samples)

        chunks.append(audio[chunk_start:chunk_end])
        start = cut_sample

    return chunks


def process_file(
    wav_path: Path,
    out_dir: Path,
    min_dur: float,
    max_dur: float,
    hop_ms: int,
    tail_min: float,
    pad_ms: int,
) -> tuple[str, int, int]:
    """Load, segment, save.  Returns (filename, n_segments, total_samples)."""
    try:
        audio, sr = sf.read(str(wav_path), dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        chunks = segment_audio(audio, sr, min_dur, max_dur, hop_ms, tail_min, pad_ms)

        for idx, chunk in enumerate(chunks):
            out_path = out_dir / f"{wav_path.stem}_{idx:04d}.wav"
            sf.write(str(out_path), chunk, sr, subtype="PCM_16")

        total_samples = sum(len(c) for c in chunks)
        return wav_path.name, len(chunks), total_samples

    except Exception as e:
        return wav_path.name, -1, str(e)


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Segment DAPS produced WAV files using energy-based VAD.",
    )
    parser.add_argument("--src-dir",  type=Path,  default="/home/shahn/Datasets/daps/produced")
    parser.add_argument("--out-dir",  type=Path,  default="/home/shahn/Datasets/daps/produced_segmented")
    parser.add_argument("--min-dur",  type=float, default=3.0,
                        help="Minimum segment duration (seconds)")
    parser.add_argument("--max-dur",  type=float, default=10.0,
                        help="Maximum segment duration (seconds)")
    parser.add_argument("--hop-ms",   type=int,   default=100,
                        help="VAD frame hop size (milliseconds)")
    parser.add_argument("--tail-min", type=float, default=1.0,
                        help="Discard file-end tail shorter than this (seconds)")
    parser.add_argument("--pad-ms",   type=int,   default=200,
                        help="Silence padding added to both ends of each segment (milliseconds)")
    parser.add_argument("--num-workers", type=int, default=min(8, os.cpu_count()),
                        help="Number of parallel workers")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Print files that would be processed without writing")
    args = parser.parse_args()

    wav_files = sorted(args.src_dir.glob("*.wav"))
    print(f"Found {len(wav_files)} WAV files in {args.src_dir}")

    if args.dry_run:
        for f in wav_files:
            print(f"  {f.name}")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)

    total_segs    = 0
    total_samples = 0
    sr_ref        = sf.info(str(wav_files[0])).samplerate

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(
                process_file,
                f, args.out_dir, args.min_dur, args.max_dur,
                args.hop_ms, args.tail_min, args.pad_ms,
            ): f
            for f in wav_files
        }
        for future in tqdm(as_completed(futures), total=len(futures),
                           unit="file", dynamic_ncols=True, smoothing=0):
            name, n_segs, payload = future.result()
            if n_segs == -1:
                tqdm.write(f"  ERROR {name}: {payload}")
            else:
                tqdm.write(f"  {name}: {n_segs} segments")
                total_segs    += n_segs
                total_samples += payload

    print(f"\nDone.  {total_segs} segments, "
          f"total audio: {get_time_string(total_samples / sr_ref)}  "
          f"→ {args.out_dir}")


if __name__ == "__main__":
    main()