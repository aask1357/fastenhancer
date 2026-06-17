#!/usr/bin/env python3
"""
Preprocess Expresso dataset: segment long WAV files using VAD annotations.

Handles:
  - conversational/**/*.wav  (stereo): merge both channels to mono,
                                        union VAD timelines from channel1+channel2
  - read/**/longform/*.wav   (mono):   use single-channel VAD timeline

Output is written to audio_48khz/segmented/, mirroring the source structure.

Long VAD segments (> --max-dur) are further split using one of two strategies:
  fixed  — fixed-length chunks; tail kept if >= --tail-min, else discarded
  vad    — split at energy-based silence near [--min-dur, --max-dur] boundary
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm


# ---------------------------------------------------------------------------
# VAD parsing
# ---------------------------------------------------------------------------

def parse_vad_file(vad_path: Path) -> dict[str, list[tuple[float, float]]]:
    vad: dict[str, list[tuple[float, float]]] = {}
    with open(vad_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, segments_str = line.split("\t", 1)
            segs = [
                (float(s), float(e))
                for s, e in re.findall(r"\(([0-9.]+),\s*([0-9.]+)\)", segments_str)
            ]
            vad[key] = segs
    return vad


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Union-merge overlapping/adjacent intervals."""
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = list([intervals[0]])
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


# ---------------------------------------------------------------------------
# Sub-segmentation strategies
# ---------------------------------------------------------------------------

def split_fixed(
    audio: np.ndarray,
    sr: int,
    max_dur: float,
    tail_min: float,
) -> list[np.ndarray]:
    """Cut into fixed max_dur chunks; discard tail shorter than tail_min."""
    max_samples      = int(max_dur  * sr)
    tail_min_samples = int(tail_min * sr)
    chunks = []
    offset = 0
    while offset < len(audio):
        chunk = audio[offset : offset + max_samples]
        if len(chunk) < tail_min_samples:
            break
        chunks.append(chunk)
        offset += max_samples
    return chunks


def split_vad_based(
    audio: np.ndarray,
    sr: int,
    max_dur: float,
    min_dur: float,
    frame_ms: int = 20,
) -> list[np.ndarray]:
    """
    Split at energy-based silence, trying to stay within [min_dur, max_dur].

    Algorithm:
      1. Compute per-frame RMS energy (20 ms frames).
      2. Mark frames as silence if RMS < 1% of the segment's peak RMS.
      3. From the current position, search [min_dur, max_dur] for the last
         silence frame to use as a cut point.
      4. If none found, force-cut at max_dur.
    """
    frame_size = int(frame_ms / 1000 * sr)
    n_frames   = max(1, len(audio) // frame_size)
    rms = np.array([
        np.sqrt(np.mean(audio[i * frame_size : (i + 1) * frame_size] ** 2))
        for i in range(n_frames)
    ], dtype=np.float32)

    peak      = rms.max() if rms.max() > 0 else 1.0
    threshold = peak * 0.01   # –40 dB relative to segment peak
    is_silence = rms < threshold

    max_samples = int(max_dur * sr)
    min_samples = int(min_dur * sr)

    chunks: list[np.ndarray] = []
    start = 0

    while start < len(audio):
        remaining = len(audio) - start
        if remaining <= max_samples:
            chunk = audio[start:]
            if len(chunk) >= int(1.0 * sr):   # keep if ≥ 1 s
                chunks.append(chunk)
            break

        # Search for a silence frame in the [min_dur, max_dur] window
        search_s_frame = (start + min_samples) // frame_size
        search_e_frame = min((start + max_samples) // frame_size, n_frames - 1)

        cut_frame = None
        for fi in range(search_e_frame, search_s_frame - 1, -1):
            if is_silence[fi]:
                cut_frame = fi
                break

        if cut_frame is not None:
            cut_sample = cut_frame * frame_size
        else:
            cut_sample = start + max_samples   # force cut

        chunk = audio[start:cut_sample]
        if len(chunk) > 0:
            chunks.append(chunk)
        start = cut_sample

    return chunks


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def process_file(
    wav_path: Path,
    vad_segments: list[tuple[float, float]],
    output_dir: Path,
    strategy: str,
    max_dur: float,
    min_dur: float,
    tail_min: float,
) -> int:
    audio, sr = sf.read(str(wav_path), dtype="float32")

    # Stereo → mono
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    total_dur = len(audio) / sr

    # Clip VAD segments to actual audio length
    segments = [
        (max(0.0, s), min(e, total_dur))
        for s, e in vad_segments
        if e > 0 and s < total_dur
    ]
    segments = [(s, e) for s, e in segments if e - s >= 0.1]

    if not segments:
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    seg_idx = 0
    for seg_start, seg_end in segments:
        s_sample = int(seg_start * sr)
        e_sample = int(seg_end   * sr)
        chunk    = audio[s_sample:e_sample]
        dur      = len(chunk) / sr

        if dur <= max_dur:
            out_path = output_dir / f"{wav_path.stem}_{seg_idx:04d}.wav"
            sf.write(str(out_path), chunk, sr, subtype="PCM_16")
            seg_idx += 1
        else:
            if strategy == "fixed":
                sub_chunks = split_fixed(chunk, sr, max_dur, tail_min)
            else:
                sub_chunks = split_vad_based(chunk, sr, max_dur, min_dur)
            for sub in sub_chunks:
                out_path = output_dir / f"{wav_path.stem}_{seg_idx:04d}.wav"
                sf.write(str(out_path), sub, sr, subtype="PCM_16")
                seg_idx += 1

    return seg_idx


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def create_read_symlinks(output_dir: Path, dataset_root: Path) -> None:
    """Symlink unsegmented read/base dirs into output_dir, mirroring source structure.

    - style dirs with only base/   → symlink the whole style dir
    - style dirs with base/ + longform/ → real dir, symlink only base/
    - style dirs with only longform/ → skip (segmentation script handles these)
    """
    src_read = dataset_root / "read"
    dst_read = output_dir / "read"

    for speaker_dir in sorted(src_read.iterdir()):
        if not speaker_dir.is_dir():
            continue
        speaker = speaker_dir.name
        dst_speaker = dst_read / speaker
        dst_speaker.mkdir(parents=True, exist_ok=True)

        for style_dir in sorted(speaker_dir.iterdir()):
            if not style_dir.is_dir():
                continue
            style      = style_dir.name
            has_base     = (style_dir / "base").is_dir()
            has_longform = (style_dir / "longform").is_dir()

            if has_base and not has_longform:
                link = dst_speaker / style
                if not link.exists() and not link.is_symlink():
                    link.symlink_to(style_dir)
                    print(f"  linked  read/{speaker}/{style}")
            elif has_base and has_longform:
                (dst_speaker / style).mkdir(exist_ok=True)
                link = dst_speaker / style / "base"
                if not link.exists() and not link.is_symlink():
                    link.symlink_to(style_dir / "base")
                    print(f"  linked  read/{speaker}/{style}/base")
            # has_longform only → skip


def collect_files(
    vad: dict,
    output_dir: Path,
    dataset_root: Path,
) -> list[tuple[Path, list[tuple[float, float]], Path]]:
    files = []

    # Conversational: stereo, merge channel1 + channel2 VAD
    for wav_path in sorted(dataset_root.glob("conversational/**/*.wav")):
        stem = wav_path.stem
        segs1 = vad.get(f"{stem}/channel1", [])
        segs2 = vad.get(f"{stem}/channel2", [])
        merged = merge_intervals(segs1 + segs2)
        if not merged:
            print(f"WARNING: no VAD entry for {stem}", file=sys.stderr)
            continue
        rel     = wav_path.relative_to(dataset_root)
        out_dir = output_dir / rel.parent
        files.append((wav_path, merged, out_dir))

    # Longform: already mono, single VAD timeline
    for wav_path in sorted(dataset_root.glob("read/**/longform/*.wav")):
        stem = wav_path.stem
        segs = vad.get(stem, [])
        if not segs:
            print(f"WARNING: no VAD entry for {stem}", file=sys.stderr)
            continue
        rel     = wav_path.relative_to(dataset_root)
        out_dir = output_dir / rel.parent
        files.append((wav_path, segs, out_dir))

    return files


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Segment Expresso long WAV files using VAD annotations."
    )
    parser.add_argument(
        "--strategy", choices=["fixed", "vad"], default="vad",
        help="Sub-segmentation strategy for chunks longer than --max-dur (default: vad)",
    )
    parser.add_argument(
        "--max-dur", type=float, default=10.0,
        help=f"Maximum segment duration in seconds)",
    )
    parser.add_argument(
        "--min-dur", type=float, default=5.0,
        help=f"Minimum duration before a split point is searched (vad strategy)",
    )
    parser.add_argument(
        "--tail-min", type=float, default=1.0,
        help=f"Minimum tail length to keep (fixed strategy)",
    )
    parser.add_argument(
        "--dataset-root", type=Path, default="/home/shahn/Datasets/expresso/audio_48khz",
        help="Root directory of the Expresso audio_48khz dataset",
    )
    parser.add_argument(
        "--vad-file", type=Path, default="/home/shahn/Datasets/expresso/VAD_segments.txt",
        help="Path to VAD_segments.txt",
    )
    parser.add_argument(
        "--output-dir", type=Path, default="/home/shahn/Datasets/expresso/audio_48khz/segmented",
        help="Root output directory",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be done without writing any files",
    )
    args = parser.parse_args()

    vad   = parse_vad_file(args.vad_file)
    files = collect_files(vad, args.output_dir, args.dataset_root)

    print(
        f"Found {len(files)} files to process  "
        f"(strategy={args.strategy}, max={args.max_dur}s"
        + (f", min={args.min_dur}s" if args.strategy == "vad" else f", tail_min={args.tail_min}s")
        + ")"
    )
    if args.dry_run:
        for wav_path, segs, out_dir in files:
            print(f"  {wav_path.relative_to(args.dataset_root)}  →  {out_dir}  ({len(segs)} VAD segs)")
        return

    print("\nCreating read/ symlinks...")
    create_read_symlinks(args.output_dir, args.dataset_root)

    total_segs = 0
    for wav_path, segs, out_dir in tqdm(files, unit="file"):
        n = process_file(
            wav_path, segs, out_dir,
            args.strategy, args.max_dur, args.min_dur, args.tail_min,
        )
        tqdm.write(f"  {wav_path.name}: {n} segments")
        total_segs += n

    print(f"\nDone. {total_segs} total segments written to {args.output_dir}")


if __name__ == "__main__":
    main()
