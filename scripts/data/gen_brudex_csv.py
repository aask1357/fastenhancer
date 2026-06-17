from pathlib import Path
import argparse
import csv
import soundfile as sf


T60_S = {
    "rev_low": 0.31,
    "rev_medium": 0.51,
    "rev_high": 1.3,
}


def iter_entries(base_dir: Path):
    for subdir_name, t60 in T60_S.items():
        for wav_path in sorted((base_dir / subdir_name).glob("*.wav")):
            num_channels = sf.info(wav_path).channels
            rel_path = wav_path.relative_to(base_dir)
            for ch in range(1, num_channels + 1):
                yield str(rel_path), ch, t60


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default="/home/shahn/Documents/trainer/filelists/rir/brudex.csv",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default="/home/shahn/Datasets/brudex/rir_16khz",
    )
    return parser.parse_args()


def main():
    args = get_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows = list(iter_entries(args.base_dir))
    print(f"Total entries: {len(rows)}")

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "channel", "t60"])
        writer.writerows(rows)

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
