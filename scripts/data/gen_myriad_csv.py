from pathlib import Path
import argparse
import csv


T60_S = {
    "SAL": 2.1,
    "AIL": 0.5,
}


def iter_entries(base_dir: Path):
    for subdir_name, t60 in T60_S.items():
        for wav_path in sorted((base_dir / subdir_name).rglob("*.wav")):
            rel_path = wav_path.relative_to(base_dir)
            yield str(rel_path), 1, t60


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default="/home/shahn/Documents/trainer/filelists/rir/myriad.csv",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default="/home/shahn/Datasets/MYRiAD/MYRiAD_V2_econ/16khz",
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
