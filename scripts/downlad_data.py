"""Download le-wm datasets from HuggingFace.

Available datasets: tworooms, cube, pusht, reacher
Usage:
    python scripts/downlad_data.py                                    # download all to data/datasets/
    python scripts/downlad_data.py --datasets tworooms                # single dataset
    python scripts/downlad_data.py --datasets cube pusht              # multiple datasets
    python scripts/downlad_data.py --data-dir /mnt/storage/datasets   # custom directory
"""

import argparse
from huggingface_hub import snapshot_download

DATASETS = ["tworooms", "cube", "pusht", "reacher"]


def download(name: str, data_dir: str) -> None:
    repo_id = f"quentinll/lewm-{name}"
    local_dir = f"{data_dir}/lewm-{name}"
    print(f"Downloading {repo_id} → {local_dir}")
    path = snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=local_dir)
    print(f"  Done: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download le-wm datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=DATASETS,
        metavar="DATASET",
        help=f"Datasets to download (default: all). Choices: {DATASETS}",
    )
    parser.add_argument(
        "--data-dir",
        default="data/datasets",
        help="Root directory for downloaded datasets (default: data/datasets)",
    )
    args = parser.parse_args()

    for name in args.datasets:
        download(name, args.data_dir)


if __name__ == "__main__":
    main()
