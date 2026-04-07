"""Download le-wm datasets from HuggingFace.

Available datasets: tworooms, cube, pusht, reacher
Usage:
    python scripts/downlad_data.py                        # download all
    python scripts/downlad_data.py --datasets tworooms    # single
    python scripts/downlad_data.py --datasets cube pusht  # multiple
"""

import argparse
from huggingface_hub import snapshot_download

DATASETS = ["tworooms", "cube", "pusht", "reacher"]


def download(name: str) -> None:
    repo_id = f"quentinll/lewm-{name}"
    local_dir = f"data/datasets/lewm-{name}"
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
    args = parser.parse_args()

    for name in args.datasets:
        download(name)


if __name__ == "__main__":
    main()
