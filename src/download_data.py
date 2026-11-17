"""Download le-wm datasets from HuggingFace.

Available datasets: tworooms, cube, pusht, reacher, antmaze
(antmaze is H-JEPA's Visual AntMaze training set, saved as lewm-antmaze.h5)
Usage:
    uv run python src/download_data.py                            # all datasets → data/datasets/
    uv run python src/download_data.py --datasets tworooms        # single dataset
    uv run python src/download_data.py --datasets cube pusht      # multiple datasets
    uv run python src/download_data.py --data-dir /path/datasets  # custom directory
"""

import argparse
import subprocess
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files

DATASETS = ["tworooms", "cube", "pusht", "reacher", "antmaze"]

ANTMAZE_REPO = "jepa-world-models/h-jepa"
ANTMAZE_FILE = "visual_antmaze_medium_explore_stitch_train.h5"


def download_antmaze(dst: Path) -> None:
    print(f"Downloading {ANTMAZE_REPO}/{ANTMAZE_FILE}...")
    # local_dir skips the ~/.cache copy; the file only appears there once complete.
    path = hf_hub_download(
        repo_id=ANTMAZE_REPO,
        filename=ANTMAZE_FILE,
        repo_type="dataset",
        local_dir=dst.parent,
    )
    Path(path).rename(dst)


def download(name: str, data_dir: str) -> None:
    repo_id = f"quentinll/lewm-{name}"
    dst = Path(data_dir) / f"lewm-{name}.h5"

    if dst.exists():
        print(f"Already exists: {dst}")
        return

    dst.parent.mkdir(parents=True, exist_ok=True)

    if name == "antmaze":
        download_antmaze(dst)
        print(f"  Done: {dst}")
        return

    repo_files = list_repo_files(repo_id, repo_type="dataset")
    tar_archives = [f for f in repo_files if f.endswith(".tar.zst")]
    # Some repos (e.g. lewm-pusht) publish a bare .h5.zst instead of a .tar.zst.
    bare_h5 = [f for f in repo_files if f.endswith(".h5.zst")]

    # A partial file at dst would pass the dst.exists() check on a rerun.
    tmp_dst = dst.with_suffix(dst.suffix + ".tmp")
    try:
        if tar_archives:
            archive = tar_archives[0]
            print(f"Downloading {repo_id}/{archive}...")
            archive_path = hf_hub_download(
                repo_id=repo_id, filename=archive, repo_type="dataset"
            )

            result = subprocess.run(
                ["tar", "--zstd", "-tf", archive_path],
                capture_output=True,
                text=True,
                check=True,
            )
            h5_files = [
                f.strip()
                for f in result.stdout.splitlines()
                if f.strip().endswith(".h5")
            ]
            if not h5_files:
                raise FileNotFoundError(f"No .h5 file inside archive {archive}")

            print(f"Extracting to {dst}...")
            with open(tmp_dst, "wb") as f:
                subprocess.run(
                    ["tar", "--zstd", "-xOf", archive_path, h5_files[0]],
                    stdout=f,
                    check=True,
                )
        elif bare_h5:
            archive = bare_h5[0]
            print(f"Downloading {repo_id}/{archive}...")
            archive_path = hf_hub_download(
                repo_id=repo_id, filename=archive, repo_type="dataset"
            )

            print(f"Decompressing to {dst}...")
            with open(tmp_dst, "wb") as f:
                # -f: the hf cache path is a symlink, which zstd refuses unless forced.
                subprocess.run(["zstd", "-dcf", archive_path], stdout=f, check=True)
        else:
            raise FileNotFoundError(
                f"No .tar.zst or .h5.zst archive found in {repo_id}"
            )

        tmp_dst.rename(dst)
    finally:
        tmp_dst.unlink(missing_ok=True)

    print(f"  Done: {dst}")


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
