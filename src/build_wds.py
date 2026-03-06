import os
import glob
import argparse
import webdataset as wds
from tqdm import tqdm


def compile_dataset(raw_dir: str, output_prefix: str, max_count: int):
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)

    print(f"[Compiler] Scanning {raw_dir} for raw trajectories...")
    all_files = os.listdir(raw_dir)
    basenames = sorted(
        list(set([f.split(".")[0] for f in all_files if f.startswith("traj_")]))
    )
    total_trajs = len(basenames)
    print(f"[Compiler] Found {total_trajs} unique trajectories.")

    if total_trajs == 0:
        print("[Compiler] Error: No files matching 'traj_*' found. Exiting.")
        return

    shard_pattern = f"{output_prefix}-%04d.tar"
    print(f"[Compiler] Packing into {shard_pattern} ({max_count} per shard)...")

    with wds.ShardWriter(shard_pattern, maxcount=max_count) as sink:
        for basename in tqdm(basenames, desc="Packing shards"):
            sample = {"__key__": basename}

            traj_files = sorted(glob.glob(os.path.join(raw_dir, f"{basename}.*")))

            for fpath in traj_files:

                ext = fpath.replace(os.path.join(raw_dir, f"{basename}."), "")

                with open(fpath, "rb") as stream:
                    sample[ext] = stream.read()

            sink.write(sample)

    print(
        f"\n[Compiler] Done. Shards successfully written to {os.path.dirname(output_prefix)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_dir",
        type=str,
        default="data/mixed_shapes_color_reduced_ratio/val/",
        help="Path to raw output directory",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="data/wds_mixed_shapes_color_reduced_ratio/val/platonic",
        help="Prefix for shards",
    )
    parser.add_argument(
        "--max_count",
        type=int,
        default=125,
        help="Number of trajectories per .tar shard",
    )

    args = parser.parse_args()
    compile_dataset(args.raw_dir, args.output_prefix, args.max_count)
