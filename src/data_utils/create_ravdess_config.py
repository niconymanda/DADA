import numpy as np
import pandas as pd
import yaml
import argparse
import os
import glob


def get_args():
    parser = argparse.ArgumentParser(description="Create RAVDESS config")
    parser.add_argument(
        "--train_ratio", type=float, default=0.8, help="Ratio of training data"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.19, help="Ratio of validation data"
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.01, help="Ratio of test data"
    )
    parser.add_argument(
        "--root_dir", type=str, required=True, help="Root directory of the dataset"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="ravdess_config.yaml",
        help="Output config file path",
    )
    return parser.parse_args()


def create_ravdess_config(args):
    train_ratio, val_ratio, test_ratio = (
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
    )

    all_wav_files = glob.glob(os.path.join(args.root_dir, "Actor_*", "*.wav"))
    np.random.shuffle(all_wav_files)
    num_files = len(all_wav_files)

    train_files = all_wav_files[: int(num_files * train_ratio)]
    val_files = all_wav_files[
        int(num_files * train_ratio) : int(num_files * (train_ratio + val_ratio))
    ]
    test_files = all_wav_files[int(num_files * (train_ratio + val_ratio)) :]

    config = {"train": train_files, "val": val_files, "test": test_files}

    with open("ravdess_config.yaml", "w") as f:
        yaml.dump(config, f)


if __name__ == "__main__":
    args = get_args()
    create_ravdess_config(args)
