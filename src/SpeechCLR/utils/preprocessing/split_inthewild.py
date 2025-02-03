import numpy as np
import pandas as pd
import yaml
import argparse
import os


def split_dataset(
    dataset,
    train_authors,
    val_authors,
    max_samples={"train": -1, "val": -1, "test": -1},
):
    authors = (
        dataset["speaker"].value_counts().index.tolist()
    )  # Authors in decreasing order of number of samples

    train_authors_set = set(authors[:train_authors])
    val_authors_set = set(authors[train_authors : train_authors + val_authors])
    test_authors_set = set(authors[train_authors + val_authors :])

    train_data = dataset[dataset["speaker"].isin(train_authors_set)][
        : max_samples["train"]
    ]
    val_data = dataset[dataset["speaker"].isin(val_authors_set)][: max_samples["val"]]
    test_data = dataset[dataset["speaker"].isin(test_authors_set)][
        : max_samples["test"]
    ]

    train_files = train_data["file"].values
    val_files = val_data["file"].values
    test_files = test_data["file"].values

    return train_files, val_files, test_files


def get_args():
    parser = argparse.ArgumentParser(
        description="Split dataset into train, val, and test sets"
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the input dataset"
    )
    parser.add_argument(
        "--train_authors",
        type=int,
        default=4,
        required=False,
        help="Number of authors in the training set",
    )
    parser.add_argument(
        "--val_authors",
        type=int,
        default=3,
        required=False,
        help="Number of authors in the validation set",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        default="configs",
        help="Path to the output directory",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        required=True,
        default="inthewild.yaml",
        help="Name of the output file",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        required=False,
        default=0.8,
        help="Ratio of training data",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        required=False,
        default=200,
        help="Maximum number of training samples",
    )

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    max_samples = {
        "train": args.max_train_samples,
        "val": (
            int(args.max_train_samples * (1 - args.train_ratio))
            if args.max_train_samples > 0
            else -1
        ),
        "test": -1,
    }

    dataset = pd.read_csv(os.path.join(args.data_path, "meta.csv"))
    train_data, val_data, test_data = split_dataset(
        dataset, args.train_authors, args.val_authors
    )

    print(train_data[:5])

    output_data = {
        "train": list(train_data),
        "val": list(val_data),
        "test": list(test_data),
    }

    output_path = os.path.join(args.output_dir, args.output_name)
    os.makedirs(args.output_dir, exist_ok=True)

    with open(output_path, "w") as outfile:
        yaml.dump(output_data, outfile)


if __name__ == "__main__":
    main()
