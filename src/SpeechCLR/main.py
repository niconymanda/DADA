import torch
import numpy as np
import random
import argparse
from utils.training import SpeechCLRTrainerVanilla
import os
import yaml


def get_args_():
    parser = argparse.ArgumentParser(description="Train SpeechCLR model")
    parser.add_argument(
        "--model_name",
        type=str,
        default="SpeechEmbedder",
        help="Name of the model to train",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=2, help="Number of epochs to train"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-5, help="Learning rate for optimizer"
    )

    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["inthewild"],
        help="List of datasets to use for training",
        choices=[
            "commonvoice",
            "voxceleb2",
            "inthewild",
            "asvspoof19",
            "ravdess",
        ],
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/infres/amathur-23/DADA/datastets/InTheWild",
        help="Path to the training data",
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="./models",
        help="Path to save the trained model",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="Interval for logging training status",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="configs/inthewild.yaml",
        help="Path to the dataset configuration file",
    )
    parser.add_argument(
        "--log_dir", type=str, default="./runs", help="Path to save logs"
    )
    parser.add_argument(
        "--max_duration", type=int, default=4, help="Maximum duration of audio files"
    )
    parser.add_argument(
        "--sampling_rate", type=int, default=16000, help="Sampling rate of audio files"
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=3,
        help="Number of epochs to wait before early stopping",
    )
    parser.add_argument(
        "--early_stopping_threshold",
        type=float,
        default=0.01,
        help="Threshold for early stopping",
    )

    parser.add_argument(
        "--loss_fn",
        type=str,
        default="triplet_cosine",
        help="Loss function to use for training",
        choices=["triplet", "triplet_cosine", "ada_triplet", "squared_similarity"],
    )
    parser.add_argument(
        "--gpu_id", type=int, default=0, help="ID of the GPU to use for training"
    )
    parser.add_argument(
        "--margin", type=float, default=0.5, help="Margin for triplet loss"
    )
    parser.add_argument(
        "--at_lambda", type=float, default=0.5, help="Lambda for AdaTriplet loss"
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument(
        "--save_visualisations",
        type=bool,
        default=False,
        help="Save visualizations of embeddings",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="step",
        help="Learning rate scheduler to use",
        choices=["plateau", "step", "cosine"],
    )
    parser.add_argument(
        "--early_stopping_metric",
        type=str,
        default="loss",
        help="Metric to use for early stopping",
        choices=["loss", "accuracy"],
    )
    parser.add_argument(
        "--load_path", type=str, default=None, help="Path to the checkpoint to load"
    )
    parser.add_argument(
        "--load_checkpoint",
        type=str,
        default="best",
        choices=["best", "latest"],
        help="Checkpoint to load",
    )

    return parser.parse_args()

def get_args():
    parser = argparse.ArgumentParser(description="Train SpeechCLR model")
    parser.add_argument(
        "--config_file",
        type=str,
        default="configs/experiments/default.yaml",
        help="Name of the configuration file",
    )
    args = parser.parse_args()
    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)
    return argparse.Namespace(**config)

def seed_everything(seed: int):
    """fix the seed for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  #
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    args = get_args()
    print(f"Using GPU: {args.gpu_id}")

    # Set huggingface cache
    os.environ["HF_HOME"] = "/data/amathur-23/DADA/hf_cache"

    seed_everything(args.seed)
    args.device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    trainer = SpeechCLRTrainerVanilla(args)
    trainer.train()
    # trainer.validate()
