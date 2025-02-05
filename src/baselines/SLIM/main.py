import torch
import numpy as np
import random
import argparse
from utils.training import StageOneTrainer, StageTwoTrainer
from utils.exp import seed_everything


def get_args():
    parser = argparse.ArgumentParser(description="Train SLIM model")

    parser.add_argument(
        "--stage",
        type=int,
        default=1,
        choices=[1, 2],
        help="Stage of training. 1 for stage 1 and 2 for stage 2",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="SLIM",
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
        "--data_path",
        type=str,
        default="/home/infres/amathur-23/DADA/datastets/InTheWild",
        help="Path to the training data",
    )

    parser.add_argument(
        "--cv_dir",
        type=str,
        default=None,
        help="Path to the root directory of the CommonVoice dataset",
    )

    parser.add_argument(
        "--ravdess_dir",
        type=str,
        default=None,
        help="Path to the root directory of the RAVDESS dataset",
    )

    parser.add_argument(
        "--model_save_dir",
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
        "--gpu_id",
        type=int,
        default=0,
        help="ID of the GPU to use for training",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for reproducibility",
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


if __name__ == "__main__":
    args = get_args()

    seed_everything(args.seed)

    if args.stage == 1:
        trainer = StageOneTrainer(args)
    elif args.stage == 2:
        trainer = StageTwoTrainer(args)
    trainer.train()
