import torch
import numpy as np
import argparse
from utils.training import SpeechCLRTrainer, SpeechCLRTrainerVanilla
import os

def get_args():
    parser = argparse.ArgumentParser(description="Train SpeechCLR model")
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate for optimizer')
    parser.add_argument('--data_path', type=str, default = '/home/infres/amathur-23/DADA/datastets/InTheWild', help='Path to the training data')
    parser.add_argument('--model_save_path', type=str, default='./models', help='Path to save the trained model')
    parser.add_argument('--log_interval', type=int, default=10, help='Interval for logging training status')
    parser.add_argument('--dataset_config', type=str, default='configs/inthewild.yaml', help='Path to the dataset configuration file')
    parser.add_argument('--output_dir', type=str, default='./output', help='Path to save the output files')
    parser.add_argument('--max_duration', type=int, default=4, help='Maximum duration of audio files')
    parser.add_argument('--sampling_rate', type=int, default=16000, help='Sampling rate of audio files')
    parser.add_argument('--early_stopping_patience', type=int, default=3, help='Number of epochs to wait before early stopping')
    parser.add_argument('--early_stopping_threshold', type=float, default=0.01, help='Threshold for early stopping')
    parser.add_argument('--triplet_margin', type=float, default=1.0, help='Margin for triplet loss')
    parser.add_argument('--loss_fn', type=str, default='triplet_cosine', help='Loss function to use for training', choices=['triplet', 'triplet_cosine'])
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of the GPU to use for training')

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    # args.device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    trainer = SpeechCLRTrainerVanilla(args)
    trainer.train()
