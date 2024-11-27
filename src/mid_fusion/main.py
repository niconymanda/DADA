import os

import torch
import numpy as np
import random
import argparse
from mid_fusion.utils.training import MidFusionTrainer
import os

def get_args():
    parser = argparse.ArgumentParser(description="Train Mid Fusion model")
    parser.add_argument('--model_name', type=str, default='mid_fusion', help='Name of the model to train')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate for optimizer')
    parser.add_argument('--data_path', type=str, default = '/home/infres/amathur-23/DADA/datastets/InTheWild', help='Path to the training data')
    parser.add_argument('--model_save_path', type=str, default='./models', help='Path to save the trained model')
    parser.add_argument('--log_interval', type=int, default=1, help='Interval for logging training status')
    parser.add_argument('--dataset_config', type=str, default='configs/data/inthewild.yaml', help='Path to the dataset configuration file')
    parser.add_argument('--log_dir', type=str, default='./runs', help='Path to save logs')
    parser.add_argument('--max_duration', type=int, default=4, help='Maximum duration of audio files')
    parser.add_argument('--sampling_rate', type=int, default=16000, help='Sampling rate of audio files')
    parser.add_argument('--early_stopping_patience', type=int, default=3, help='Number of epochs to wait before early stopping')
    parser.add_argument('--early_stopping_threshold', type=float, default=0.01, help='Threshold for early stopping')
    # parser.add_argument('--loss_fn', type=str, default='triplet_cosine', help='Loss function to use for training', choices=['triplet', 'triplet_cosine', 'ada_triplet'])
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of the GPU to use for training')
    # parser.add_argument('--margin', type=float, default=0.5, help='Margin for triplet loss')
    # parser.add_argument('--at_lambda', type=float, default=0.5, help='Lambda for AdaTriplet loss')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--save_visualisations', type=bool, default=False, help='Save visualizations of embeddings')
    parser.add_argument('--text_model_name', type=str, default='microsoft/deberta-v3-large', help='Name of the text model to use')
    parser.add_argument('--text_model_path', type=str, default=None, help='Path to the text model')
    parser.add_argument('--speech_model_name', type=str, default=None, help='Name of the speech model to use')
    parser.add_argument('--speech_model_path', type=str, default=None, help='Path to the speech model')
    parser.add_argument('--load_checkpoint', type=str, default=None, help='Path to the checkpoint to load')
    return parser.parse_args()

def seed_everything(seed: int):
    """fix the seed for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' # 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    args = get_args()
    print(f"Using GPU: {args.gpu_id}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    # Set huggingface cache
    os.environ['HF_HOME'] = '/data/amathur-23/DADA/hf_cache'

    seed_everything(args.seed)
    # args.device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    trainer = MidFusionTrainer(args)
    trainer.train()
    # trainer.validate()
