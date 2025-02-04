import os

import torch
import numpy as np
import random
import argparse
from mid_fusion.utils.training import MidFusionTrainer
import os
import json

def get_args():
    parser = argparse.ArgumentParser(description="Train Mid Fusion model")
    parser.add_argument('--model_name', type=str, default='mid_fusion', help='Name of the model to train')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate for optimizer')
    parser.add_argument('--data_path', type=str, default = '/data/amathur-23/DADA/InTheWild/release_in_the_wild/', help='Path to the training data')
    parser.add_argument('--model_save_path', type=str, default='./models', help='Path to save the trained model')
    parser.add_argument('--log_interval', type=int, default=1, help='Interval for logging training status')
    parser.add_argument('--dataset_config', type=str, default='/home/infres/iivanova-23/DADA/src/mid_fusion/configs/inthewild_full.yaml', help='Path to the dataset configuration file')
    parser.add_argument('--log_dir', type=str, default='./runs', help='Path to save logs')
    parser.add_argument('--max_duration', type=int, default=4, help='Maximum duration of audio files')
    parser.add_argument('--sampling_rate', type=int, default=16000, help='Sampling rate of audio files')
    parser.add_argument('--early_stopping_patience', type=int, default=3, help='Number of epochs to wait before early stopping')
    parser.add_argument('--early_stopping_threshold', type=float, default=0.01, help='Threshold for early stopping')
    # parser.add_argument('--loss_fn', type=str, default='triplet_cosine', help='Loss function to use for training', choices=['triplet', 'triplet_cosine', 'ada_triplet'])
    parser.add_argument('--gpu_id', type=int, default=1, help='ID of the GPU to use for training')
    # parser.add_argument('--margin', type=float, default=0.5, help='Margin for triplet loss')
    # parser.add_argument('--at_lambda', type=float, default=0.5, help='Lambda for AdaTriplet loss')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--save_visualisations', type=bool, default=False, help='Save visualizations of embeddings')
    parser.add_argument('--text_model_name', type=str, default='google-t5/t5-large', help='Name of the text model to use')
    parser.add_argument('--text_model_path', type=str, default=None, help='Path to the text model')
    parser.add_argument('--speech_model_name', type=str, default=None, help='Name of the speech model to use')
    parser.add_argument('--speech_model_path', type=str, default='/data/amathur-23/DADA/models/SpeechEmbedder/triplet_cosine_epsP35_51v2/best_model.pth', help='Path to the speech model')
    parser.add_argument('--load_checkpoint', type=str, default=None, help='Path to the checkpoint to load')
    parser.add_argument('--mlp_layers', type=int, default=2, help='Number of layers in MLP head')
    parser.add_argument('--hidden_layers', type=lambda s: [int(item) for item in s.split(',')], default="-1", help='List of hidden layer sizes for the model')
    # /home/infres/iivanova-23/DADA/iivanova-23/models/google-t5/final
    return parser.parse_args()

def load_config_text_model(args):
    if args.text_model_path is not None:
        config_path = args.text_model_path.replace('final', 'model_config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        args.text_model_name = config['architecture']['args']['model_name']
        args.mlp_layers = config['architecture']['args']['num_mlp_layers']
        args.hidden_layers = config['architecture']['args']['hidden_layers mean pool']
    return args

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
    args = load_config_text_model(args)
    print(f"Using GPU: {args.gpu_id}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    # Set huggingface cache
    os.environ['HF_HOME'] = '/data/iivanova-23/cache/'

    seed_everything(args.seed)
    # args.device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    trainer = MidFusionTrainer(args)
    trainer.train()
    # trainer.validate()
