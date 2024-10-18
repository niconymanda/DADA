import argparse
import numpy as np
import torch
import os
import pandas as pd

def load_data(data_path):
    data = pd.read_csv(data_path)
    data = data.dropna()
    number_quotes = 450
    data['label'] = data['label'].astype('int')
    label_counts = data['label'].value_counts()
    labels_to_keep = label_counts[label_counts >= number_quotes].index
    data = data[data['label'].isin(labels_to_keep)]
    print(f"Number of authors that have more then {number_quotes} quotes: {len(data['label'].unique())}")
    
    spoofed_data = data[data['type'] == 'spoof']
    data = data[data['type'] != 'spoof']

    return data, spoofed_data

def write_results_to_file(results, file_path, args):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    print(f"Writing results to file {file_path}")
    with open(file_path, 'a') as f: 
        f.write(f"Model: {args.model}, epochs {args.epochs}, batch_size {args. batch_size}, learning rate {args.learning_rate}\n")
        f.write("ABX accuracy\n")
        f.write(f"{results['abx_accuracy']}\n")

def get_args():
    parser = argparse.ArgumentParser(description='Train a text classification model')
    parser.add_argument('--data', type=str, default='~/DADA/Data/WikiQuotes.csv', help='Path to the input data file')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--model_name', type=str, default='FacebookAI/roberta-large', help='Model to use')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--layers_to_train', type=str, default="classifier", help='Layers to train: "classifier", "all", etc.')
    parser.add_argument('--early_stopping_patience', type=int, default=3, help='Patience for early stopping based on validation loss')
    parser.add_argument('--logging_step', type=int, default=50, help='Loggings step')
    
    return parser.parse_args()

def init_env(args):
    if args.seed is not None:
        seed_val = args.seed
        os.environ['PYTHONHASHSEED'] = str(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
    os.environ["CUDA_VISIBLE_DEVICES"]="2"

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
