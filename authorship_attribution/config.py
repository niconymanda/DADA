import argparse
import numpy as np
import torch
import os
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser(description='Train a text classification model')
    parser.add_argument('--data', type=str, default='~/DADA/Data/WikiQuotes.csv', help='Path to the input data file')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--model_name', type=str, default='FacebookAI/roberta-base', help='Model to use')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--layers_to_train', type=str, default="classifier", help='Layers to train: "classifier", "all", etc.')
    parser.add_argument('--early_stopping_patience', type=int, default=2, help='Patience for early stopping based on validation loss')
    parser.add_argument('--logging_step', type=int, default=50, help='Loggings step')
    parser.add_argument('--min_quotes_per_author', type=int, default=350, help='Min number of quotes per author')
    # 350 quotes=5 authors, 450 quotes=3authors
    return parser.parse_args()

def load_data(args):
    data = pd.read_csv(args.data)
    data = data.dropna()
    number_quotes = args.min_quotes_per_author
    data['label'] = data['label'].astype('int')
    label_counts = data['label'].value_counts()
    labels_to_keep = label_counts[label_counts >= number_quotes].index
    data = data[data['label'].isin(labels_to_keep)]
    num_authors = len(data['label'].unique())
    print(f"Number of authors that have more then {number_quotes} quotes: {num_authors}")
    
    spoofed_data = data[data['type'] == 'spoof']
    data = data[data['type'] != 'spoof'] 
    author_id_map = data[['label', 'author_name']].drop_duplicates().set_index('label').to_dict()['author_name']
    
    return data, spoofed_data, author_id_map

def write_results_to_file(results, file_path, args):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    print(f"Writing results to file {file_path}")
    with open(file_path, 'a') as f: 
        f.write(f"Model: {args.model}, epochs {args.epochs}, batch_size {args. batch_size}, learning rate {args.learning_rate}\n")
        f.write("ABX accuracy\n")
        f.write(f"{results['abx_accuracy']}\n")


def init_env(args):
    if args.seed is not None:
        seed_val = args.seed
        os.environ['PYTHONHASHSEED'] = str(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_checkpoint(model, optimizer, epoch, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    torch.save(checkpoint, path)
    print(f"Model saved to {path}")
    
def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    print(f"Loading model from {path}")
    print(checkpoint.keys())    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Model loaded from {path}")
    return model, optimizer, checkpoint['epoch']
