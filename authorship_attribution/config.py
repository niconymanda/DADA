import argparse
import numpy as np
import torch
import os
import pandas as pd


def get_args():
    """
    Parses command-line arguments for training a the authorship attribution classification model.
    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    
    parser = argparse.ArgumentParser(description='Train a text classification model')
    parser.add_argument('--data', type=str, default='~/DADA/Data/WikiQuotes.csv', help='Path to the input data file')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs to train for')
    parser.add_argument('--epochs_classification', type=int, default=2, help='Number of epochs to train the classifcation head for')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--model_name', type=str, default='google-bert/bert-large-uncased', help='Model to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--layers_to_train', type=str, default="classifier", help='Layers to train: "classifier", "all", etc.')
    parser.add_argument('--early_stopping_patience', type=int, default=5, help='Patience for early stopping based on validation loss')
    parser.add_argument('--logging_step', type=int, default=10, help='Loggings step')
    parser.add_argument('--min_quotes_per_author', type=int, default=450, help='Min number of quotes per author')
    # 350 quotes=5 authors, 450 quotes=3 authors
    return parser.parse_args()

def load_data(args):
    """
    Loads and processes the dataset.
    Args:
        args (argparse.Namespace): The arguments containing the following attributes:
            - data (str): Path to the CSV file containing the dataset.
            - min_quotes_per_author (int): Minimum number of quotes required per author.
    Returns:
        tuple: A tuple containing:
            - data (pd.DataFrame): The processed dataset excluding spoofed data.
            - spoofed_data (pd.DataFrame): The dataset containing only spoofed data.
            - author_id_map (dict): A dictionary mapping new author labels to author names. E.g. {0: 'Donald Trump', 1: 'Barack Obama', 2: 'John F. Kennedy', ...}
    """
    
    data = pd.read_csv(args.data)
    data = data.dropna()
    number_quotes = args.min_quotes_per_author
    data['label'] = data['label'].astype('int')
    label_counts = data['label'].value_counts()
    # Keep only authors with more than number_quotes quotes
    labels_to_keep = label_counts[label_counts >= number_quotes].index
    data = data[data['label'].isin(labels_to_keep)]
    num_authors = len(data['label'].unique())
    print(f"Number of authors that have more then {number_quotes} quotes: {num_authors}")
    
    # Map author labels to new labels starting from 0
    author_id_map = data[['label', 'author_name']].drop_duplicates().set_index('label').to_dict()['author_name']
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(labels_to_keep)}
    data['label'] = data['label'].map(label_mapping)
    author_id_map = {new_label: author_id_map[old_label] for old_label, new_label in label_mapping.items()}
    
    # Split spoofed data
    spoofed_data = data[data['type'] == 'spoof']
    data = data[data['type'] != 'spoof'] 
    return data[:500], spoofed_data, author_id_map

def write_results_to_file(results, file_path, args):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    print(f"Writing results to file {file_path}")
    with open(file_path, 'a') as f: 
        f.write(f"Model: {args.model}, epochs {args.epochs}, batch_size {args. batch_size}, learning rate {args.learning_rate}\n")
        f.write("ABX accuracy\n")
        f.write(f"{results['abx_accuracy']}\n")


def init_env(args):
    """
    Initialize the environment for reproducibility and set CUDA device.
    Parameters:
    args (Namespace): A namespace object containing the following attributes:
        - seed (int, optional): The seed value for random number generators. If provided, it sets the seed for 
          Python's hash seed, NumPy, and PyTorch to ensure reproducibility.
    Environment Variables:
    - PYTHONHASHSEED: Set to the provided seed value if args.seed is not None.
    - CUDA_VISIBLE_DEVICES: Set to "1" to specify the CUDA device to be used.
    """
    
    if args.seed is not None:
        seed_val = args.seed
        os.environ['PYTHONHASHSEED'] = str(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
    os.environ["CUDA_VISIBLE_DEVICES"]="1"

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
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Model loaded from {path}")
    return model, optimizer, checkpoint['epoch']
