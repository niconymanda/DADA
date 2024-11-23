import argparse
import numpy as np
import torch
import os
import pandas as pd
import json
import torch.nn.functional as F
import random


def get_args():
    """
    Parses command-line arguments for training a the authorship attribution classification model.
    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    
    parser = argparse.ArgumentParser(description='Train a text classification model')
    parser.add_argument('--data', type=str, default='~/DADA/Data/WikiQuotes.csv', help='Path to the input data file')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train for')
    parser.add_argument('--epochs_classification', type=int, default=8, help='Number of epochs to train the classifcation head for')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--learning_rate_classification', type=float, default=1e-4, help='Learning rate classification')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight_decay')
    parser.add_argument('--model_name', type=str, default='facebook/xlm-roberta-xxl', help='Model to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--layers_to_train', type=str, default="classifier", help='Layers to train: "classifier", "all", etc.')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Patience for early stopping based on validation loss')
    parser.add_argument('--logging_step', type=int, default=10, help='Loggings step')
    parser.add_argument('--min_quotes_per_author', type=int, default=450, help='Min number of quotes per author')
    parser.add_argument('--distance_function', type=str, default='l2', help='Distance function for triplet loss (l2 or cosine)')
    parser.add_argument('--loss_function', type=str, default='triplet', help='Loss function for training [triplet, contrastive, ada_triplet, hinge, cos2]')
    parser.add_argument('--margin', type=float, default=0.5, help='Margin for triplet loss')
    parser.add_argument('--lr_scheduler', type=str, default='cosine', help='Learning rate scheduler')
    
    # 350 quotes=5 authors, 450 quotes=3 authors,
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
    return data, spoofed_data, author_id_map

def write_results_to_file(results, file_path, args):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    print(f"Writing results to file {file_path}")
    with open(file_path, 'a') as f: 
        f.write(f"Model: {args.model_name}, epochs {args.epochs}, batch_size {args. batch_size}, learning rate {args.learning_rate}\n")
        f.write("ABX accuracy, Accuracy, Precision, Recall, F1, AUC \n")
        f.write(f"{results['abx_accuracy']:.4f}, {results['accuracy']:.4f}, {results['precision']:.4f}, {results['recall']:.4f}, {results['f1_score']:.4f}, {results['auc']:.4f}\n")



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
        random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    os.environ['HF_HOME'] = '/data/iivanova-23/cache/'

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

def save_model_config(
    args,
    output_path: str = "model_config.json"
):
    config_new = {
        "min_quotes_per_author": args.min_quotes_per_author,
        "architecture":{
            "name": "AuthorshipLLM",
            "args": {
                "model_name": args.model_name,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "early_stopping_patience": args.early_stopping_patience,
                "logging_step": args.logging_step,
                "n_gpu": 1
                }
        },
        "architecture_classifier":{
            "name": "AuthorshipClassificationLLM",
            "args": {
                "model_name": "Liner head with softmax",
                "epochs": args.epochs_classification,
                "learning_rate": args.learning_rate_classification,
                "weight_decay": args.weight_decay,
                "early_stopping_patience": args.early_stopping_patience,
                "logging_step": args.logging_step,
                "n_gpu": 1
                }
        },
        "loss_function": {
            "name": args.loss_function,
            "args": {
                "margin": args.margin,
                "distance_function": args.distance_function,
                "reduction": "mean"
            }
        },
        "data_loader": {
          "type": "AuthorTripletLossDataset",         # selecting data loader
          "args":{
            "data": args.data,             # dataset path
            "batch_size": args.batch_size,                # batch size
            "shuffle": True,                 # shuffle training data before splitting
            "validation_split": 0.2,         # size of validation dataset. float(portion) or int(number of samples)
            "num_workers": 0,                # number of cpu processes to be used for data loading
          }
        },
        "optimizer_LLM_model": {
          "type": "AdamW",
          "args":{
            "lr": args.learning_rate,                     # learning_rate
            "weight_decay": args.weight_decay,               # (optional) weight decay
          }
        },                        
        "lr_scheduler": {
          "type": args.lr_scheduler,                  # learning rate scheduler
          "args":{
            
          }
        },
        "trainer": {
          "epochs": args.epochs,                     # number of training epochs
          "save_dir": "output/",              # checkpoints are saved in save_dir/models/name
          "monitor": "min val_loss",          # mode and metric for model performance monitoring. set 'off' to disable.
          "early_stop": args.early_stopping_patience,                 # number of epochs to wait before early stop. set 0 to disable.
          "report_to": "tensorboard",               # enable tensorboard visualization
        },
        "seed": args.seed,                         # random seed	
    }
    
    # Save to JSON file
    with open(output_path, "w") as f:
        json.dump(config_new, f, indent=4)
    
    print(f"Configuration saved to {output_path}")

def get_distance_function(distance_function):
    if distance_function == 'l2':
        return torch.pairwise_distance
    elif distance_function == 'cosine':
       return lambda x, y: 1 - F.cosine_similarity(x, y)
    else:
        raise ValueError(f"Unknown distance function: {distance_function}")