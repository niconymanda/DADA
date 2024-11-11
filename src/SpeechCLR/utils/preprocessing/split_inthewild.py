import numpy as np
import pandas as pd
import yaml
import argparse

def split_dataset(dataset, train_authors, val_authors):
    authors = dataset['speaker'].value_counts().index.tolist() # Authors in decreasing order of number of samples
    
    train_authors_set = set(authors[:train_authors])
    val_authors_set = set(authors[train_authors:train_authors + val_authors])
    test_authors_set = set(authors[train_authors + val_authors:])
    
    train_data = dataset[dataset['speaker'].isin(train_authors_set)]
    val_data = dataset[dataset['speaker'].isin(val_authors_set)]
    test_data = dataset[dataset['speaker'].isin(test_authors_set)]
    
    return train_data, val_data, test_data

def main():
    parser = argparse.ArgumentParser(description="Split dataset into train, val, and test sets")
    parser.add_argument('--input', type=str, required=True, help='Path to the input dataset')
    parser.add_argument('--train_authors', type=int, required=True, help='Number of authors in the training set')
    parser.add_argument('--val_authors', type=int, required=True, help='Number of authors in the validation set')
    parser.add_argument('--output_dir', type=str, required=True, default = 'configs', help='Path to the output directory')
    parser.add_argument('--output_name', type=str, required=True, default = 'inthewild.yaml', help='Name of the output file')
    parser.add_argument('--train_ratio', type=float, required=True, default = 0.8, help='Ratio of training data')
    
    args = parser.parse_args()
    
    dataset = pd.read_csv(args.input)
    train_data, val_data, test_data = split_dataset(dataset, args.train_authors, args.val_authors)
    
    train_data.to_csv(args.output_train, index=False)
    val_data.to_csv(args.output_val, index=False)
    test_data.to_csv(args.output_test, index=False)

if __name__ == "__main__":
    main()