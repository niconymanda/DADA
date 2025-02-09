import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, DebertaV2Tokenizer
import random 
import json
import pandas as pd 
import os
import numpy as np

class AuthorClassificationDataset(Dataset):
    """
    A custom dataset class for author classification tasks using Hugging Face tokenizers.
    Args:
        data (pd.DataFrame): A pandas DataFrame containing the text and label columns.
    Attributes:
        texts (list): A list of texts from the DataFrame.
        labels (list): A list of labels corresponding to the texts.
        tokenizer (AutoTokenizer): The tokenizer instance loaded from the pre-trained model.
        max_length (int): The maximum length of the tokenized sequences.
    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Returns a dictionary containing the tokenized input_ids, attention_mask, and label for the given index.
    """
    
    def __init__(self, data):
        self.texts = data['transcription'].tolist()
        self.labels = data['type'].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        label = 1 if label == 'spoof' else 0  

        return {
            "text": text,
            "label": torch.tensor(label, dtype=torch.long)
        }
        
# class ASVSpoof21Dataset(Dataset):
#     """
#     A custom dataset class for handling ASVSpoof 2021 dataset.
#     Args:
#         data (pd.DataFrame): A pandas DataFrame containing the text and label columns.
#     Attributes:
#         texts (list): A list of texts from the DataFrame.
#         labels (list): A list of labels corresponding to the texts.
#     Methods:
#         __len__(): Returns the number of samples in the dataset.
#         __getitem__(idx): Returns a dictionary containing the text and the label.
#     """
    
#     def __init__(self, data):
#         self.texts = data['transcription'].tolist()
#         self.labels = data['type'].tolist()
        
#     def __len__(self):
#         return len(self.texts)

#     def __getitem__(self, idx):
#         text = self.texts.iloc[idx]
#         label = self.labels.iloc[idx]
        
#         return {
#             "text": text,
#             "label": label
#         }
        
        
class AuthorTripletLossDataset(Dataset):
    """
    A PyTorch Dataset class for generating triplets of text data for training a triplet loss model for authorship attribution.
    Attributes:
        data (pd.DataFrame): The dataset containing text data and labels.
        train (bool): A flag indicating whether the dataset is for training or testing.
        predefined_set (str): Path to a JSON file containing predefined triplets.
        mode (str): The mode of the dataset, either 'train' or 'test'.
    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(idx): Returns a triplet of text data (anchor, positive, negative) and their labels for the given index.
        _get_positive_example(label): Returns a positive example for the given label.
        _get_negative_example(label): Returns a negative example and its label for the given label.
        _get_negative_examples_all_authors(anchor_label): Returns negative examples and their labels for all authors except the given anchor label.
    """

    
    def __init__(self, data, train=True, predefined_set=None, mode=None):
        self.data = data
        self.train = train
        self.texts_by_author = data.groupby('label')['text'].apply(list).to_dict()
        self.labels = list(self.texts_by_author.keys())
        self.predefined_set = predefined_set
        self.mode = mode
        
        if self.predefined_set is not None:
            with open(self.predefined_set, 'r') as f:
                self.predefined_triplets = json.load(f)
        else:
            self.predefined_triplets = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mode = self.mode if self.mode is not None else 'train' if self.train else 'test'
        if self.predefined_triplets is not None:
            if mode not in self.predefined_triplets:
                raise KeyError(f"Mode '{mode}' not found in predefined triplets.")  
            triplet = self.predefined_triplets[mode][idx]
            try:
                anchor_example = self.data[self.data['index'] == triplet[0]].iloc[0]
                positive_example = self.data[self.data['index'] == triplet[1]].iloc[0]
                negative_example = self.data[self.data['index'] == triplet[2]].iloc[0]
            except:
                print(f"Index {triplet} not found in the dataset.")
                pass
            return {
                "anchor": anchor_example['text'],
                "positive": positive_example['text'],
                "negative": negative_example['text'],
                "label": anchor_example['label'],
                "negative_label": negative_example['label']
            }
        anchor_data = self.data.iloc[idx]
        anchor_example = anchor_data['text']
        anchor_label = anchor_data['label']
        
        positive_example = self._get_positive_example(anchor_label)
        
        if self.train:
            negative_example, negative_label = self._get_negative_example(anchor_label)
            
        else:
            negative_examples, negative_labels = self._get_negative_examples_all_authors(anchor_label)
        if self.train:
            return {
                "anchor": anchor_example,
                "positive": positive_example,
                "negative": negative_example,
                "label": anchor_label,
                "negative_label": negative_label
            }
        else:    
            return {
                "anchor": anchor_example,
                "positive": positive_example,
                "negatives": negative_examples,
                "label": anchor_label,
                "negative_labels": negative_labels
            }
            
        
    def _get_positive_example(self, label):
        positive_samples = self.texts_by_author[label]
        return random.choice(positive_samples)  
    
    def _get_negative_example(self, label):
        negative_label = random.choice(self.labels)
        while negative_label == label:
            negative_label = random.choice(self.labels)
        negative_samples = self.texts_by_author[negative_label]
        return random.choice(negative_samples), negative_label
    
    def _get_negative_examples_all_authors(self, anchor_label):
        
        negative_sampes_all_authors = []
        negative_labels = []
        for label, texts in self.texts_by_author.items():
            if label != anchor_label:
                negative_sample =  random.choice(texts)
                negative_sampes_all_authors.append(negative_sample)
                negative_labels.append(label)
        return negative_sampes_all_authors, negative_labels
    
    def __len__(self):
        # Return the length of the triplet list for the current mode
        if self.predefined_set is not None:
            mode = 'train' if self.train else 'test'
            return len(self.predefined_triplets[mode])
        else:
            return len(self.data)
        
        
