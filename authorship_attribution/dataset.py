import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer  
import random 

class AuthorClassificationDataset(Dataset):
    """
    A custom dataset class for author classification tasks using Hugging Face tokenizers.
    Args:
        data (pd.DataFrame): A pandas DataFrame containing the text and label columns.
        tokenizer (str): The name or path of the pre-trained tokenizer to use.
        max_length (int, optional): The maximum length of the tokenized sequences. Defaults to 128.
    Attributes:
        texts (list): A list of texts from the DataFrame.
        labels (list): A list of labels corresponding to the texts.
        tokenizer (AutoTokenizer): The tokenizer instance loaded from the pre-trained model.
        max_length (int): The maximum length of the tokenized sequences.
    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Returns a dictionary containing the tokenized input_ids, attention_mask, and label for the given index.
    """
    
    def __init__(self, data, tokenizer, max_length=128):
        self.texts = data['text'].tolist()
        self.labels = data['label'].tolist()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        inputs = self.tokenizer(
            f"Identify author: \"{text}\"",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = inputs["input_ids"].squeeze()
        attention_mask = inputs["attention_mask"].squeeze()
        return {
            "input_ids": input_ids.to(dtype=torch.long),
            "attention_mask": attention_mask.to(dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long)
        }
        
        
class AuthorTripletLossDataset(Dataset):
    """
    A PyTorch Dataset class for generating triplets of (anchor, positive, negative) samples for training
    a triplet loss model in the context of authorship attribution.
    Attributes:
        data (pd.DataFrame): The dataset containing text samples and their corresponding labels.
        tokenizer (AutoTokenizer): The tokenizer used to preprocess the text samples.
        max_length (int): The maximum length of the tokenized sequences.
        train (bool): A flag indicating whether the dataset is used for training or evaluation.
        texts_by_author (dict): A dictionary mapping author labels to lists of their text samples.
        labels (list): A list of unique author labels.
    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Returns a dictionary containing tokenized anchor, positive, and negative samples
                          along with their attention masks and labels for the given index.
        _get_positive_example(label): Returns a positive example (text) for the given author label.
        _get_negative_example(label): Returns a negative example (text) and its label for the given author label.
        _get_negative_examples_all_authors(anchor_label): Returns negative examples (texts) and their labels
                                                          for all authors except the given anchor label.
    """
    
    def __init__(self, data, tokenizer_name, max_length=64, train=True):
        self.data = data
        self.tokenizer =  AutoTokenizer.from_pretrained(tokenizer_name)

        self.max_length = max_length
        self.train = train
        self.texts_by_author = data.groupby('label')['text'].apply(list).to_dict()
        self.labels = list(self.texts_by_author.keys())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        auchor_data = self.data.iloc[idx]
        auchor_text = auchor_data['text']
        anchor_label = auchor_data['label']
        
        positive_example = self._get_positive_example(anchor_label)
        
        if self.train:
            negative_example, negative_label = self._get_negative_example(anchor_label)
            
        else:
            negative_examples, negative_labels = self._get_negative_examples_all_authors(anchor_label)
            
        
        anchor_inputs = self.tokenizer(
            auchor_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        # avg_token_length = sum(len(self.tokenizer.tokenize(text)) for text in self.texts_by_author[anchor_label]) / len(self.texts_by_author[anchor_label])
        # print(f"Average token length for author {anchor_label}: {avg_token_length}")
        
        positive_inputs = self.tokenizer(
            positive_example,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        if self.train:
            negative_inputs = self.tokenizer(
                negative_example,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            return{
                "anchor_input_ids": anchor_inputs["input_ids"].squeeze(),
                "anchor_attention_mask": anchor_inputs["attention_mask"].squeeze(),
                "positive_input_ids": positive_inputs["input_ids"].squeeze(),
                "positive_attention_mask": positive_inputs["attention_mask"].squeeze(),
                "negative_input_ids": negative_inputs["input_ids"].squeeze(),
                "negative_attention_mask": negative_inputs["attention_mask"].squeeze(),
                "label": anchor_label, 
                "negative_label": negative_label
            }
        else:
            negative_inputs = [self.tokenizer(
                neg_example,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ) for neg_example in negative_examples]
                
            negative_input_ids = torch.stack([neg_inputs["input_ids"].squeeze() for neg_inputs in negative_inputs])
            negative_attention_masks = torch.stack([neg_inputs["attention_mask"].squeeze() for neg_inputs in negative_inputs])
            return {
                "anchor_input_ids": anchor_inputs["input_ids"].squeeze(),
                "anchor_attention_mask": anchor_inputs["attention_mask"].squeeze(),
                "positive_input_ids": positive_inputs["input_ids"].squeeze(),
                "positive_attention_mask": positive_inputs["attention_mask"].squeeze(),
                "negative_input_ids": negative_input_ids,
                "negative_attention_mask": negative_attention_masks,
                "label": anchor_label,
                "negative_labels": torch.stack([torch.tensor(neg_label) for neg_label in negative_labels])
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
        
        

