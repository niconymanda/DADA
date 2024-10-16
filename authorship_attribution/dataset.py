import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer  
import random 

class AuthorClassificationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.texts = data['text'].tolist()
        self.labels = data['label'].tolist()
        # self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.tokenizer = tokenizer
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
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.texts_by_author = data.groupby('label')['text'].apply(list).to_dict()
        self.labels = list(self.texts_by_author.keys())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        auchor_data = self.data.iloc[idx]
        auchor_text = auchor_data['text']
        auchor_label = auchor_data['label']
        
        positive_example = self._get_positive_example(auchor_label)
        negative_example = self._get_negative_example(auchor_label)
        
        anchor_inputs = self.tokenizer(
            auchor_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        positibe_inputs = self.tokenizer(
            positive_example,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
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
            "positive_input_ids": positibe_inputs["input_ids"].squeeze(),
            "positive_attention_mask": positibe_inputs["attention_mask"].squeeze(),
            "negative_input_ids": negative_inputs["input_ids"].squeeze(),
            "negative_attention_mask": negative_inputs["attention_mask"].squeeze(),
        }

    def _get_positive_example(self, label):
        positive_samples = self.texts_by_author[label]
        return random.choice(positive_samples)  
    
    def _get_negative_example(self, label):
        negative_label = random.choice(self.labels)
        while negative_label == label:
            negative_label = random.choice(self.labels)
        negative_samples = self.texts_by_author[negative_label]
        return random.choice(negative_samples)
        
        

