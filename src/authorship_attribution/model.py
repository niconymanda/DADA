import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from torch.nn import functional as F
import torch
from sklearn.mixture import GaussianMixture

class AuthorshipClassificationLLM(nn.Module):
    def __init__(self, model, num_labels, head_type='gmm', class_weights=None):
        super(AuthorshipClassificationLLM, self).__init__()
        self.num_labels = num_labels
        self.class_weights = class_weights
        self.model = model
        self.tokenizer = model.tokenizer
        self.max_length = model.max_length

        hidden_size = self.model.model.config.hidden_size
        self.head_type = head_type
        if head_type == 'linear':
            self.classifier = nn.Linear(hidden_size, num_labels)
            self.softmax = nn.Softmax(dim=1)
            self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
            self.freeze_params()
        elif head_type == 'mlp':
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, num_labels)
            )
            self.softmax = nn.Softmax(dim=1)
            self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
            self.freeze_params()
        
    def freeze_params(self):
        # Freeze all layers except the classification head
        for name, param in self.model.named_parameters():
            if 'classifier' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False


    def forward(self, example):
        outputs = self.model(example)
        logits = self.classifier(outputs)
        probs = self.softmax(logits)
        return probs
    
class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings
    
class AuthorshipLLM(nn.Module):
    def __init__(self, model_name, 
                 bottleneck_dropout=0.5,
                 dropout_rate=0.2, 
                 out_features=256, 
                 max_length=64, 
                 device='cuda', 
                 freeze_encoder=False):
        super(AuthorshipLLM, self).__init__()
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

        self.freeze_params() if freeze_encoder else None

        self.pooler = MeanPooling()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(in_features=self.model.config.hidden_size,out_features=256)

        self.device = device
    def freeze_params(self):
        for param in self.model.base_model.parameters():
            param.requires_grad = False

    def init_embeddings(self):
        """
        Initialize embeddings with a Gaussian distribution N(0, 1).
        """
        for param in self.model.embeddings.parameters():
            nn.init.normal_(param, mean=0.0, std=1.0)

    def get_features(self, input):
        with torch.no_grad():
            tokens = self.tokenizer(input, 
                                       padding=True, 
                                       truncation=True, 
                                       max_length=self.max_length, 
                                       return_tensors="pt")

            input_ids = tokens["input_ids"].to(self.device)
            token_type_ids = tokens["token_type_ids"].to(self.device)
            attention_mask = tokens["attention_mask"].to(self.device)
            inputs = {
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_mask,
            }
        return inputs

    def forward(self, input, mode='triplet'):
   
        a, p, n = input["anchor"], input["positive"], input["negative"]
        x_a, x_p, x_n = self.get_features(a), self.get_features(p), self.get_features(n)

        x_a_output, x_p_output, x_n_output = (
            self.model(**x_a, return_dict=True),
            self.model(**x_p, return_dict=True),
            self.model(**x_n, return_dict=True))
        
        x_a_output, x_p_output, x_n_output = (
            self.pooler(x_a_output['last_hidden_state'], x_a['attention_mask']),
            self.pooler(x_p_output['last_hidden_state'], x_p['attention_mask']),
            self.pooler(x_n_output['last_hidden_state'], x_n['attention_mask']))
        x_a_output, x_p_output, x_n_output = (
            self.fc1(x_a_output),
            self.fc1(x_p_output),
            self.fc1(x_n_output))
        x_a_output, x_p_output, x_n_output = (
            F.normalize(x_a_output, p=2, dim=-1),
            F.normalize(x_p_output, p=2, dim=-1),
            F.normalize(x_n_output, p=2, dim=-1))   

        return {
            "anchor": x_a_output,
            "positive": x_p_output,
            "negative": x_n_output,
        }
        
