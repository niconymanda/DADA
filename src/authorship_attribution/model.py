import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from torch.nn import functional as F
import torch
from sklearn.mixture import GaussianMixture
from einops import rearrange

class BottleNeck(nn.Module):
    def __init__(self, K=199, F_in=1024, F_out=256, bottleneck_dropout=0.5):
        super(BottleNeck, self).__init__()
        self.bottleneck_dropout = bottleneck_dropout

        self.lin1 = nn.Linear(F_in, F_out)
        self.bn1 = nn.BatchNorm1d(K)
        self.relu = nn.LeakyReLU()
        self.drop = nn.Dropout(self.bottleneck_dropout)
        self.lin2 = nn.Linear(F_out, F_in)

    def forward(self, x):
        x = self.lin1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.lin2(x)
        return x


class CompressionModule(nn.Module):
    def __init__(
        self, K=199, F_in=1024, F_out=256, bottleneck_dropout=0.5, head_dropout=0.5
    ):
        super(CompressionModule, self).__init__()
        self.K = K
        self.bottleneck_dropout = bottleneck_dropout
        self.head_dropout = head_dropout
        self.pool = lambda x: torch.mean(x, dim=-1)

        self.bottleneck = BottleNeck(K, F_in, F_out, bottleneck_dropout)

        self.head = nn.Sequential(
            nn.Dropout(self.head_dropout),
            nn.LeakyReLU(),
            nn.Linear(F_in, F_out),
        )

    def forward(self, x):
        x_pool = self.pool(x)
        x_bottle = self.bottleneck(x_pool)
        x_head = self.head(x_bottle + x_pool)
        x_norm = nn.functional.normalize(x_head, p=2, dim=-1)
        return x_norm
    

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
    
class AuthorshipLLM(nn.Module):
    def __init__(self, model_name, 
                 bottleneck_dropout=0.5,
                 dropout_rate=0.2, 
                 out_features=256, 
                 max_length=64, 
                 device='cuda'):
        super(AuthorshipLLM, self).__init__()
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        hidden_size = self.model.config.hidden_size
        self.compression = CompressionModule(K=199, 
                                                    F_in=hidden_size, 
                                                    F_out=out_features,
                                                    bottleneck_dropout=bottleneck_dropout,
                                                    head_dropout=dropout_rate)
        self.device = device

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
            attention_mask = tokens["attention_mask"].to(self.device)
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs.last_hidden_state[:, 0, :]
        return pooled_output
    
    def forward(self, input, mode='triplet'):
        if mode=="classification":
            x = self.get_features(input['x'])
            x = self.compression(x)
            x = rearrange(x, "n t f -> n f t")
            x = x.mean(dim=-1)
            return x
        
        elif mode == 'triplet':
            a, p, n = input["anchor"], input["positive"], input["negative"]
            x_a, x_p, x_n = self.get_features(a), self.get_features(p), self.get_features(n)
            x_a, x_p, x_n = (
                self.compression(x_a),
                self.compression(x_p),
                self.compression(x_n),
            )
            x_a, x_p, x_n = (
                rearrange(x_a, "n t f -> n f t"),
                rearrange(x_p, "n t f -> n f t"),
                rearrange(x_n, "n t f -> n f t"),
            )
            x_a, x_p, x_n = (
                x_a.mean(dim=-1),
                x_p.mean(dim=-1),
                x_n.mean(dim=-1),
            )
            return {
                "anchor": x_a,
                "positive": x_p,
                "negative": x_n,
            }
        else:
            raise NotImplementedError(f"Mode {self.mode} not implemented")
