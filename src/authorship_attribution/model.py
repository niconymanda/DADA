import torch.nn as nn
from transformers import AutoModel
from torch.nn import functional as F
import torch
from sklearn.mixture import GaussianMixture

class AuthorshipClassificationLLM(nn.Module):
    def __init__(self, model, num_labels, head_type='gmm', class_weights=None):
        super(AuthorshipClassificationLLM, self).__init__()
        self.num_labels = num_labels
        self.class_weights = class_weights
        self.model = model
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


    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs)
        probs = self.softmax(logits)
        return probs
    
class AuthorshipLLM(nn.Module):
    def __init__(self, model_name, dropout_rate=0.1):
        super(AuthorshipLLM, self).__init__()
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name)
        # print(self.model)
        hidden_size = self.model.config.hidden_size
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)  
        # self.freeze_layers()

    def freeze_layers(self, n_layers=5):
        """
        Freeze the first n_layers of the model.
        """
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.encoder.parameters():
            param.requires_grad = True
        for param in self.model.pooler.parameters():
            param.requires_grad = True

    def init_embeddings(self):
        """
        Initialize embeddings with a Gaussian distribution N(0, 1).
        """
        for param in self.model.embeddings.parameters():
            nn.init.normal_(param, mean=0.0, std=1.0)
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs.last_hidden_state[:, 0, :] 
        # pooled_output = F.normalize(pooled_output, p=2, dim=1)
        # pooled_output = self.batch_norm(pooled_output)
        pooled_output = self.layer_norm(pooled_output)
        pooled_output = self.dropout(pooled_output)
        
        return pooled_output
