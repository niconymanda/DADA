import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from torch.nn import functional as F
import torch
from sklearn.mixture import GaussianMixture
import torchvision.ops as ops

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


    def forward(self, input):
        outputs = self.model(input)['anchor']
        logits = self.classifier(outputs)
        probs = self.softmax(logits)
        return probs
    
class MeanPooling(nn.Module):
    """
    Mean Pooling layer for aggregating hidden states.
    This layer computes the mean of the hidden states, taking into account the attention mask.
    Methods
    -------
    forward(last_hidden_state, attention_mask)
        Computes the mean of the hidden states, weighted by the attention mask.
    Parameters
    ----------
    last_hidden_state : torch.Tensor
        The hidden states from the last layer of the model, with shape (batch_size, sequence_length, hidden_size).
    attention_mask : torch.Tensor
        The attention mask indicating which tokens are valid (1) and which are padding (0), with shape (batch_size, sequence_length).
    Returns
    -------
    torch.Tensor
        The mean-pooled hidden states, with shape (batch_size, hidden_size).
    """
    
    def __init__(self, use_layers=None):
        super(MeanPooling, self).__init__()
        self.use_layers = use_layers
        
    def forward(self, hidden_states, attention_mask):
        if self.use_layers is None:
            # Use all layers if not specified
            self.use_layers = list(range(len(hidden_states))) 

        selected_states = [hidden_states[layer_idx] for layer_idx in self.use_layers]
        concatenated_states = torch.cat(selected_states, dim=-1)

        attention_mask = attention_mask.unsqueeze(-1)
        mean_embeddings = torch.sum(concatenated_states * attention_mask, dim=1) / attention_mask.sum(dim=1).clamp(min=1e-9)

        return mean_embeddings
    

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) for projecting hidden states to a lower-dimensional space.
    Methods
    -------
    forward(x)
        Projects the input tensor to a lower-dimensional space.
    Parameters
    ----------
    num_layers : int
        The number of layers in the MLP.
    """
    
    def __init__(self, num_layers, in_features=1024, out_features=512, dropout_rate=0.3, norm_layer=None):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        # self.mlp = nn.Sequential(
        #     nn.Linear(in_features, in_features//2),
        #     # nn.BatchNorm1d(out_features),
        #     nn.ReLU(),
        #     nn.Dropout(self.dropout_rate),
        #     nn.Linear(in_features//2, out_features),  
        #     # nn.Dropout(self.dropout_rate),
        # )
        self.layers = []
        in_dim = in_features
        norm_layer = nn.BatchNorm1d
        for _ in range(num_layers-1):
            out_features_layer = in_dim // 2
            self.layers.append(torch.nn.Linear(in_dim, out_features_layer))
            if norm_layer is not None:
                self.layers.append(norm_layer(out_features_layer))
            self.layers.append(torch.nn.LeakyReLU())
            self.layers.append(torch.nn.Dropout(self.dropout_rate))
            in_dim = out_features_layer

        self.layers.append(torch.nn.Linear(in_dim, out_features))
        self.layers.append(torch.nn.Dropout(dropout_rate))
        self.mlp = nn.Sequential(*self.layers)
        
    def forward(self, x):
        return self.mlp(x)
    
class AuthorshipLLM(nn.Module):
    def __init__(self, model_name, 
                 dropout_rate=0.2, 
                 out_features=1024, 
                 max_length=64, 
                 num_layers=4,
                 device='cuda', 
                 freeze_encoder=False, 
                 use_layers=[-1, -2]):  
        super(AuthorshipLLM, self).__init__()
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.use_layers = use_layers
        self.device = device
        self.freeze_params() if freeze_encoder else None

        self.pooler = MeanPooling(self.use_layers)

        input_size = self._get_hidden_size() * len(self.use_layers)
        self.MLP = MLP(num_layers, dropout_rate=dropout_rate, in_features=input_size, out_features=out_features)
        # self.dropout = nn.Dropout(dropout_rate)
        # self.fc1 = nn.Linear(in_features=self.model.config.hidden_size,out_features=out_features)


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
            inputs = {k: v.to(self.device) for k, v in tokens.items()}
        return inputs
    
    def _get_hidden_size(self):
        
        return self.model.config.hidden_size

    def forward(self, input, mode='triplet'):
   
        a, p, n = input["anchor"], input["positive"], input["negative"]
        x_a, x_p, x_n = self.get_features(a), self.get_features(p), self.get_features(n)

        x_a_output, x_p_output, x_n_output = (
            self.model(**x_a, return_dict=True),
            self.model(**x_p, return_dict=True),
            self.model(**x_n, return_dict=True))
        # print(x_a_output.shape)
        x_a_output, x_p_output, x_n_output = (
            self.pooler(x_a_output.hidden_states, x_a['attention_mask']),
            self.pooler(x_p_output.hidden_states, x_p['attention_mask']),
            self.pooler(x_n_output.hidden_states, x_n['attention_mask']))
        
        # print(x_a_output.shape)
        x_a_output, x_p_output, x_n_output = (
            self.MLP(x_a_output),
            self.MLP(x_p_output),
            self.MLP(x_n_output))
        
        # print(x_a_output.shape)
        x_a_output, x_p_output, x_n_output = (
            F.normalize(x_a_output, p=2, dim=-1),
            F.normalize(x_p_output, p=2, dim=-1),
            F.normalize(x_n_output, p=2, dim=-1))   
        # print(x_a_output.shape)
        return {
            "anchor": x_a_output,
            "positive": x_p_output,
            "negative": x_n_output,
        }
        