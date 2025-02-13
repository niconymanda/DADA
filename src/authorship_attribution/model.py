import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from torch.nn import functional as F
import torch
from transformers import T5EncoderModel

class AuthorshipClassificationLLM(nn.Module):
    """
    A class used to represent an Authorship Classification model using a pre-trained language model.
    Attributes
    ----------
    model : AuthorshipLLM
        The pre-trained language model to use for authorship classification.
    num_labels : int
        The number of labels for classification.
    head_type : str
        The type of classification head to use ('linear' or 'mlp').

    Methods
    -------
    __init__(self, model, num_labels, head_type='linear')
        Initializes the AuthorshipClassificationLLM with the given parameters.
    freeze_params(self)
        Freezes the parameters of the encoder.
    eval(self)
        Sets the model to evaluation mode.
    train(self)
        Sets the model to training mode.
    forward(self, input)
        Defines the forward pass of the model.
    """
    def __init__(self, model, num_labels, head_type='linear'):
        super(AuthorshipClassificationLLM, self).__init__()
        self.num_labels = num_labels
        self.model = model
        self.tokenizer = model.tokenizer
        self.max_length = model.max_length
        self.device = model.device

        hidden_size = self.model.out_features
        self.head_type = head_type
        if head_type == 'linear':
            self.classifier = nn.Linear(hidden_size, num_labels)
            # self.softmax = nn.Softmax(dim=1)
            self.freeze_params()
        elif head_type == 'mlp':
            self.classifier = MLP(num_layers=3, in_features=hidden_size, out_features=num_labels, dropout_rate=0.1)
            self.softmax = nn.Softmax(dim=1)
            
            self.freeze_params()
        
    def freeze_params(self):
        for name, param in self.model.named_parameters():
            if 'classifier' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
                
    def eval(self):
        self.model.eval()
        self.classifier.eval()
        
    def train(self):
        self.model.eval()
        self.classifier.train()

    def forward(self, input, from_triplet=False):
        if from_triplet:
            input = input['anchor']
        outputs = self.model(input, mode = 'classification')
        logits = self.classifier(outputs)
        probs = self.softmax(logits)
        return logits
    
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
        self.layers = []
        in_dim = in_features
        norm_layer = nn.BatchNorm1d
        for _ in range(num_layers-1):
            out_features_layer = in_dim // 2
            self.layers.append(torch.nn.Linear(in_dim, out_features_layer))
            if norm_layer is not None:
                self.layers.append(norm_layer(out_features_layer))
            self.layers.append(torch.nn.ReLU())
            self.layers.append(torch.nn.Dropout(self.dropout_rate))
            in_dim = out_features_layer

        self.layers.append(torch.nn.Linear(in_dim, out_features))
        # self.layers.append(torch.nn.Dropout(dropout_rate))
        self.mlp = nn.Sequential(*self.layers)
        
    def forward(self, x):
        return self.mlp(x)
    
class AuthorshipLLM(nn.Module):
    '''
    A class used to represent an Authorship Attribution model using a pre-trained language model.
    Attributes
    ----------
    model_name : str The name of the pre-trained model to use.
    dropout_rate : float  The dropout rate for the MLP layers.
    out_features : int The number of output features for the MLP.
    max_length : int The maximum length of the input sequences.
    num_layers : int The number of layers in the MLP.
    device : str The device to run the model on ('cuda' or 'cpu').
    freeze_encoder : bool  Whether to freeze the encoder parameters.
    use_layers : list The layers of the model to use for pooling.
    
    Methods
    -------
    __init__(self, model_name, dropout_rate=0.2, out_features=1024, max_length=64, num_layers=4, device='cuda', freeze_encoder=False, use_layers=[-1, -2])
        Initializes the AuthorshipLLM with the given parameters.
    _get_model(self, model_name)
        Loads the pre-trained model based on the model name.
    freeze_params(self)
        Freezes the parameters of the encoder.
    init_embeddings(self)
        Initializes the embeddings with a Gaussian distribution N(0, 1).
    get_features(self, input)
        Tokenizes the input text and returns the tokenized inputs.
    _get_hidden_size(self)
        Returns the hidden size of the pre-trained model.
    forward(self, input, mode='triplet')
        Defines the forward pass of the model. Supports 'triplet' and 'classification' modes.
    '''
    
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
        self.model = self._get_model(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.use_layers = use_layers
        self.device = device
        self.out_features = out_features
        self.freeze_params() if freeze_encoder else None

        self.pooler = MeanPooling(self.use_layers)
        input_size = self._get_hidden_size() * len(self.use_layers)
        self.MLP = MLP(num_layers, dropout_rate=dropout_rate, in_features=input_size, out_features=out_features)
        
        
    def _get_model(self, model_name):
        
        if 't5' in model_name:
            return T5EncoderModel.from_pretrained(model_name, output_hidden_states=True)
        else:
            return AutoModel.from_pretrained(model_name, output_hidden_states=True)
        
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
        if mode == 'triplet':
            a, p, n = input["anchor"], input["positive"], input["negative"]
            x_a, x_p, x_n = self.get_features(a), self.get_features(p), self.get_features(n)

            x_a_output, x_p_output, x_n_output = (
                self.model(**x_a, return_dict=True),
                self.model(**x_p, return_dict=True),
                self.model(**x_n, return_dict=True))
            x_a_output, x_p_output, x_n_output = (
                self.pooler(x_a_output.hidden_states, x_a['attention_mask']),
                self.pooler(x_p_output.hidden_states, x_p['attention_mask']),
                self.pooler(x_n_output.hidden_states, x_n['attention_mask']))

            x_a_output, x_p_output, x_n_output = (
                self.MLP(x_a_output),
                self.MLP(x_p_output),
                self.MLP(x_n_output))
            x_a_output, x_p_output, x_n_output = (
                F.normalize(x_a_output, p=2, dim=-1),
                F.normalize(x_p_output, p=2, dim=-1),
                F.normalize(x_n_output, p=2, dim=-1))   
            return {
                "anchor": x_a_output,
                "positive": x_p_output,
                "negative": x_n_output,
            }
            
        elif mode == 'classification':
            input = input['text'] if isinstance(input, dict) else input
            x = self.get_features(input)
            x_output = self.model(input_ids=x['input_ids'], attention_mask=x['attention_mask'], return_dict=True)
            x_output = self.pooler(x_output.hidden_states[-1], x['attention_mask'])
            x_output = self.MLP(x_output)
            x_output = F.normalize(x_output, p=2, dim=-1)
            
            return x_output 
        else:
            raise ValueError("Invalid mode. Choose 'triplet' or 'classification'.")
        
        