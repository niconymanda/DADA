import torch.nn as nn
from transformers import AutoModel

class AuthorshipClassificationLLM(nn.Module):
    def __init__(self, model, num_labels, class_weights=None):
        super(AuthorshipClassificationLLM, self).__init__()
        self.num_labels = num_labels
        self.class_weights = class_weights
        self.model = model
        hidden_size = self.model.model.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.softmax = nn.Softmax(dim=1)
        self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs)
        probs = self.softmax(logits)
        return probs
class AuthorshipLLM(nn.Module):
    def __init__(self, model_name):
        super(AuthorshipLLM, self).__init__()

        self.model = AutoModel.from_pretrained(model_name)
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output 
        return pooled_output
