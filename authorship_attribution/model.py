import torch.nn as nn
from transformers import AutoModel, TrainingArguments, Trainer

class AuthorshipAttributionLLM(nn.Module):
    def __init__(self, model_name, num_labels):
        super(AuthorshipAttributionLLM, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.model.config.hidden_size, num_labels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.pooler_output) 
        probs = self.softmax(logits)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(probs, labels)

        return {'loss': loss, 'logits': probs}
