import torch 
import torch.nn as nn

import numpy as np

class MidFuse(nn.Module):
    def __init__(self, text_model, speech_model, text_features, speech_features):
        super(MidFuse, self).__init__()
        self.text_model = text_model
        self.speech_model = speech_model
        self.text_features = text_features
        self.speech_features = speech_features

        self.classifier = nn.Sequential(
            nn.Linear(text_features + speech_features, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def train(self):
        self.text_model.eval()
        self.speech_model.eval()
        self.classifier.train()

    def eval(self):
        self.text_model.eval()
        self.speech_model.eval()
        self.classifier.eval()

    def trainable_parameters(self):
        return self.classifier.parameters()
    
    def load_(self, path):
        self.classifier.load_state_dict(torch.load(path))

    def save_(self, path):
        torch.save(self.classifier.state_dict(), path)
    
    def eval(self):
        self.text_model.eval()
        self.speech_model.eval()
        self.classifier.eval()

    def forward(self, text_input, speech_input):
        text_features = self.text_model(input_ids = text_input['input_ids'], attention_mask = text_input['attention_mask'])
        speech_features = self.speech_model(speech_input, mode='classification')
        features = torch.cat([text_features, speech_features], dim=1)
        x = self.classifier(features)
        return x
