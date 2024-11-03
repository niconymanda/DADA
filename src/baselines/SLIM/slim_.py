# %%
# !export CUDA_VISIBLE_DEVICES=1

# %%
# !echo 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 >> shuggingface-cli login 

# %%
from transformers import AutoProcessor, AutoModelForAudioClassification, AutoModelForCTC

# %%
# style_processor = AutoProcessor.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
style_model = AutoModelForAudioClassification.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")

# %%
linguistics_processor = AutoProcessor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
linguistics_model = AutoModelForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")

# %%
linguistics_model

# %%
import torch
import librosa
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

LANG_ID = "en"
MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
SAMPLES = 10

acces_token = "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"

# test_dataset = load_dataset("common_voice", LANG_ID, split=f"test[:{SAMPLES}]", trust_remote_code=True)

# test_dataset = load_dataset("common_voice", LANG_ID, split="test", trust_remote_code=True)
# test_dataset = load_dataset("mozilla-foundation/common_voice_13_0", "en", split="test", acces_token=acces_token)

test_dataset = load_dataset("mozilla-foundation/common_voice_13_0", "en", split="test", trust_remote_code=True)
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)

# Preprocessing the datasets.
# We need to read the audio files as arrays
def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = librosa.load(batch["path"], sr=16_000)
    batch["speech"] = speech_array
    batch["sentence"] = batch["sentence"].upper()
    return batch
 
test_dataset = test_dataset.map(speech_file_to_array_fn)
inputs = processor(test_dataset["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)

with torch.no_grad():
    logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

predicted_ids = torch.argmax(logits, dim=-1)
predicted_sentences = processor.batch_decode(predicted_ids)

for i, predicted_sentence in enumerate(predicted_sentences):
    print("-" * 100)
    print("Reference:", test_dataset[i]["sentence"])
    print("Prediction:", predicted_sentence)


# %%
import torch

import torch.nn as nn


class SelfContrastiveLoss(nn.Module):
    def __init__(self, lambda_ = 0.1):
        super(SelfContrastiveLoss, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x_style, x_linguistics):
        # Define the custom loss computation
        # Normalize embeddings 
        B = x_style.size(0)
        x_style_norm = nn.BatchNorm1d(x_style, affine=False) / B
        x_linguistics_norm = nn.BatchNorm1d(x_linguistics, affine=False) / B

        # Compute cross-subspace loss
        D = torch.linalg.norm(x_style_norm - x_linguistics_norm, ord='fro')
        D = torch.pow(D, 2)

        # Compute intra-subspace redundancy loss
        v_linguistic = torch.mm(x_linguistics_norm.T, x_linguistics_norm)
        C_linguistic = torch.linalg.norm(v_linguistic - torch.eye(v_linguistic.size(1)))
        C_linguistic = torch.pow(C_linguistic, 2)

        v_style = torch.mm(x_style_norm.T, x_style_norm)
        C_style = torch.linalg.norm(v_style - torch.eye(v_style.size(1)))
        C_style = torch.pow(C_style, 2)

        # Composite Loss
        loss = D = self.lambda_ * (C_linguistic + C_style)

        return loss 

# Example usage:
# criterion = CustomLoss()
# loss = criterion(outputs, targets)

# %%
class CompressionModule(nn.Module):
    def __init__(self, K=23, F_in=1024, F_out = 256, bottleneck_dropout=0.1, head_dropout=0.1):
        super(CompressionModule, self).__init__()
        self.K = K
        self.bottleneck_dropout = bottleneck_dropout
        self.head_dropout = head_dropout
        
        self.pool = nn.AdaptiveAvgPool1d(F_out) # need to ensure the shape of input is {N, C, F_in} ; OR figure out how to get this done without reshaping
        self.bottleneck = nn.Sequential([
            nn.Linear(F_in, F_out),
            nn.Batch(),
            nn.LeakyReLU(),
            nn.Dropout(self.bottleneck_dropout),
            nn.Linear(F_out, F_in)
        ])

        self.head = nn.Sequential(
            [
                nn.Dropout(self.head_dropout),
                nn.LeakyReLU(),
                nn.Linear(F_in, F_out),
            ]
        )

    def forward(self, x):
        x_pool = self.pool(x)
        x = self.bottleneck(x_pool)
        x = self.head(x + x_pool)
        return x
    
    # TODO @abhaydmathur : check shapes, sanity checks. 
    # TODO @abhaydmathur : tune hyperparameters.

# %%

class ASP_MLP(nn.Module):
    """
    the original embedding’s dimensions are reduced from 1024 to 256 through an
attentive statistics pooling (ASP) layer and a multi-layer perceptron (MLP) network. The projected
subspace embeddings when concatenated with dependency features result in 1024-dim vectors
    """

    def __init__(self, F_in=1024, F_out=256, head_dropout=0.1):
        super(ASP_MLP, self).__init__()
        self.head_dropout = head_dropout

        self.head = nn.Sequential(
            [
                nn.Dropout(self.head_dropout),
                nn.LeakyReLU(),
                nn.Linear(F_in, F_out),
            ]
        )

    def forward(self, x):
        x = self.head(x)
        return x

class ClassificationHead(nn.Module):
    """
    The
classification head consists of two fully-connected layers and a dropout layer. Binary cross-entropy
loss is used to jointly train the ASP and MLP modules alongside the classification head.
    """
    def __init__(self, F_in=1024, num_classes=2):
        super(ClassificationHead, self).__init__()
        self.head = nn.Sequential(
            [
                nn.Linear(F_in, num_classes),
            ]
        )

    def forward(self, x):
        x = self.head(x)
        return x

