import torch
import torch.nn as nn
from speechbrain.lobes.models.ECAPA_TDNN import AttentiveStatisticsPooling
from einops import rearrange
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

BASE_MODEL = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
PROC_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
# LINGUISTICS_MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-english"


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
        x_norm = rearrange(x_norm, "n t f -> n f t")
        x_norm = x_norm.mean(dim=-1)
        x_norm = nn.functional.normalize(x_norm, p=2, dim=-1)
        return x_norm


class SpeechEmbedder(nn.Module):
    def __init__(
        self,
        feature_layers=(0, 10),
        K=199,
        F_in=1024,
        F_out=256,
        bottleneck_dropout=0.5,
        head_dropout=0.5,
        load_path=None,
        device="cuda",
    ):
        super(SpeechEmbedder, self).__init__()
        self.processor = Wav2Vec2Processor.from_pretrained(PROC_ID)
        self.feature_model = Wav2Vec2ForCTC.from_pretrained(BASE_MODEL)
        self.feature_model.eval()
        self.compression = CompressionModule(
            K=K,
            F_in=F_in,
            F_out=F_out,
            bottleneck_dropout=bottleneck_dropout,
            head_dropout=head_dropout,
        )
        if load_path:
            self.compression.load_state_dict(torch.load(load_path))
        self.feature_layers = feature_layers
        self.device = device

    def get_features(self, x):
        with torch.no_grad():
            x = self.processor(
                x,
                return_tensors="pt",
                padding=True,
                sampling_rate=16_000,
                device=self.device,
            ).input_values[0]
            x = x.to(self.device)
            out = self.feature_model(x, output_hidden_states=True, return_dict=True)
            feat = torch.stack(
                out.hidden_states[self.feature_layers[0] : self.feature_layers[1]],
                dim=-1,
            )
        return feat

    def to(self, device):
        self.compression.to(device)
        self.feature_model.to(device)
        return super().to(device)

    def train(self):
        self.compression.train()

    def eval(self):
        self.compression.eval()

    def save_(self, path):
        torch.save(self.compression.state_dict(), path)

    def load_(self, path):
        self.compression.load_state_dict(torch.load(path))

    def forward(self, input, mode="triplet"):

        if mode == "classification":
            x = self.get_features(input["x"])
            x = self.compression(x)
            return x

        elif mode == "triplet":
            a, p, n = input["anchor"], input["positive"], input["negative"]
            x_a, x_p, x_n = (
                self.get_features(a),
                self.get_features(p),
                self.get_features(n),
            )

            x_a, x_p, x_n = (
                self.compression(x_a),
                self.compression(x_p),
                self.compression(x_n),
            )

            return {
                "anchor": x_a,
                "positive": x_p,
                "negative": x_n,
            }

        elif mode == "pair":
            a, b = input["a"], input["b"]
            x_a, x_b = self.get_features(a), self.get_features(b)

            x_a, x_b = self.compression(x_a), self.compression(x_b)

            return {
                "a": x_a,
                "b": x_b,
            }

        else:
            raise NotImplementedError(f"Mode {self.mode} not implemented")
