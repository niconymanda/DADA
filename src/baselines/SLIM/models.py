import torch
import torch.nn as nn
from speechbrain.lobes.models.ECAPA_TDNN import AttentiveStatisticsPooling
from utils.losses import SelfContrastiveLoss
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torchaudio
import librosa 

STYLE_MODEL = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
LINGUISTICS_MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-english"


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


class SLIMStage1(nn.Module):
    """
    ...
    TODO: Add docstring
    TODO: add freezing of the style and linguistics models etc
    """

    def __init__(
        self,
        style_layers=(0, 10),
        linguistics_layers=(14, 21),
        K=23,
        F_in=99,
        F_out=256,
        bottleneck_dropout=0.5,
        device="cuda",
    ):
        super(SLIMStage1, self).__init__()

        self.processor = Wav2Vec2Processor.from_pretrained(LINGUISTICS_MODEL)
        self.style_model = Wav2Vec2ForCTC.from_pretrained(STYLE_MODEL)
        self.linguistics_model = Wav2Vec2ForCTC.from_pretrained(LINGUISTICS_MODEL)

        self.style_model.eval()
        self.linguistics_model.eval()

        self.style_layers = style_layers
        self.linguistics_layers = linguistics_layers

        self.style_compression = CompressionModule(K, F_in, F_out, bottleneck_dropout)
        self.linguistics_compression = CompressionModule(
            K, F_in, F_out, bottleneck_dropout
        )
        self.device = device

        self.self_contrastive_loss = SelfContrastiveLoss()

    def load_(self, path):
        # self.style_model.load_state_dict(torch.load(f"{dir}/style_model.pth"))
        # self.linguistics_model.load_state_dict(
        #     torch.load(f"{dir}/linguistics_model.pth")
        # )
        # self.style_compression.load_state_dict(
        #     torch.load(f"{dir}/style_compression.pth")
        # )
        self.linguistics_compression.load_state_dict(torch.load(path))

    def save_(self, dir):
        torch.save(self.style_model.state_dict(), f"{dir}/style_model.pth")
        torch.save(self.linguistics_model.state_dict(), f"{dir}/linguistics_model.pth")
        torch.save(self.style_compression.state_dict(), f"{dir}/style_compression.pth")
        torch.save(
            self.linguistics_compression.state_dict(),
            f"{dir}/linguistics_compression.pth",
        )

    def get_features(self, x):
        with torch.no_grad():
            # x = torchaudio.functional.resample(x, 32_000, 16_000)
            x = self.processor(
                x,
                return_tensors="pt",
                padding=True,
                sampling_rate=16_000,
                device=self.device,
            ).input_values[0]
            x = x.to(self.device)

            style = self.style_model(x, output_hidden_states=True, return_dict=True)
            linguistics = self.linguistics_model(
                x, output_hidden_states=True, return_dict=True
            )
        return style, linguistics

    def train_(self):
        self.style_model.eval()
        self.linguistics_model.eval()
        self.style_compression.train()
        self.linguistics_compression.train()

    def eval_(self):
        self.style_model.eval()
        self.linguistics_model.eval()
        self.style_compression.eval()
        self.linguistics_compression.eval()

    def forward(self, x):

        out_style, out_linguistics = self.get_features(x)

        x_style = torch.cat(
            out_style.hidden_states[self.style_layers[0] : self.style_layers[1]], dim=-1
        )
        x_linguistics = torch.cat(
            out_linguistics.hidden_states[
                self.linguistics_layers[0] : self.linguistics_layers[1]
            ],
            dim=-1,
        )

        print(x_style.shape, x_linguistics.shape)
        x_style = self.style_compression(x_style)
        x_linguistics = self.linguistics_compression(x_linguistics)

        loss = self.self_contrastive_loss(x_style, x_linguistics)
        return loss


class ASP_MLP(nn.Module):
    """
    input : Features from wav2vec2-xlsr of size (batch_size, transformer_layers, features, Timesteps)
    Applies an AttentiveStatisticsPooing Layers followed by an MLP to reduce features
    ouput : embeddings of size (batch_size, out_channels)
    """

    def __init__(self, in_channels, attention_channels, out_channels):
        self.in_channels = in_channels
        self.attention_channels = attention_channels

        self.asp = AttentiveStatisticsPooling(
            channels=self.in_channels, attention_channels=self.attention_channels
        )
        self.mlp = nn.Sequential(nn.Linear)
        pass

    def forward(self, x):
        pass


class SLIMStage2(nn.Module):
    """
    ...
    """

    def __init__(self, *args, **kwargs):

        self.style_model = Wav2Vec2ForCTC.from_pretrained(STYLE_MODEL)
        self.linguistics_model = Wav2Vec2ForCTC.from_pretrained(LINGUISTICS_MODEL)

        self.style_compression = CompressionModule(K, F_in, F_out, bottleneck_dropout)
        self.linguistics_compression = CompressionModule(
            K, F_in, F_out, bottleneck_dropout
        )

        self.style_asp_mlp = ...
        self.linguistics_asp_mlp = ...

        self.head = nn.Sequential(
            [nn.Linear(1024, 256), nn.LeakyReLU(), nn.Linear(256, 2), nn.Sigmoid()]
        )

    def load_(self):
        pass

    def forward():
        pass


if __name__ == "__main__":
    cm = CompressionModule()
    print(cm)
