import torch
import torch.nn as nn
from speechbrain.lobes.models.ECAPA_TDNN import AttentiveStatisticsPooling
from utils.losses import SelfContrastiveLoss
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

STYLE_MODEL = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
LINGUISTICS_MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-english"


class CompressionModule(nn.Module):
    def __init__(
        self, K=23, F_in=1024, F_out=256, bottleneck_dropout=0.5, head_dropout=0.5
    ):
        super(CompressionModule, self).__init__()
        self.K = K
        self.bottleneck_dropout = bottleneck_dropout
        self.head_dropout = head_dropout

        self.pool = nn.AdaptiveAvgPool1d(
            F_out
        )  # need to ensure the shape of input is {N, C, F_in} ; OR figure out how to get this done without reshaping
        self.bottleneck = nn.Sequential(
            nn.Linear(F_in, F_out),
            nn.BatchNorm1d(F_out),
            nn.LeakyReLU(),
            nn.Dropout(self.bottleneck_dropout),
            nn.Linear(F_out, F_in),
        )

        self.head = nn.Sequential(
            nn.Dropout(self.head_dropout),
            nn.LeakyReLU(),
            nn.Linear(F_in, F_out),
        )

    def forward(self, x):
        x_pool = self.pool(x)
        x = self.bottleneck(x_pool)
        x = self.head(x + x_pool)
        return x


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
        F_in=1024,
        F_out=256,
        bottleneck_dropout=0.5,
    ):
        super(SLIMStage1, self).__init__()

        self.style_model = Wav2Vec2ForCTC.from_pretrained(STYLE_MODEL)
        self.linguistics_model = Wav2Vec2ForCTC.from_pretrained(LINGUISTICS_MODEL)

        self.style_layers = style_layers
        self.linguistics_layers = linguistics_layers

        self.style_compression = CompressionModule(K, F_in, F_out, bottleneck_dropout)
        self.linguistics_compression = CompressionModule(
            K, F_in, F_out, bottleneck_dropout
        )

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
        with torch.no_grad():
            print(x.size())
            out_style = self.style_model(x, output_hidden_states=True, return_dict=True)
            out_linguistics = self.linguistics_model(
                x, output_hidden_states=True, return_dict=True
            )

            x_style = torch.cat(
                out_style.hidden_states[self.style_layers[0] : self.style_layers[1]], dim=-1
            )
            x_linguistics = torch.cat(
                out_linguistics.hidden_states[
                    self.linguistics_layers[0] : self.linguistics_layers[1]
                ],
                dim=-1,
            )

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
