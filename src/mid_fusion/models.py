import torch
import torch.nn as nn
import os
import numpy as np


class FeatBinaryClassifier(nn.Module):
    def __init__(self, features, use_bn=True):
        super(FeatBinaryClassifier, self).__init__()
        self.features = features

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.features) if use_bn else nn.Identity(),
            nn.Linear(self.features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def train_(self):
        self.classifier.train()

    def eval_(self):
        self.classifier.eval()

    def load_(self, path):
        self.classifier.load_state_dict(torch.load(path, weights_only=True))

    def save_(self, path):
        torch.save(self.classifier.state_dict(), path)

    def forward(self, input):
        x = self.classifier(input)
        return x


class MidFuse(nn.Module):
    def __init__(self, text_model, speech_model, text_features, speech_features):
        super(MidFuse, self).__init__()
        self.text_model = text_model
        self.speech_model = speech_model
        self.text_features = text_features
        self.speech_features = speech_features

        self.classifier = nn.Sequential(
            nn.Linear(self.text_features + self.speech_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def train_(self):
        self.text_model.eval()
        self.speech_model.eval()
        self.classifier.train()

    def eval_(self):
        self.text_model.eval()
        self.speech_model.eval()
        self.classifier.eval()

    def trainable_parameters(self):
        return self.classifier.parameters()

    def load_(self, path):
        self.classifier.load_state_dict(torch.load(path, weights_only=True))

    def save_(self, path):
        torch.save(self.classifier.state_dict(), path)

    def forward(self, text_input, speech_input):
        with torch.no_grad():
            text_features = self.text_model(text_input, mode="classification")
            speech_features = self.speech_model(speech_input, mode="classification")
            features = torch.cat([text_features, speech_features], dim=1)
        x = self.classifier(features)
        return x


class MidFusev2(nn.Module):
    def __init__(self, text_model, speech_model, text_features, speech_features):
        super(MidFusev2, self).__init__()
        self.text_model = text_model
        self.speech_model = speech_model
        self.text_features = text_features
        self.speech_features = speech_features

        self.speech_feat_pass = nn.Sequential(
            nn.BatchNorm1d(self.speech_features),
            nn.Linear(self.speech_features, self.speech_features),
        )

        self.text_feat_pass = nn.Sequential(
            nn.BatchNorm1d(self.text_features),
            nn.Linear(self.text_features, self.text_features),
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.text_features + self.speech_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def train_(self):
        self.text_model.eval()
        self.speech_model.eval()
        self.speech_feat_pass.train()
        self.text_feat_pass.train()
        self.classifier.train()

    def eval_(self):
        self.text_model.eval()
        self.speech_model.eval()
        self.speech_feat_pass.eval()
        self.text_feat_pass.eval()
        self.classifier.eval()

    def trainable_parameters(self):
        return (
            list(self.classifier.parameters())
            + list(self.speech_feat_pass.parameters())
            + list(self.text_feat_pass.parameters())
        )

    def load_(self, path):
        self.classifier.load_state_dict(torch.load(path, weights_only=True))
        self.speech_feat_pass.load_state_dict(
            torch.load(path.replace(".pth", "_speech_feat_pass.pth"), weights_only=True)
        )
        self.text_feat_pass.load_state_dict(
            torch.load(path.replace(".pth", "_text_feat_pass.pth"), weights_only=True)
        )

    def save_(self, path):
        torch.save(self.classifier.state_dict(), path)
        torch.save(
            self.speech_feat_pass.state_dict(),
            path.replace(".pth", "_speech_feat_pass.pth"),
        )
        torch.save(
            self.text_feat_pass.state_dict(),
            path.replace(".pth", "_text_feat_pass.pth"),
        )

    def forward(self, text_input, speech_input):
        with torch.no_grad():
            text_features = self.text_model(text_input, mode="classification")
            speech_features = self.speech_model(speech_input, mode="classification")
        text_features = self.text_feat_pass(text_features)
        speech_features = self.speech_feat_pass(speech_features)
        features = torch.cat([text_features, speech_features], dim=1)
        x = self.classifier(features)
        return x


class ConditionalLateFuse(nn.Module):
    def __init__(
        self,
        text_model,
        speech_model,
        text_features,
        speech_features,
        num_classes,
        alpha=0.5,
        tune_classifiers=False,
    ):
        super(ConditionalLateFuse, self).__init__()
        self.text_model = text_model
        self.speech_model = speech_model
        self.text_features = text_features
        self.speech_features = speech_features
        self.tune_classifiers = tune_classifiers

        self.text_head = nn.Sequential(
            nn.Linear(self.text_features, 512),
            nn.Linear(512, 256),
            nn.Linear(256, num_classes),
            nn.Softmax(dim=1),
        )

        self.audio_head = nn.Sequential(
            nn.Linear(self.speech_features, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        self.alpha = nn.Parameter(torch.tensor(alpha))

    def train_(self):
        self.text_model.eval()
        self.speech_model.eval()
        if self.tune_classifiers:
            self.text_head.train()
            self.audio_head.train()
        else:
            self.text_head.eval()
            self.audio_head.eval()
        self.alpha.requires_grad = True

    def eval_(self):
        self.text_model.eval()
        self.speech_model.eval()
        self.text_head.eval()
        self.audio_head.eval()
        self.alpha.requires_grad = False

    def trainable_parameters(self):
        return (
            list(self.text_head.parameters())
            + list(self.audio_head.parameters())
            + [self.alpha]
        )

    def load_(self, text_path, audio_path):
        self.text_head.load_state_dict(torch.load(text_path, weights_only=True))
        self.audio_head.load_state_dict(torch.load(audio_path, weights_only=True))

    def save_(self, path):
        torch.save(self.text_head.state_dict(), path.replace(".pth", "_text_head.pth"))
        torch.save(
            self.audio_head.state_dict(), path.replace(".pth", "_audio_head.pth")
        )
        torch.save(self.alpha, path.replace(".pth", "_alpha.pth"))

    def remove(self, path):
        os.remove(path.replace(".pth", "_text_head.pth"))
        os.remove(path.replace(".pth", "_audio_head.pth"))
        os.remove(path.replace(".pth", "_alpha.pth"))

    def forward(self, text_input, speech_input, class_idx):
        with torch.no_grad():
            text_features = self.text_model(text_input, mode="classification")
            speech_features = self.speech_model(speech_input, mode="classification")
            if not self.tune_classifiers:
                y_text = self.text_head(text_features)[:, class_idx]
                y_audio = self.audio_head(speech_features)

        if self.tune_classifiers:
            y_text = self.text_head(text_features)[:, class_idx]
            y_audio = self.audio_head(speech_features)

        y = self.alpha * y_audio + (1 - self.alpha) * y_text

        return y


class LateFuse(nn.Module):
    def __init__(
        self, text_model, speech_model, text_features, speech_features, alpha=0.5
    ):
        super(LateFuse, self).__init__()
        self.text_model = text_model
        self.speech_model = speech_model
        self.text_features = text_features
        self.speech_features = speech_features

        self.text_head = nn.Sequential(
            nn.Linear(self.text_features, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        self.audio_head = nn.Sequential(
            nn.Linear(self.speech_features, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        self.alpha = nn.Parameter(torch.tensor(alpha))

    def train_(self):
        self.text_model.eval()
        self.speech_model.eval()
        self.text_head.train()
        self.audio_head.train()
        self.alpha.requires_grad = True

    def eval_(self):
        self.text_model.eval()
        self.speech_model.eval()
        self.text_head.eval()
        self.audio_head.eval()
        self.alpha.requires_grad = False

    def trainable_parameters(self):
        return (
            list(self.text_head.parameters())
            + list(self.audio_head.parameters())
            + [self.alpha]
        )

    def load_(self, text_path, audio_path):
        self.text_head.load_state_dict(torch.load(text_path, weights_only=True))
        self.audio_head.load_state_dict(torch.load(audio_path, weights_only=True))

    def save_(self, path):
        torch.save(self.text_head.state_dict(), path.replace(".pth", "_text_head.pth"))
        torch.save(
            self.audio_head.state_dict(), path.replace(".pth", "_audio_head.pth")
        )
        torch.save(self.alpha, path.replace(".pth", "_alpha.pth"))

    def remove(self, path):
        os.remove(path.replace(".pth", "_text_head.pth"))
        os.remove(path.replace(".pth", "_audio_head.pth"))
        os.remove(path.replace(".pth", "_alpha.pth"))

    def forward(self, text_input, speech_input):
        with torch.no_grad():
            text_features = self.text_model(text_input, mode="classification")
            speech_features = self.speech_model(speech_input, mode="classification")
        y_text = self.text_head(text_features)
        y_audio = self.audio_head(speech_features)

        y = self.alpha * y_audio + (1 - self.alpha) * y_text

        return y


class EarWorm(torch.nn.Module):
    def __init__(self, speech_model, speech_features, train_encoder=True):
        super(EarWorm, self).__init__()
        self.speech_model = speech_model
        self.speech_features = speech_features
        self.train_encoder = train_encoder

        self.audio_head = nn.Sequential(
            nn.BatchNorm1d(self.speech_features),
            nn.Linear(self.speech_features, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def trainable_parameters(self):
        return self.audio_head.parameters()

    def train_(self):
        self.speech_model.train()
        if not self.train_encoder:
            self.speech_model.eval()
        self.audio_head.train()

    def eval_(self):
        self.speech_model.eval()
        self.audio_head.eval()

    def save_(self, path):
        torch.save(self.audio_head.state_dict(), path)
        if self.train_encoder:
            self.speech_model.save_(path.replace(".pth", "_speech_embedder.pth"))

    def load_(self, path):
        self.audio_head.load_state_dict(torch.load(path, weights_only=True))

    def forward(self, text_input, speech_input):
        if not self.train_encoder:
            with torch.no_grad():
                speech_features = self.speech_model(speech_input, mode="classification")
        else:
            speech_features = self.speech_model(speech_input, mode="classification")
        y_audio = self.audio_head(speech_features)
        return y_audio


class BookWorm(torch.nn.Module):
    def __init__(self, text_model, text_features):
        super(BookWorm, self).__init__()
        self.text_model = text_model
        self.text_features = text_features

        self.text_head = nn.Sequential(
            nn.Linear(self.text_features, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def trainable_parameters(self):
        return self.text_head.parameters()

    def train_(self):
        self.text_model.eval()
        self.text_head.train()

    def eval_(self):
        self.text_model.eval()
        self.text_head.eval()

    def save_(self, path):
        torch.save(self.text_head.state_dict(), path)

    def load_(self, path):
        self.text_head.load_state_dict(torch.load(path, weights_only=True))

    def forward(self, text_input, speech_input):
        with torch.no_grad():
            text_features = self.text_model(text_input, mode="classification")
        y_text = self.text_head(text_features)
        return y_text


class FrozenConditionalLateFuse(nn.Module):
    def __init__(
        self,
        text_classifier,
        speech_classifier,
        alpha=0.5,
    ):
        super(FrozenConditionalLateFuse, self).__init__()

        self.text_classifier = text_classifier
        self.speech_classifier = speech_classifier

        self.alpha = nn.Parameter(torch.tensor(alpha))

    def train_(self):
        self.text_classifier.eval()
        self.speech_classifier.eval_()
        self.alpha.requires_grad = True

    def eval_(self):
        self.text_classifier.eval()
        self.speech_classifier.eval_()
        self.alpha.requires_grad = False

    def trainable_parameters(self):
        return [self.alpha]

    def load_(self, path):
        self.alpha = torch.load(path.replace(".pth", "_alpha.pth"))

    def save_(self, path):
        torch.save(self.alpha, path.replace(".pth", "_alpha.pth"))

    def remove(self, path):
        os.remove(path.replace(".pth", "_alpha.pth"))

    def forward(self, text_input, speech_input, class_idx):
        with torch.no_grad():
            y_text = self.text_classifier(text_input)
            y_text = torch.nn.functional.softmax(y_text, dim=1)
            y_text = y_text.gather(1, class_idx.unsqueeze(1)).squeeze(1)
            y_audio = self.speech_classifier(text_input, speech_input).squeeze()

        y = self.alpha * y_audio + (1 - self.alpha) * y_text
        return y


class MidFuseWithGradientReversal(nn.Module):
    def __init__(
        self,
        text_model,
        speech_model,
        text_features,
        speech_features,
        alpha=0.5,
        beta=1.0,
    ):
        super(MidFuseWithGradientReversal, self).__init__()
        self.text_model = text_model
        self.speech_model = speech_model
        self.text_features = text_features
        self.speech_features = speech_features

        self.speech_feat_pass = nn.Sequential(
            nn.BatchNorm1d(self.speech_features),
            nn.Linear(self.speech_features, self.speech_features),
        )

        self.text_feat_pass = nn.Sequential(
            nn.BatchNorm1d(self.text_features),
            nn.Linear(self.text_features, self.text_features),
        )

        self.fused_classifier = nn.Sequential(
            nn.Linear(self.text_features + self.speech_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        self.speech_classifier = nn.Sequential(
            nn.Linear(self.speech_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        self.alpha = alpha
        self.beta = beta

    def train_(self):
        self.text_model.eval()
        self.speech_model.eval()
        self.fused_classifier.train()
        self.speech_classifier.train()
        self.speech_feat_pass.train()
        self.text_feat_pass.train()

    def eval_(self):
        self.text_model.eval()
        self.speech_model.eval()
        self.fused_classifier.eval()
        self.speech_classifier.eval()
        self.speech_feat_pass.eval()
        self.text_feat_pass.eval()

    def remove(self, path):
        os.remove(path)
        os.remove(path.replace(".pth", "_speech_feat_pass.pth"))
        os.remove(path.replace(".pth", "_text_feat_pass.pth"))
        os.remove(path.replace(".pth", "_speech_classifier.pth"))

    def trainable_parameters(self):
        return (
            list(self.fused_classifier.parameters())
            + list(self.speech_feat_pass.parameters())
            + list(self.text_feat_pass.parameters())
            + list(self.speech_classifier.parameters())
        )

    def load_(self, path):
        self.fused_classifier.load_state_dict(torch.load(path, weights_only=True))
        self.speech_feat_pass.load_state_dict(
            torch.load(path.replace(".pth", "_speech_feat_pass.pth"), weights_only=True)
        )
        self.text_feat_pass.load_state_dict(
            torch.load(path.replace(".pth", "_text_feat_pass.pth"), weights_only=True)
        )

    def save_(self, path):
        torch.save(self.fused_classifier.state_dict(), path)
        torch.save(
            self.speech_feat_pass.state_dict(),
            path.replace(".pth", "_speech_feat_pass.pth"),
        )
        torch.save(
            self.text_feat_pass.state_dict(),
            path.replace(".pth", "_text_feat_pass.pth"),
        )
        torch.save(
            self.speech_classifier.state_dict(),
            path.replace(".pth", "_speech_classifier.pth"),
        )

    def forward(self, text_input, speech_input, mode="inference"):
        with torch.no_grad():
            text_features = self.text_model(text_input, mode="classification")
            speech_features = self.speech_model(speech_input, mode="classification")
        text_features = self.text_feat_pass(text_features)
        speech_features = self.speech_feat_pass(speech_features)
        features = torch.cat([text_features, speech_features], dim=1)
        y_fuse = self.fused_classifier(features)
        if mode == "inference":
            return y_fuse
        y_speech = self.speech_classifier(speech_features)
        return y_fuse, y_speech
