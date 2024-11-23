"""
... 

TODO @abhaydmathur ; 
 - debug adatriplet
 - generalise computation and logging for arbitrary metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os

from torch import autograd

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.nn import TripletMarginLoss, TripletMarginWithDistanceLoss
from utils.datasets import InTheWildDataset, ASVSpoof21Dataset
from torch.utils.tensorboard import SummaryWriter

from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback
from utils.losses import (
    TripletMarginCosineLoss,
    CosineDistance,
    AdaTriplet,
    SquaredSimilarity,
)
from utils.metrics import ABXAccuracy
from utils.logging import Logger
from models import SpeechEmbedder


class SpeechCLRTrainerVanilla:
    def __init__(self, args):
        self.args = args
        self.model = SpeechEmbedder()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=args.learning_rate, weight_decay=0.01
        )
        
        self.loss_to_data_mode = {
            "triplet": "triplet",
            "triplet_cosine": "triplet",
            "ada_triplet": "triplet",
            "cross_entropy": "classification",
            "squared_similarity": "pair",
        }

        if args.loss_fn == "triplet":
            self.criterion = TripletMarginLoss(margin=self.args.margin)
        elif args.loss_fn == "triplet_cosine":
            self.criterion = TripletMarginCosineLoss(margin=self.args.margin)
        elif args.loss_fn == "ada_triplet":
            self.criterion = AdaTriplet(lambda_=self.args.at_lambda)
        elif args.loss_fn == "squared_similarity":
            self.criterion = SquaredSimilarity()
        else:
            raise NotImplementedError(f"Loss function {args.loss_fn} not implemented")

        self.metrics_fn = ABXAccuracy()

        self.train_dataset = InTheWildDataset(
            root_dir=args.data_path,
            metadata_file="meta.csv",
            include_spoofs=False,
            bonafide_label="bona-fide",
            filename_col="file",
            sampling_rate=args.sampling_rate,
            max_duration=args.max_duration,
            split="train",
            config=args.dataset_config,
            mode=self.loss_to_data_mode[args.loss_fn],
        )

        self.val_dataset = InTheWildDataset(
            root_dir=args.data_path,
            metadata_file="meta.csv",
            include_spoofs=False,
            bonafide_label="bona-fide",
            filename_col="file",
            sampling_rate=args.sampling_rate,
            max_duration=args.max_duration,
            split="val",
            config=args.dataset_config,
            mode=self.loss_to_data_mode[args.loss_fn],
        )

        self.vis_dataset = InTheWildDataset(
            root_dir=args.data_path,
            metadata_file="meta.csv",
            include_spoofs=False,
            bonafide_label="bona-fide",
            filename_col="file",
            sampling_rate=args.sampling_rate,
            max_duration=args.max_duration,
            split="val",
            config=args.dataset_config,
            mode="classification",
        )

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=args.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=args.batch_size, shuffle=True
        )

        self.logger = Logger(log_dir=args.log_dir)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def train(self):
        self.model.train()
        for epoch in range(self.args.epochs):
            epoch_info = self.train_epoch(epoch)
            epoch_info = {f"train/{k}": v for k, v in epoch_info.items()}
            self.save_to_log(self.args.log_dir, self.logger, epoch_info, epoch)

            val_info = self.validate()
            val_info = {f"val/{k}": v for k, v in val_info.items()}
            self.save_to_log(self.args.log_dir, self.logger, val_info, epoch)

    # @staticmethod
    def save_to_log(self, logdir, logger, info, epoch, w_summary=False, model=None):
        # save scalars
        for tag, value in info.items():
            logger.scalar_summary(tag, value, epoch)

        # save summaries of weights and biases
        if w_summary and model:
            for tag, value in model.named_parameters():
                tag = tag.replace(".", "/")
                try:
                    logger.histo_summary(tag, value.data.cpu().numpy(), epoch)
                except:
                    continue
                    logger.histo_summary(tag, value.data, epoch)
                if value.grad is not None:
                    logger.histo_summary(
                        tag + "/grad", value.grad.data.cpu().numpy(), epoch
                    )

    def train_epoch(self, epoch, verbose=True):
        running_loss = 0.0
        # if self.args.loss_fn == "ada_triplet":
        #     self.criterion.reset()
        losses = []
        for i, input in enumerate(self.train_loader):

            input = {k: v.to(self.device) for k, v in input.items()}
            output = self.model(input)

            self.optimizer.zero_grad()
            autograd.set_detect_anomaly(True)
            with autograd.detect_anomaly():
                loss = self.criterion(
                    output["anchor"], output["positive"], output["negative"]
                )
                
                loss.backward(retain_graph=self.args.loss_fn == "ada_triplet")
            self.optimizer.step()
            self.optimizer.zero_grad()
            running_loss = running_loss + loss.item()
            losses.append(loss.item())
            if i % 100 == 99 and verbose:
                print(f"\r[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}", end="")
                running_loss = 0.0
        print()
        return {
            "loss": np.mean(losses),
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    def validate(self, verbose=True):
        self.model.eval()
        correct = 0
        total = 0
        losses = []
        accs = []
        with torch.no_grad():
            for i, input in enumerate(self.val_loader):
                input = {k: v.to(self.device) for k, v in input.items()}
                output = self.model(input)
                loss = self.criterion(
                    output["anchor"], output["positive"], output["negative"]
                )
                abx_acc = self.metrics_fn(
                    output["anchor"], output["positive"], output["negative"]
                )

                losses.append(loss.item())
                accs.append(abx_acc)
                if verbose:
                    print(
                        f"\rValidation : {i+1}/{len(self.val_loader)}: loss: {loss.item():.3f}, abx_acc: {abx_acc:.3f}     ",
                        end="",
                    )
        if verbose: print()
        return {"loss": np.mean(losses), "abx_acc": np.mean(accs)}
    
    def visualise_clusters(self, n_samples=4000):

        self.model.eval()
        with torch.no_grad():
            for i, input in enumerate(self.val_loader):
                input = {k: v.to(self.device) for k, v in input.items()}
                output = self.model(input)
                break
        return output
