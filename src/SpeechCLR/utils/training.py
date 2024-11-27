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
from tqdm import tqdm
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
from utils.visualisation import get_tsne_img, get_tsne_fig

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
        self.vis_loader = DataLoader(
            self.vis_dataset, batch_size=args.batch_size, shuffle=True
        )

        self.logger = Logger(log_dir=args.log_dir)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        self.max_init_steps = 500

        self.training_history = {}

        self.callbacks = []

        print(f"Training on device: {self.device}")
        print(f"Saving logs to {args.log_dir}")
        print(f"Loss function: {args.loss_fn}")
        print(f"Training Samples : {len(self.train_dataset)}")
        print(f"Validation Samples : {len(self.val_dataset)}")

    def log_init(self):

        print("Getting initial metrics.")
        # Get loss on train set
        print("Train:")
        self.model.eval()
        train_losses = train_accs = []
        with torch.no_grad():
            for i, input in tqdm(enumerate(self.train_loader)):
                input = {k: v.to(self.device) for k, v in input.items()}
                output = self.model(input)
                loss = self.criterion(output["anchor"], output["positive"], output["negative"], eval=True)
                # abx_acc = self.metrics_fn(
                #     output["anchor"], output["positive"], output["negative"]
                # )
                train_losses.append(loss.item())
                if i > self.max_init_steps:
                    break
                # train_accs.append(abx_acc)
        train_loss = np.mean(train_losses)
        # train_acc = np.mean(train_accs)

        train_info = {
            "loss": train_loss,
            # "train/abx_acc": train_acc,
        }        

        if isinstance(self.criterion, AdaTriplet):
            train_info["train/ada_triplet/eps"] = self.criterion.eps
            train_info["train/ada_triplet/beta"] = self.criterion.beta

        log_train_info = {f"train/{k}": v for k, v in train_info.items()}
        self.save_to_log(self.args.log_dir, self.logger, log_train_info, 0)


        print("Val:")
        # Get loss on val set
        self.model.eval()
        val_losses = val_accs = []
        with torch.no_grad():
            for i, input in tqdm(enumerate(self.val_loader)):
                input = {k: v.to(self.device) for k, v in input.items()}
                output = self.model(input)
                loss = self.criterion(output["anchor"], output["positive"], output["negative"], eval=True)
                abx_acc = self.metrics_fn(
                    output["anchor"], output["positive"], output["negative"]
                )
                val_losses.append(loss.item())
                val_accs.append(abx_acc)
        val_loss = np.mean(val_losses)
        val_acc = np.mean(val_accs)

        # Log the initial values
        val_info = {
            "loss": val_loss,
            "abx_acc": val_acc,
        }

        log_val_info = {f"val/{k}": v for k, v in val_info.items()} 
        self.save_to_log(self.args.log_dir, self.logger, val_info, 0)

        self.training_history[0] = {
            "train": train_info,
            "val": val_info,
        }

        if self.args.save_visualisations:
            self.log_cluster_visualisation(0, split='val', n_samples=1000)

    def train(self):
        self.log_init()
        self.model.train()
        for epoch in range(self.args.epochs):
            epoch_info = self.train_epoch(epoch)
            log_epoch_info = {f"train/{k}": v for k, v in epoch_info.items()}
            self.save_to_log(self.args.log_dir, self.logger, log_epoch_info, epoch+1)

            val_info = self.validate()
            log_val_info = {f"val/{k}": v for k, v in val_info.items()}
            self.save_to_log(self.args.log_dir, self.logger, log_val_info, epoch+1)

            self.training_history[epoch] = {
                "train": epoch_info,
                "val": val_info,
            }

            if self.args.save_visualisations:
                self.log_cluster_visualisation(epoch, split='val', n_samples=1000)

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
        self.model.train()
        losses = []
        n_steps = len(self.train_loader)
        for i, input in enumerate(self.train_loader):

            input = {k: v.to(self.device) for k, v in input.items()}
            output = self.model(input)

            self.optimizer.zero_grad()
            loss = self.criterion(
                output["anchor"], output["positive"], output["negative"]
            )
                
            loss.backward(retain_graph=self.args.loss_fn == "ada_triplet")
            self.optimizer.step()
            self.optimizer.zero_grad()
            losses.append(loss.item())
            if i % 100 == 99 and verbose:
                print(f"\rEpoch {epoch + 1} [{i + 1}/{n_steps}] loss: {np.mean(losses):.3f}    ", end="")
        print()

        info = {
            "loss": np.mean(losses),
            "lr": self.optimizer.param_groups[0]["lr"],
        }

        if isinstance(self.criterion, AdaTriplet):
            info["ada_triplet/eps"] = self.criterion.eps
            info["ada_triplet/beta"] = self.criterion.beta

        return info

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
                    output["anchor"], output["positive"], output["negative"], eval=True
                )
                abx_acc = self.metrics_fn(
                    output["anchor"], output["positive"], output["negative"]
                )

                losses.append(loss.item())
                accs.append(abx_acc)
                if verbose:
                    print(
                        f"\rValidation : {i+1}/{len(self.val_loader)}: loss: {np.mean(losses):.3f}, abx_acc: {np.mean(accs):.3f}     ",
                        end="",
                    )
        if verbose: print()
        return {"loss": np.mean(losses), "abx_acc": np.mean(accs)}
    
    def log_cluster_visualisation(self, epoch, split = 'val', n_samples=1000):
        self.model.eval()
        feats = []
        labels = []
        n_batches = n_samples // self.vis_loader.batch_size
        with torch.no_grad():
            for i, input in enumerate(self.vis_loader[:n_batches]):
                input = {k: v.to(self.device) for k, v in input.items()}
                batch_feats = self.model(input, mode='classification')
                batch_labels = input['label']
                feats.append(batch_feats)
                labels.extend(batch_labels.flatten())
                break
        feats = torch.cat(feats, dim=0).cpu().numpy()
        labels = np.array(labels)

        img = get_tsne_img(feats, labels, f"Epoch {epoch} {split} set")
        self.logger.image_summary(f"{split}/tsne", img, epoch)
        pass
