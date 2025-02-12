"""
... 

TODO @abhaydmathur ; 
 - Add interpretable losses when training on ada_triplet OR make ada_triplet interpretable across varying margins???
 - generalise computation and logging for arbitrary metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import glob
from tqdm import tqdm
from torch import autograd
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.utils.data import Dataset, ConcatDataset
from torch.nn import TripletMarginLoss, TripletMarginWithDistanceLoss
from utils.datasets import (
    InTheWildDataset,
    ASVSpoof21Dataset,
    VoxCeleb2Dataset,
    CommonVoiceDataset,
    RAVDESSDataset,
)

from torch.utils.tensorboard import SummaryWriter

from torch.optim.lr_scheduler import (  # TODO @abhaydmathur : Add more schedulers, move all schedulers to callbacks somehow?
    ReduceLROnPlateau,
    CosineAnnealingWarmRestarts,
    StepLR,
)


from utils.losses import (
    TripletMarginCosineLoss,
    CosineDistance,
    TripletSquaredSimilarity,
    AdaTriplet,
    SquaredSimilarity,
)
from utils.metrics import ABXAccuracy
from utils.logging import Logger
from utils.visualisation import get_tsne_img, get_tsne_fig

from models import SpeechEmbedder
import yaml


class SpeechCLRTrainerVanilla:
    def __init__(self, args):
        self.args = args
        self.model = SpeechEmbedder()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=args.learning_rate, weight_decay=0.01
        )

        if self.args.load_path is not None:
            print(f"Loading {self.args.model_name} weights from {self.args.load_path}")
            if self.args.load_checkpoint == "best":
                load_ckpt = os.path.join(self.args.load_path, "best_model.pth")
            elif self.args.load_checkpoint == "latest":
                if self.args.load_path[-1] == "/":
                    self.args.load_path = self.args.load_path[:-1]
                load_ckpts = glob.glob(f"{self.args.load_path}/latest_model_*.pth")
                if len(load_ckpts) == 0:
                    raise FileNotFoundError(
                        f"No checkpoints found in {self.args.load_path}"
                    )
                load_ckpt = load_ckpts[0]
            else:
                raise NotImplementedError(
                    f"Checkpoint {self.args.load_checkpoint} not implemented"
                )
            self.model.load_(load_ckpt)

        self.loss_to_data_mode = {
            "triplet": "triplet",
            "triplet_cosine": "triplet",
            "ada_triplet": "triplet",
            "cross_entropy": "classification",
            "squared_similarity": "triplet",
        }

        if args.loss_fn == "triplet":
            self.criterion = TripletMarginLoss(margin=self.args.margin)
        elif args.loss_fn == "triplet_cosine":
            self.criterion = TripletMarginCosineLoss(margin=self.args.margin)
        elif args.loss_fn == "ada_triplet":
            self.criterion = AdaTriplet(lambda_=self.args.at_lambda)
        elif args.loss_fn == "squared_similarity":
            self.criterion = TripletSquaredSimilarity()
        else:
            raise NotImplementedError(f"Loss function {args.loss_fn} not implemented")

        self.metrics_fn = ABXAccuracy()

        self.data_mode = self.loss_to_data_mode[args.loss_fn]

        self.train_datasets_list = []
        self.train_vis_datasets_list = []
        self.val_datasets_list = []
        self.vis_datasets_list = []

        if "inthewild" in self.args.datasets:

            self.train_datasets.append(
                InTheWildDataset(
                    root_dir=os.path.join(args.data_path, "InTheWild"),
                    metadata_file="meta.csv",
                    include_spoofs=False,
                    bonafide_label="bona-fide",
                    filename_col="file",
                    sampling_rate=args.sampling_rate,
                    max_duration=args.max_duration,
                    split="train",
                    config=args.dataset_config,
                    mode=self.data_mode,
                )
            )

            self.val_datasets_list.append(
                InTheWildDataset(
                    root_dir=os.path.join(args.data_path, "InTheWild"),
                    metadata_file="meta.csv",
                    include_spoofs=False,
                    bonafide_label="bona-fide",
                    filename_col="file",
                    sampling_rate=args.sampling_rate,
                    max_duration=args.max_duration,
                    split="val",
                    config=args.dataset_config,
                    mode=self.data_mode,
                )
            )

            self.train_vis_datasets_list.append(
                InTheWildDataset(
                    root_dir=os.path.join(args.data_path, "InTheWild"),
                    metadata_file="meta.csv",
                    include_spoofs=False,
                    bonafide_label="bona-fide",
                    filename_col="file",
                    sampling_rate=args.sampling_rate,
                    max_duration=args.max_duration,
                    split="train",
                    config=args.dataset_config,
                    mode="classification",
                )
            )

            self.vis_datasets_list.append(
                InTheWildDataset(
                    root_dir=os.path.join(args.data_path, "InTheWild"),
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
            )

        if "asvspoof" in self.args.datasets:
            raise NotImplementedError

        if "voxceleb2" in self.args.datasets:
            self.train_datasets_list.append(
                VoxCeleb2Dataset(
                    root_dir=os.path.join(args.data_path, "VoxCeleb2"),
                    split="train",
                    sampling_rate=args.sampling_rate,
                    max_duration=args.max_duration,
                    mode=self.data_mode,
                    max_samples=3000,
                )
            )

            self.val_datasets_list.append(
                VoxCeleb2Dataset(
                    root_dir=os.path.join(args.data_path, "VoxCeleb2"),
                    split="val",
                    sampling_rate=args.sampling_rate,
                    max_duration=args.max_duration,
                    mode=self.data_mode,
                    max_samples=1000,
                )
            )

            self.train_vis_datasets_list.append(
                VoxCeleb2Dataset(
                    root_dir=os.path.join(args.data_path, "VoxCeleb2"),
                    split="train",
                    sampling_rate=args.sampling_rate,
                    max_duration=args.max_duration,
                    mode="classification",
                    max_samples=1000,
                )
            )

            self.vis_datasets_list.append(
                VoxCeleb2Dataset(
                    root_dir=os.path.join(args.data_path, "VoxCeleb2"),
                    split="val",
                    sampling_rate=args.sampling_rate,
                    max_duration=args.max_duration,
                    mode="classification",
                    max_samples=1000,
                )
            )

        if "commonvoice" in self.args.datasets:
            self.train_datasets_list.append(
                CommonVoiceDataset(
                    root_dir=os.path.join(args.data_path, "CommonVoice"),
                    split="train",
                    sampling_rate=args.sampling_rate,
                    max_duration=args.max_duration,
                    mode=self.data_mode,
                )
            )

            self.val_datasets_list.append(
                CommonVoiceDataset(
                    root_dir=os.path.join(args.data_path, "CommonVoice"),
                    split="val",
                    sampling_rate=args.sampling_rate,
                    max_duration=args.max_duration,
                    mode=self.data_mode,
                )
            )

            self.train_vis_datasets_list.append(
                CommonVoiceDataset(
                    root_dir=os.path.join(args.data_path, "CommonVoice"),
                    split="train",
                    sampling_rate=args.sampling_rate,
                    max_duration=args.max_duration,
                    mode="classification",
                )
            )

            self.vis_datasets_list.append(
                CommonVoiceDataset(
                    root_dir=os.path.join(args.data_path, "CommonVoice"),
                    split="val",
                    sampling_rate=args.sampling_rate,
                    max_duration=args.max_duration,
                    mode="classification",
                )
            )

        if "ravdess" in self.args.datasets:
            self.train_datasets_list.append(
                RAVDESSDataset(
                    root_dir=os.path.join(args.data_path, "RAVDESS"),
                    split="train",
                    sampling_rate=args.sampling_rate,
                    max_duration=args.max_duration,
                    mode=self.data_mode,
                )
            )

            self.val_datasets_list.append(
                RAVDESSDataset(
                    root_dir=os.path.join(args.data_path, "RAVDESS"),
                    split="val",
                    sampling_rate=args.sampling_rate,
                    max_duration=args.max_duration,
                    mode=self.data_mode,
                )
            )

            self.train_vis_datasets_list.append(
                RAVDESSDataset(
                    root_dir=os.path.join(args.data_path, "RAVDESS"),
                    split="train",
                    sampling_rate=args.sampling_rate,
                    max_duration=args.max_duration,
                    mode="classification",
                )
            )

            self.vis_datasets_list.append(
                RAVDESSDataset(
                    root_dir=os.path.join(args.data_path, "RAVDESS"),
                    split="val",
                    sampling_rate=args.sampling_rate,
                    max_duration=args.max_duration,
                    mode="classification",
                )
            )

        if len(self.train_datasets_list) == 0:
            raise ValueError("No datasets found")

        self.train_dataset = ConcatDataset(self.train_datasets_list)
        self.val_dataset = ConcatDataset(self.val_datasets_list)
        self.train_vis_dataset = ConcatDataset(self.train_vis_datasets_list)
        self.vis_dataset = ConcatDataset(self.vis_datasets_list)

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=args.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=args.batch_size, shuffle=True
        )
        self.train_vis_loader = DataLoader(
            self.train_vis_dataset, batch_size=args.batch_size, shuffle=True
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

        if self.args.lr_scheduler == "plateau":
            self.lr_scheduler = ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=3, verbose=True
            )
        elif self.args.lr_scheduler == "cosine":
            self.lr_scheduler = CosineAnnealingWarmRestarts(
                self.optimizer, T_0=10, T_mult=2
            )
        elif self.args.lr_scheduler == "step":
            self.lr_scheduler = StepLR(self.optimizer, step_size=10, gamma=0.1)
        else:
            raise NotImplementedError(
                f"LR scheduler {args.lr_scheduler} not implemented"
            )

        # Ensure self.args.model_save_path exists
        self.model_name = self.args.model_name
        self.model_save_path = os.path.join(
            self.args.model_save_path, self.model_name, self.args.log_dir.split("/")[-1]
        )
        os.makedirs(self.model_save_path, exist_ok=True)

        self.best_model_path = None
        self.latest_model_path = None
        self.best_loss = np.inf
        self.best_acc = 0
        self.best_epoch = 0

        print(f"Training on device: {self.device}")
        print(f"Saving logs to {args.log_dir}")
        print(f"Saving models to {self.model_save_path}")
        print(f"Loss function: {args.loss_fn}")
        print(f"Training Samples : {len(self.train_dataset)}")
        print(f"Validation Samples : {len(self.val_dataset)}")

        args_dict = vars(self.args)
        args_save_path = os.path.join(self.model_save_path, "config.yaml")
        with open(args_save_path, "w") as file:
            yaml.dump(args_dict, file)

    def log_init(self):

        print("Getting initial metrics.")

        train_info = self.validate(split="train", verbose=True, limit_samples=True)

        if isinstance(self.criterion, AdaTriplet):
            train_info["ada_triplet/eps"] = self.criterion.eps
            train_info["ada_triplet/beta"] = self.criterion.beta

        log_train_info = {f"train/{k}": v for k, v in train_info.items()}
        self.save_to_log(self.args.log_dir, self.logger, log_train_info, 0)

        val_info = self.validate(split="val", verbose=True, limit_samples=True)

        log_val_info = {f"val/{k}": v for k, v in val_info.items()}
        self.save_to_log(self.args.log_dir, self.logger, log_val_info, 0)

        self.training_history[0] = {
            "train": train_info,
            "val": val_info,
        }

        if self.args.save_visualisations:
            self.log_cluster_visualisation(0, split="val", n_samples=1000)
            self.log_cluster_visualisation(0, split="train", n_samples=1000)

    def train(self):
        self.log_init()
        self.model.train()
        for epoch in range(self.args.epochs):
            epoch_info = self.train_epoch(epoch)
            log_epoch_info = {f"train/{k}": v for k, v in epoch_info.items()}
            self.save_to_log(self.args.log_dir, self.logger, log_epoch_info, epoch + 1)

            val_info = self.validate()
            log_val_info = {f"val/{k}": v for k, v in val_info.items()}
            self.save_to_log(self.args.log_dir, self.logger, log_val_info, epoch + 1)

            if self.args.lr_scheduler == "plateau":
                self.lr_scheduler.step(val_info["loss"])
            elif self.args.lr_scheduler == "step":
                self.lr_scheduler.step()

            self.training_history[epoch + 1] = {
                "train": epoch_info,
                "val": val_info,
            }

            # Save best and latest models

            if self.best_model_path is None or self.is_best_model(val_info):
                self.best_model_path = os.path.join(
                    self.model_save_path, "best_model.pth"
                )
                self.save(self.best_model_path)
                self.best_epoch = epoch

            if self.latest_model_path is not None:
                try:
                    os.remove(self.latest_model_path)
                except Exception as e:
                    print(e)
            self.latest_model_path = os.path.join(
                self.model_save_path, f"latest_model_{epoch+1}eps.pth"
            )
            self.save(self.latest_model_path)

            if self.execute_callbacks(epoch):
                break

            if self.args.save_visualisations:
                self.log_cluster_visualisation(epoch + 1, split="train", n_samples=1000)
                self.log_cluster_visualisation(epoch + 1, split="val", n_samples=1000)

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

            if self.args.lr_scheduler == "cosine":
                self.lr_scheduler.step(epoch + i / n_steps)

            losses.append(loss.item())
            print(
                f"\rEpoch {epoch + 1} [{i + 1}/{n_steps}] loss: {np.mean(losses):.3f}    ",
                end="",
            )
        print()

        info = {
            "loss": np.mean(losses),
            "lr": self.optimizer.param_groups[0]["lr"],
        }

        if isinstance(self.criterion, AdaTriplet):
            info["ada_triplet/eps"] = self.criterion.eps
            info["ada_triplet/beta"] = self.criterion.beta

        return info

    def validate(self, split="val", verbose=True, limit_samples=False):
        self.model.eval()
        losses = []
        accs = []

        if split == "train":
            loader = self.train_loader
        elif split == "val":
            loader = self.val_loader
        else:
            raise ValueError("Invalid split")
        n_batches = len(loader)
        if limit_samples:
            n_samples = 1000
            n_batches = n_samples // loader.batch_size

        with torch.no_grad():
            for i, input in enumerate(loader):
                if self.data_mode == "triplet":
                    input = {k: v.to(self.device) for k, v in input.items()}
                elif self.data_mode == "classification":
                    input["x"] = input["x"].to(self.device)
                    input["label"] = input["label"].to(self.device)
                elif self.data_mode == "pair":
                    input["a"] = input["a"].to(self.device)
                    input["b"] = input["b"].to(self.device)
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
                        f"\rEvaluation on {split} [{i+1}/{n_batches}]: loss: {np.mean(losses):.3f}, abx_acc: {np.mean(accs):.3f}     ",
                        end="",
                    )
                if limit_samples and (i + 1) >= n_batches:
                    break
        if verbose:
            print()
        return {"loss": np.mean(losses), "abx_acc": np.mean(accs)}

    def log_cluster_visualisation(self, epoch, split="val", n_samples=1000):
        self.model.eval()
        feats = []
        labels = []
        if split == "train":
            loader = self.train_vis_loader
        elif split == "val":
            loader = self.vis_loader
        n_batches = n_samples // loader.batch_size
        with torch.no_grad():
            for i, input in enumerate(loader):
                input["x"] = input["x"].to(self.device)
                batch_feats = self.model(input, mode="classification")
                batch_labels = input["author"]
                feats.append(batch_feats)
                labels.extend(batch_labels)
                print(f"\rGetting embeddings for {split} [{i+1}/{n_batches}]", end="")
                if i > n_batches:
                    break
        print()
        feats = torch.cat(feats, dim=0).cpu().numpy()
        labels = np.array(labels)

        print(f"Getting t-SNE plot for {split} set")
        fig = get_tsne_fig(feats, labels, f"t-SNE on {split} : Epoch {epoch}")
        plt.savefig(f"{self.args.log_dir}/tsne_{split}_{epoch}.png")
        self.logger.figure_summary(f"{split}/tsne", fig, epoch)
        plt.close()

    def save(self, path):
        self.model.save_(path)

    def execute_callbacks(self, epoch):
        # Early Stopping
        if self.args.early_stopping_patience is not None:
            if epoch - self.best_epoch > self.args.early_stopping_patience:
                print(f"Early Stopping after {epoch} epochs")
                return True
        return False

    def is_best_model(self, val_info):
        if self.args.early_stopping_metric == "loss":
            k = val_info["loss"] < self.best_loss
            if k:
                self.best_loss = val_info["loss"]
            return k
        elif self.args.early_stopping_metric == "accuracy":
            k = val_info["abx_acc"] > self.best_acc
            if k:
                self.best_acc = val_info["abx_acc"]
            return k
