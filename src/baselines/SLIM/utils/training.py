import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os

from torch.utils.data import DataLoader
from torch.utils.data import Dataset, ConcatDataset

from utils.datasets import CommonVoiceDataset, RAVDESSDataset, InTheWildDataset, ASVSpoof21Dataset
from torch.utils.tensorboard import SummaryWriter

from transformers import Trainer, TrainingArguments
from transformers import TrainerCallback

from models import SLIMStage1, SLIMStage2

from torch.optim.lr_scheduler import (
    ReduceLROnPlateau, 
    CosineAnnealingWarmRestarts,
    StepLR,
)

from utils.logging import Logger
from utils.losses import SelfContrastiveLoss


class SLIMTrainer():
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f"cuda:{self.args.gpu_id}" if torch.cuda.is_available() else "cpu")
        # self.model = SLIMStage1().to(self.device)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate)
        # self.loss_fn = nn.CrossEntropyLoss()
        self.writer = SummaryWriter(log_dir=args.log_dir)

        if self.args.lr_scheduler == "plateau":
            self.lr_scheduler = ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=3, verbose=True
            )
        elif self.args.lr_scheduler == "cosine":
            self.lr_scheduler = CosineAnnealingWarmRestarts(
                self.optimizer, T_0=10, T_mult=2                
            )  
        elif self.args.lr_scheduler == "step": 
            self.lr_scheduler = StepLR(
                self.optimizer, step_size=10, gamma=0.1
            )
        else:
            raise NotImplementedError(f"LR scheduler {args.lr_scheduler} not implemented")

        self.best_model_path = None
        self.latest_model_path = None
        self.best_loss = np.inf
        self.best_acc = 0
        self.best_epoch = 0

        self.logger = Logger(log_dir=args.log_dir)


        print(f"Training on device: {self.device}")
        print(f"Saving logs to {args.log_dir}")
        print(f"Saving models to {self.model_save_path}")
        print(f"Loss function: {args.loss_fn}")
        print(f"Training Samples : {len(self.train_dataset)}")
        print(f"Validation Samples : {len(self.val_dataset)}")

    def train(self):
        self.log_init()
        self.model.train_()

        for epoch in range(self.args.epochs):
            epoch_info = self.train_epoch(epoch)
            log_epoch_info = {f"train/{k}": v for k, v in epoch_info.items()}
            self.save_to_log(self.args.log_dir, self.logger, log_epoch_info, epoch+1)

            val_info = self.evaluate()
            log_val_info = {f"val/{k}": v for k, v in val_info.items()}
            self.save_to_log(self.args.log_dir, self.logger, log_val_info, epoch+1)

            if self.args.lr_scheduler == "plateau":
                self.lr_scheduler.step(val_info["loss"])
            elif self.args.lr_scheduler == "step":
                self.lr_scheduler.step()

            self.training_history[epoch+1] = {
                "train": epoch_info,
                "val": val_info,
            }

            # Save best and latest models

            if self.best_model_path is None or self.is_best_model(val_info):
                self.best_model_path = os.path.join(self.model_save_path, "best_model.pth")
                self.save(self.best_model_path)
                self.best_epoch = epoch
            
            if self.latest_model_path is not None:
                try: os.remove(self.latest_model_path)
                except Exception as e: print(e)
            self.latest_model_path = os.path.join(self.model_save_path, f"latest_model_{epoch+1}eps.pth")
            self.save(self.latest_model_path)

            if self.execute_callbacks(epoch):
                break


    def train_epoch(self):
        pass

    def log_init(self):
        pass 

    def evaluate(self):
        pass

    def save_model(self, path):
        self.model.save_(self.path)
    
    def load_model(self, path):
        self.model.load_(self.args.load_path)

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
    

class StageOneTrainer(SLIMTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.model = SLIMStage1().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()

        self.train_dataset = ConcatDataset(
            [
                CommonVoiceDataset(
                    root_dir=args.cv_dir,
                    split="train",
                ),
                RAVDESSDataset(
                    root_dir=args.ravdess_dir,
                    split="train",
                )
            ]
        )

        self.val_dataset = ConcatDataset(
            [
                CommonVoiceDataset(
                    root_dir=args.cv_dir,
                    split="val",
                ),
                RAVDESSDataset(
                    root_dir=args.ravdess_dir,
                    split="val",
                )
            ]
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        self.model_save_path = os.path.join(args.model_save_dir, args.model_name,"stage1")


    def train_epoch(self, epoch, verbose=True):
        self.model.train()

        losses = []

        n_steps = len(self.train_loader)
        for i, batch in enumerate(self.train_loader):
            
            batch = batch.to(self.device)

            self.optimizer.zero_grad()

            loss = self.model(batch).squeeze()

            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

            if verbose:
                print(
                    f"\rEpoch {epoch + 1} [{i + 1}/{n_steps}] loss: {np.mean(losses):.3f}    ",
                    end="",
                )
        if verbose : print()

        info = {
            "loss": np.mean(losses),
            "lr": self.optimizer.param_groups[0]["lr"],
        }

        return info

    def evaluate(self, verbose=True):
        self.model.eval()

        losses = []
        accs = []

        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                batch = batch.to(self.device)

                loss = self.model(batch).squeeze()

                losses.append(loss.item())

                if verbose:
                    print(
                        f"\rValidation [{i + 1}/{len(self.val_loader)}] loss: {np.mean(losses):.3f}   ",
                        end="",
                    )
        if verbose: print()

        info = {
            "loss": np.mean(losses),
        }

        return info



class StageTwoTrainer:
    pass
