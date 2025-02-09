"""
Training Module for Context Aware Audio Spoof Detection with Mid-Fusion

TODO @abhaydmathur
1. Callbacks
    - Early Stopping
    - LR Schedule
2. Metrics
    - EER
    - F1
    - Accuracy
    - t-DCF
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
from mid_fusion.utils.datasets import InTheWildDataset
from SpeechCLR.utils.logging import Logger
from SpeechCLR.models import SpeechEmbedder
from authorship_attribution.model import AuthorshipLLM
from mid_fusion.models import MidFuse

class MidFusionTrainer():
    def __init__(self, args):
        self.args = args
        self.train_dataset = InTheWildDataset(
            root_dir=args.data_path,
            metadata_file="wild_transcription_meta.json",
            include_spoofs=True,
            bonafide_label="bona-fide",
            filename_col="file",
            sampling_rate=args.sampling_rate,
            max_duration=args.max_duration,
            split="train",
            config=args.dataset_config,
            text_tokenizer_name = self.args.text_model_name,
            mode='classification',
        )

        self.val_dataset = InTheWildDataset(
            root_dir=args.data_path,
            metadata_file="wild_transcription_meta.json",
            include_spoofs=True,
            bonafide_label="bona-fide",
            filename_col="file",
            sampling_rate=args.sampling_rate,
            max_duration=args.max_duration,
            split="val",
            config=args.dataset_config,
            text_tokenizer_name = self.args.text_model_name,
            mode='classification',
        )

        print("Loaded Dataset - ")
        print(f"Training Samples : {len(self.train_dataset)}")
        print(f"Validation Samples : {len(self.val_dataset)}")

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=args.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=args.batch_size, shuffle=True
        )

        self.logger = Logger(log_dir=args.log_dir)
        print(f"Logging to {args.log_dir}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(F"Using device: {self.device}")

        self.speech_model = SpeechEmbedder()
        print(self.args.text_model_name, self.args.mlp_layers, self.args.hidden_layers)
        self.text_model = AuthorshipLLM(self.args.text_model_name, 
                          num_layers=self.args.mlp_layers, 
                          use_layers=self.args.hidden_layers,
                          out_features=512)
        
        if args.text_model_path is not None:
            checkpoint = torch.load(args.text_model_path, map_location=torch.device('cpu'), weights_only=True)
            self.text_model.load_state_dict(checkpoint['model_state_dict']) # TODO @abhaydmathur : ensure this is how Ivi loads the model
            print(f"Loaded text model from {args.text_model_path}")
        if args.speech_model_path is not None:
            self.speech_model.load_(args.speech_model_path)
            print(f"Loaded speech model from {args.speech_model_path}")

        self.model = MidFuse(
            text_model=self.text_model,
            speech_model=self.speech_model,
            text_features=512,
            speech_features=256,
        )

        if self.args.load_checkpoint is not None:
            self.model.load_(self.args.load_checkpoint)
            print(f"Loaded mid-fusion model from {self.args.load_checkpoint}")


        self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate)
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3, verbose=True
        ) # TODO @abhaydmathur : mod

        self.criterion = nn.BCELoss()
        self.model.to(self.device)

        # Ensure self.args.model_save_path exists
        self.model_name = self.args.model_name
        self.model_save_path = os.path.join(self.args.model_save_path, self.model_name, self.args.log_dir.split("/")[-1])    
        os.makedirs(self.model_save_path, exist_ok=True)

        self.best_model_path = None
        self.latest_model_path = None
        self.best_loss = np.inf
        self.best_epoch = 0

        self.training_history = {}

        pass

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

    def log_init(self):
        print("Getting Inital Metrics")
        train_info = self.validate(split="train", verbose=True)
        val_info = self.validate(split="val", verbose=True)

        self.lr_scheduler.step(val_info["loss"])

        log_train_info = {f"train/{k}": v for k, v in train_info.items()}
        self.save_to_log(self.args.log_dir, self.logger, log_train_info, 0)


        log_val_info = {f"val/{k}": v for k, v in val_info.items()}
        self.save_to_log(self.args.log_dir, self.logger, log_val_info, 0)

    def train_epoch(self, epoch, verbose=True):
        self.model.train()
        losses = []
        accs = []
        n_steps = len(self.train_loader)
        for i, batch in enumerate(self.train_loader):
            text = batch['transcription']
            label = batch['label'].squeeze().float().to('cuda')

            # text = {k: v.to('cuda') for k, v in text.items()}
            batch['x'] = batch['x'].to('cuda')

            self.optimizer.zero_grad()

            output = self.model(text_input = text, speech_input = batch).squeeze()
            loss = self.criterion(output, label)

            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

            with torch.no_grad():
                acc = (output.round() == label).float().mean().item()
                accs.append(acc)

            if verbose:
                print(f"\rEpoch {epoch + 1} [{i + 1}/{n_steps}] loss: {np.mean(losses):.3f}, acc: {np.mean(accs):.3f}   ", end="")
        print()

        info = {
            "loss": np.mean(losses),
            "acc": np.mean(accs),
            "lr": self.optimizer.param_groups[0]["lr"],
        }

        return info

    def train(self):
        self.log_init()
        self.model.train()

        for epoch in range(self.args.epochs):
            epoch_info  = self.train_epoch(epoch)
            
            log_epoch_info = {f"train/{k}": v for k, v in epoch_info.items()}
            self.save_to_log(self.args.log_dir, self.logger, log_epoch_info, epoch+1)

            val_info = self.validate()
            log_val_info = {f"val/{k}": v for k, v in val_info.items()}
            self.save_to_log(self.args.log_dir, self.logger, log_val_info, epoch+1)

            self.training_history[epoch] = {
                "train": epoch_info,
                "val": val_info,
            }

            # Save best and latest models
            if self.best_model_path is None or val_info["loss"] < self.best_loss: # TODO @abhaydmathur : best metrics??
                self.best_loss = val_info["loss"]
                self.best_model_path = os.path.join(self.model_save_path, "best_model.pth")
                self.save(self.best_model_path)
                self.best_epoch = epoch
            
            if self.latest_model_path is not None:
                os.remove(self.latest_model_path)
            self.latest_model_path = os.path.join(self.model_save_path, f"latest_model_{epoch+1}eps.pth")
            self.save(self.latest_model_path)

            if self.execute_callbacks(epoch):
                break

    def validate(self, split = "val", verbose=True):
        self.model.eval()
        losses = []
        accs = []

        if split=="train":
            loader = self.train_loader
        elif split=='val':
            loader = self.val_loader
        else:
            raise ValueError("Invalid split")

        with torch.no_grad():
            for i, batch in enumerate(loader):
                text = batch['transcription']
                label = batch['label'].squeeze().float().to('cuda')

                # text = {k: v.to('cuda') for k, v in text.items()}
                batch['x'] = batch['x'].to('cuda')

                output = self.model(text_input = text, speech_input = batch).squeeze()
                loss = self.criterion(output, label)
                losses.append(loss.item())
                
                acc = (output.round() == label).float().mean().item()
                accs.append(acc)

                if verbose:
                    print(
                        f"\rEvaluation on {split} [{i+1}/{len(loader)}] loss: {np.mean(losses):.3f}, acc: {np.mean(accs):.3f}     ",
                        end="",
                    )
        if verbose: print()
        return {"loss": np.mean(losses), "acc": np.mean(accs)}

    def save(self, path):
        self.model.save_(path)

    def execute_callbacks(self, epoch):
        # Early Stopping
        if self.args.early_stopping_patience is not None:
            if epoch - self.best_epoch > self.args.early_stopping_patience:
                print(f"Early Stopping after {epoch} epochs")
                return True
        return False
