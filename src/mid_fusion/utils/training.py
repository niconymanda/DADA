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

from torch.utils.data import Dataset, ConcatDataset

from mid_fusion.utils.datasets import (
    InTheWildDataset,
    ASVSpoof21Dataset,
    ASVSpoof19LADataset,
    MLAADEnDataset,
    custom_collate
)
from mid_fusion.utils.metrics import equal_error_rate
from SpeechCLR.utils.logging import Logger
from SpeechCLR.models import SpeechEmbedder
from authorship_attribution.model import AuthorshipLLM, AuthorshipClassificationLLM
from authorship_attribution.config import load_checkpoint
from mid_fusion.models import (
    MidFuse, LateFuse, EarWorm, BookWorm, ConditionalLateFuse, FrozenConditionalLateFuse
)
from sklearn.metrics import f1_score


class MidFusionTrainer:
    def __init__(self, args):
        self.args = args

        self.args.learning_rate = float(self.args.learning_rate)

        self.train_datasets_list = []
        self.val_datasets_list = []
        self.test_datasets_list = []

        if "inthewild" in args.train_datasets:
            self.train_datasets_list.append(
                InTheWildDataset(
                    root_dir=args.train_datasets["inthewild"],
                    metadata_file=args.inthewild_meta_file,
                    transcription_col=args.inthewild_transcription_column,
                    include_spoofs=True,
                    bonafide_label="bona-fide",
                    filename_col="file",
                    sampling_rate=args.sampling_rate,
                    max_duration=args.max_duration,
                    split="train",
                    config=args.dataset_config,
                    text_tokenizer_name=self.args.text_model_name,
                    mode="classification",
                )
            )

            self.val_datasets_list.append(
                InTheWildDataset(
                    root_dir=args.train_datasets["inthewild"],
                    metadata_file=args.inthewild_meta_file,
                    transcription_col=args.inthewild_transcription_column,
                    include_spoofs=True,
                    bonafide_label="bona-fide",
                    filename_col="file",
                    sampling_rate=args.sampling_rate,
                    max_duration=args.max_duration,
                    split="val",
                    config=args.dataset_config,
                    text_tokenizer_name=self.args.text_model_name,
                    mode="classification",
                )
            )

        if "asvspoof21" in args.train_datasets:
            self.train_datasets_list.append(
                ASVSpoof21Dataset(
                    root_dir=args.train_datasets["asvspoof21"],
                    is_train=True,
                    is_eval=False,
                    split="train",
                    sampling_rate=16000,
                    max_duration=4,
                    get_transcription=True,
                    transcription_file="/home/infres/amathur-23/DADA/src/mid_fusion/asvspoof21_df_eval_transcriptions.csv",
                )
            )

            self.val_datasets_list.append(
                ASVSpoof21Dataset(
                    root_dir=args.train_datasets["asvspoof21"],
                    is_train=True,
                    is_eval=False,
                    split="val",
                    sampling_rate=16000,
                    max_duration=4,
                    get_transcription=True,
                    transcription_file="/home/infres/amathur-23/DADA/src/mid_fusion/asvspoof21_df_eval_transcriptions.csv",
                )
            )

        if "asvspoof19" in args.train_datasets:
            self.train_datasets_list.append(
                ASVSpoof19LADataset(
                    root_dir=args.train_datasets["asvspoof19"],
                    split="train",
                    sampling_rate=16000,
                    max_duration=4,
                    get_transcription=True,
                )
            )

            self.val_datasets_list.append(
                ASVSpoof19LADataset(
                    root_dir=args.train_datasets["asvspoof19"],
                    split="val",
                    sampling_rate=16000,
                    max_duration=4,
                    get_transcription=True,
                )
            )

        if "inthewild" in args.test_datasets:
            self.test_datasets_list.append(
                InTheWildDataset(
                    root_dir=args.test_datasets["inthewild"],
                    metadata_file=args.inthewild_meta_file,
                    transcription_col=args.inthewild_transcription_column,
                    include_spoofs=True,
                    bonafide_label="bona-fide",
                    filename_col="file",
                    sampling_rate=args.sampling_rate,
                    max_duration=args.max_duration,
                    split="val",
                    config=args.dataset_config,
                    text_tokenizer_name=self.args.text_model_name,
                    mode="classification",
                )
            )

        if "asvspoof21" in args.test_datasets:
            self.test_datasets_list.append(
                ASVSpoof21Dataset(
                    root_dir=args.test_datasets["asvspoof21"],
                    meta_dir=args.asv_meta_dir,
                    is_train=False,
                    is_eval=True,
                    split="val",
                    sampling_rate=16000,
                    max_duration=4,
                    get_transcription=True,
                    transcription_file="/home/infres/amathur-23/DADA/src/mid_fusion/asvspoof21_df_eval_transcriptions.csv",
                )
            )

        if "asvspoof19" in args.test_datasets:
            self.test_datasets_list.append(
                ASVSpoof19LADataset(
                    root_dir=args.test_datasets["asvspoof19"],
                    split="val",
                    sampling_rate=16000,
                    max_duration=4,
                    get_transcription=True,
                )
            )

        if "mlaad_en" in args.test_datasets:
            self.test_datasets_list.append(
                MLAADEnDataset(
                    root_dir=args.test_datasets["mlaad_en"],
                    split="val",
                    sampling_rate=16000,
                    max_duration=4,
                )
            )

        print("Loaded Train Datasets - ", self.train_datasets_list)
        print("Loaded Val Datasets - ", self.val_datasets_list)

        self.train_dataset = ConcatDataset(self.train_datasets_list)
        self.val_dataset = ConcatDataset(self.val_datasets_list)

        print("Loaded Dataset - ")

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate
        )

        print(f"Training Samples : {len(self.train_loader)*args.batch_size}")
        print(f"Validation Samples : {len(self.val_loader)*args.batch_size}")

        self.test_loaders = dict(
            zip(
                self.args.test_datasets,
                [
                    DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate)
                    for ds in self.test_datasets_list
                ],
            )
        )

        self.log_path = os.path.join(args.log_dir, args.model_name)
        self.logger = Logger(log_dir=self.log_path)
        print(f"Logging to {self.log_path}")

        self.device = self.args.device
        print(f"Using device: {self.device}")

        self.speech_model = SpeechEmbedder(device=self.device)
        self.text_model = AuthorshipLLM(
            self.args.text_model_name,
            num_layers=self.args.mlp_layers,
            use_layers=self.args.hidden_layers,
            out_features = self.args.text_features,
            device=self.device
        )

        if args.text_model_path is not None:
            checkpoint = torch.load(
                args.text_model_path,
                map_location=torch.device("cpu"),
                weights_only=True,
            )
            self.text_model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded text model from {args.text_model_path}")

        if args.speech_model_path is not None:
            self.speech_model.load_(args.speech_model_path)
            print(f"Loaded speech model from {args.speech_model_path}")

        if args.fusion_strategy == "mid":
            self.model = MidFuse(
                text_model=self.text_model,
                speech_model=self.speech_model,
                text_features=self.args.text_features, # TODO @abhaydmathur : change to args....
                speech_features=self.args.audio_features,
            )

            if self.args.load_checkpoint is not None:
                self.model.load_(self.args.load_checkpoint)
                print(f"Loaded mid-fusion model from {self.args.load_checkpoint}")

        elif args.fusion_strategy == "late":
            self.model = LateFuse(
                text_model=self.text_model,
                speech_model=self.speech_model,
                text_features=self.args.text_features,
                speech_features=self.args.speech_features,
                alpha=0.5,
            )

        elif args.fusion_strategy == "conditional_late":
            self.text_classifier = AuthorshipClassificationLLM(
                model = self.text_model,
                num_labels = 54,
                head_type='mlp',
            )

            self.text_classifier = load_checkpoint(self.text_classifier, self.args.text_classifier_path)

            self.speech_classifier = EarWorm(
                speech_model=self.speech_model,
                speech_features=self.args.speech_features,
                train_encoder=False
            )

            self.speech_classifier.load_(self.args.speech_classifier_path)

            self.model = FrozenConditionalLateFuse(
                text_classifier = self.text_classifier,
                speech_classifier = self.speech_classifier
            )

            self.text_id_to_author = self.args.text_id_to_author
            self.author_to_id = {v: k for k, v in self.text_id_to_author.items()}

        elif args.fusion_strategy == "audio":
            self.model = EarWorm(speech_model=self.speech_model, speech_features=self.args.speech_features, train_encoder=self.args.train_audio_encoder)

        elif args.fusion_strategy == "text":
            self.model = BookWorm(text_model=self.text_model, text_features=self.args.text_features)

        self.optimizer = optim.Adam(
            self.model.trainable_parameters(), lr=args.learning_rate
        )
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3, verbose=True
        )  # TODO @abhaydmathur : mod

        self.criterion = nn.BCELoss()
        self.model = self.model.to(self.device)

        # Ensure self.args.model_save_path exists
        self.model_name = self.args.model_name
        self.model_save_path = os.path.join(
            self.args.model_save_path, self.model_name, self.args.log_dir.split("/")[-1]
        )
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

        if self.args.test_at_launch:
            test_info = self.test(verbose=True)
            for loader_name in test_info.keys():
                log_test_info = {f"test/{loader_name}/{k}": v for k, v in test_info[loader_name].items()}
                self.save_to_log(self.args.log_dir, self.logger, log_test_info, 0)

        train_info = self.validate(split="train", verbose=True)
        val_info = self.validate(split="val", verbose=True)


        self.lr_scheduler.step(val_info["loss"])

        log_train_info = {f"train/{k}": v for k, v in train_info.items()}
        self.save_to_log(self.args.log_dir, self.logger, log_train_info, 0)

        log_val_info = {f"val/{k}": v for k, v in val_info.items()}
        self.save_to_log(self.args.log_dir, self.logger, log_val_info, 0)



    def train_epoch(self, epoch, verbose=True):
        self.model.train_()
        losses = []
        accs = []
        n_steps = len(self.train_loader)
        for i, batch in enumerate(self.train_loader):
            text = batch["transcription"]
            label = batch["label"].squeeze().float().to(self.device)

            batch["x"] = batch["x"].to(self.device)

            self.optimizer.zero_grad()

            if self.args.fusion_strategy == "conditional_late":
                text_ids = torch.tensor([self.author_to_id[a] for a in batch["author"]]).to(self.device)
                output = self.model(text_input=text, speech_input=batch, class_idx=text_ids).squeeze()
            else:
                output = self.model(text_input=text, speech_input=batch).squeeze()
    
            loss = self.criterion(output, label)

            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

            with torch.no_grad():
                acc = (output.round() == label).float().mean().item()
                accs.append(acc)

            if verbose:
                print(
                    f"\rEpoch {epoch + 1} [{i + 1}/{n_steps}] loss: {np.mean(losses):.3f}, acc: {np.mean(accs):.3f}   ",
                    end="",
                )
        print()

        info = {
            "loss": np.mean(losses),
            "acc": np.mean(accs),
            "lr": self.optimizer.param_groups[0]["lr"],
        }

        if self.args.fusion_strategy in ["late", "conditional_late"]:
            info["alpha"] = self.model.alpha.item()

        return info

    def train(self):
        self.log_init()
        self.model.train_()

        for epoch in range(self.args.epochs):
            epoch_info = self.train_epoch(epoch)

            log_epoch_info = {f"train/{k}": v for k, v in epoch_info.items()}
            self.save_to_log(self.args.log_dir, self.logger, log_epoch_info, epoch + 1)

            val_info = self.validate()
            log_val_info = {f"val/{k}": v for k, v in val_info.items()}
            self.save_to_log(self.args.log_dir, self.logger, log_val_info, epoch + 1)

            self.training_history[epoch] = {
                "train": epoch_info,
                "val": val_info,
            }

            # Save best and latest models
            if (
                self.best_model_path is None or val_info["loss"] < self.best_loss
            ):  # TODO @abhaydmathur : best metrics??
                self.best_loss = val_info["loss"]
                self.best_model_path = os.path.join(
                    self.model_save_path, "best_model.pth"
                )
                self.save(self.best_model_path)
                self.best_epoch = epoch

            if self.latest_model_path is not None:
                try:
                    os.remove(self.latest_model_path)
                except:
                    try:
                        self.model.remove(self.latest_model_path)
                    except:
                        print(".")
                    print("Could not remove latest model")
            self.latest_model_path = os.path.join(
                self.model_save_path, f"latest_model_{epoch+1}eps.pth"
            )
            self.save(self.latest_model_path)

            if self.execute_callbacks(epoch):
                break

            if (epoch + 1) % self.args.test_freq == 0:
                test_info = self.test()
                for loader_name in test_info.keys():
                    log_test_info = {f"test/{loader_name}/{k}": v for k, v in test_info[loader_name].items()}
                    self.save_to_log(self.args.log_dir, self.logger, log_test_info, epoch+1)

        test_info = self.test()
        for loader_name in test_info.keys():
            log_test_info = {f"test/{loader_name}/{k}": v for k, v in test_info[loader_name].items()}
            self.save_to_log(self.args.log_dir, self.logger, log_test_info, epoch+1)

    def validate(self, split="val", verbose=True):
        self.model.eval_()
        losses = []
        accs = []

        labels = []
        preds = []

        if split == "train":
            loader = self.train_loader
        elif split == "val":
            loader = self.val_loader
        else:
            raise ValueError("Invalid split")

        with torch.no_grad():
            for i, batch in enumerate(loader):
                text = batch["transcription"]
                label = batch["label"].squeeze().float().to(self.device)

                batch["x"] = batch["x"].to(self.device)

                if self.args.fusion_strategy == "conditional_late":
                    text_ids = torch.tensor([self.author_to_id[a] for a in batch["author"]]).to(self.device)
                    output = self.model(text_input=text, speech_input=batch, class_idx=text_ids).squeeze()
                else:
                    output = self.model(text_input=text, speech_input=batch).squeeze()
                # print(label, output)
                loss = self.criterion(output, label)
                losses.append(loss.item())

                acc = (output.round() == label).float().mean().item()
                accs.append(acc)

                labels.extend(np.atleast_1d(label.cpu().numpy()))
                preds.extend(np.atleast_1d(output.cpu().numpy()))

                if verbose:
                    print(
                        f"\rEvaluation on {split} [{i+1}/{len(loader)}] loss: {np.mean(losses):.3f}, acc: {np.mean(accs):.3f}   ",
                        end="",
                    )

        labels = np.array(labels)
        preds = np.array(preds)
        eer = equal_error_rate(labels, preds)

        f1 = f1_score(labels, preds.round())

        if verbose:
            print(
                f"\rEvaluation on {split} [{i+1}/{len(loader)}] loss: {np.mean(losses):.3f}, acc: {np.mean(accs):.3f}, eer: {eer:.3f}, f1_score: {f1:.3f}   ",
            )
        return {
            "loss": np.mean(losses),
            "acc": np.mean(accs),
            "eer": eer,
            "f1_score": f1,
        }

    def test(self, verbose=True):
        test_dict = {}

        for loader_name in self.test_loaders.keys():
            loader = self.test_loaders[loader_name]
            losses = []
            accs = []
            labels = []
            preds = []

            with torch.no_grad():
                for i, batch in enumerate(loader):
                    text = batch["transcription"]
                    label = batch["label"].squeeze().float().to(self.device)
                    batch["x"] = batch["x"].to(self.device)

                    if self.args.fusion_strategy == "conditional_late":
                        text_ids = torch.tensor([self.author_to_id[a] for a in batch["author"]]).to(self.device)
                        output = self.model(text_input=text, speech_input=batch, class_idx=text_ids).squeeze()
                    else:
                        output = self.model(text_input=text, speech_input=batch).squeeze()
                    loss = self.criterion(output, label)
                    losses.append(loss.item())

                    acc = (output.round() == label).float().mean().item()
                    accs.append(acc)

                    labels.extend(np.atleast_1d(label.cpu().numpy()))
                    preds.extend(np.atleast_1d(output.cpu().numpy()))

                    if verbose:
                        print(
                            f"\rEvaluation on {loader_name} [{i+1}/{len(loader)}] loss: {np.mean(losses):.3f}, acc: {np.mean(accs):.3f}   ",
                            end="",
                        )

            labels = np.array(labels)
            preds = np.array(preds)

            eer = equal_error_rate(labels, preds)
            f1 = f1_score(labels, preds.round())

            if verbose:
                print(
                    f"\rEvaluation on {loader_name} [{i+1}/{len(loader)}] loss: {np.mean(losses):.3f}, acc: {np.mean(accs):.3f}, eer: {eer:.3f}, f1_score: {f1:.3f}   ",
                )
            test_dict[loader_name] = {
                "loss": np.mean(losses),
                "acc": np.mean(accs),
                "eer": eer,
                "f1_score": f1,
            }
        return test_dict

    def save(self, path):
        self.model.save_(path)

    def execute_callbacks(self, epoch):
        # Early Stopping
        if self.args.early_stopping_patience is not None:
            if epoch - self.best_epoch > self.args.early_stopping_patience:
                print(f"Early Stopping after {epoch} epochs")
                return True
        return False
