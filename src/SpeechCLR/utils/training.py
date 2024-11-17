import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from utils.datasets import InTheWildDataset, ASVSpoof21Dataset
from torch.utils.tensorboard import SummaryWriter

from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback
from utils.losses import TripletMarginLoss, TripletMarginWithDistanceLoss, CosineDistance
from utils.metrics import ABXAccuracy

from models import SpeechEmbedder


class SpeechCLRTrainer:
    def __init__(self, args):

        self.model = SpeechEmbedder()

        self.training_args = TrainingArguments(
            output_dir=args.output_dir,  # output directory
            num_train_epochs=3,  # total number of training epochs
            per_device_train_batch_size=args.batch_size,  # batch size per device during training
            per_device_eval_batch_size=1,  # batch size for evaluation
            warmup_steps=50,  # number of warmup steps for learning rate scheduler
            weight_decay=0.01,  # strength of weight decay
            logging_dir="./logs",  # directory for storing logs
            do_train=True,
            do_eval=True,
            eval_strategy="epoch",
            learning_rate=args.learning_rate,
            max_grad_norm=1e3,
            lr_scheduler_type="linear",  # Compare with cosine, cosine_with_restarts, polynomial, constant, constant_with_warmup, inverse_sqrt, reduce_on_plateau, cosine_with_hard_restarts, warmup_cosine, warmup_constant, warmup_linear, warmup_cosine_hard_restarts
            logging_strategy="steps",
            logging_steps=50,
            save_strategy="epoch",
            save_safetensors=True,
            seed=42,
            load_best_model_at_end=True,
            eval_on_start=False,
        )

        train_dataset = InTheWildDataset(
            root_dir=args.data_path,
            metadata_file='meta.csv',
            include_spoofs=False,
            bonafide_label="bona-fide",
            filename_col="file",
            sampling_rate=args.sampling_rate,
            max_duration=args.max_duration,
            split="train",
            config=args.dataset_config,
            mode="triplet",
        )

        val_dataset = InTheWildDataset(
            root_dir=args.data_path,
            metadata_file='meta.csv',
            include_spoofs=False,
            bonafide_label="bona-fide",
            filename_col="file",
            sampling_rate=args.sampling_rate,
            max_duration=args.max_duration,
            split="val",
            config=args.dataset_config,
            mode="triplet",
        )

        if args.loss_fn == "triplet":
            self.loss_fn = TripletMarginLoss(margin=1.0)
        elif args.loss_fn == "triplet_cosine":
            self.loss_fn = TripletMarginWithDistanceLoss(distance_function=CosineDistance(), margin=1.0)
        else:
            raise NotImplementedError(f"Loss function {args.loss_fn} not implemented")  

        self.metrics_fn = ABXAccuracy()

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_loss_func=self.loss_fn,
            compute_metrics=self.metrics_fn,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=args.early_stopping_patience,
                    early_stopping_threshold=args.early_stopping_threshold,
                )
            ],
        )

    def train(self):
        self.trainer.train()


class SpeechCLRTrainerVanilla():
    def __init__(self, args):
        self.args = args
        self.model = SpeechEmbedder()
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate, weight_decay=0.01)
        # self.criterion = nn.CrossEntropyLoss()  # or any other loss function you prefer
        if args.loss_fn == "triplet":
            self.loss_fn = TripletMarginLoss(margin=1.0)
        elif args.loss_fn == "triplet_cosine":
            self.loss_fn = TripletMarginWithDistanceLoss(distance_function=CosineDistance(), margin=1.0)
        else:
            raise NotImplementedError(f"Loss function {args.loss_fn} not implemented")  


        self.train_dataset = InTheWildDataset(
            root_dir=args.data_path,
            metadata_file='meta.csv',
            include_spoofs=False,
            bonafide_label="bona-fide",
            filename_col="file",
            sampling_rate=args.sampling_rate,
            max_duration=args.max_duration,
            split="train",
            config=args.dataset_config,
            mode="triplet",
        )

        self.val_dataset = InTheWildDataset(
            root_dir=args.data_path,
            metadata_file='meta.csv',
            include_spoofs=False,
            bonafide_label="bona-fide",
            filename_col="file",
            sampling_rate=args.sampling_rate,
            max_duration=args.max_duration,
            split="val",
            config=args.dataset_config,
            mode="triplet",
        )

        self.train_loader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=1, shuffle=False)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

    def train(self):
        self.model.train()
        for epoch in range(self.args.epochs):
            running_loss = 0.0
            for i, input in enumerate(self.train_loader):

                input = {k: v.to(self.device) for k, v in input.items()}
                # if 'cuda' in self.device:
                #     input = {k: v.cuda() for k, v in input.items()}
                # print(type(input['anchor']), print(input['anchor'].device))

                # assert self.model.device == input['anchor'].device, f"Model device: {self.model.device}, input device: {input['anchor'].device}"

                output = self.model(input)  

                self.optimizer.zero_grad()
                loss = self.loss_fn(output['anchor'], output['positive'], output['negative'])
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i % 100 == 99:  # print every 100 mini-batches
                    print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
                    running_loss = 0.0

            # self.validate()

    def validate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.val_loader:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Accuracy on validation set: {100 * correct / total:.2f}%")