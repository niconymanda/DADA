import torch
from tqdm import tqdm  
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import tempfile
from ray import train
from loss_functions import TripletLoss, ContrastiveLoss, SquaredCosineSimilarityLoss
from ray.train import Checkpoint, get_checkpoint
from pathlib import Path
import ray.cloudpickle as pickle
import os 
from torch.utils.tensorboard import SummaryWriter
from early_stopping import EarlyStopping
import config as cfg
from model import AuthorshipClassificationLLM
from torch.nn import functional as F
from typing import Optional, Literal
import wandb
import numpy as np
from transformers import get_linear_schedule_with_warmup
from sklearn.mixture import GaussianMixture


class TrainerAuthorshipAttribution:
    """
    TrainerAuthorshipAttribution is a class designed to handle the training and validation of a model for authorship attribution tasks. It supports early stopping, logging to TensorBoard, and optional training of a classification head.
    Attributes:
        model (torch.nn.Module): The model to be trained.
        loss_fn (callable): The loss function to be used (e.g. TripletLoss).
        optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for the training data.
        val_dataloader (torch.utils.data.DataLoader): DataLoader for the validation data.
        report_to (str, optional): The logging tool to report to (e.g., 'tensorboard'). Defaults to None.
        early_stopping (EarlyStopping, optional): An instance of EarlyStopping for early stopping. Defaults to None.
        args (argparse.Namespace): Additional arguments, including epochs, logging_step, and learning_rate.
        repository_id (str): The repository ID for saving logs and checkpoints.
        author_id_map (dict): A mapping of author IDs.
        writer (torch.utils.tensorboard.SummaryWriter, optional): TensorBoard writer for logging. Initialized if report_to is 'tensorboard'.
        output_dir (str): Directory for saving logs and checkpoints.
    
    """
    
    def __init__(self, model, 
                 train_dataloader, 
                 val_dataloader, 
                 args, 
                 repository_id,
                 author_id_map,
                 loss_fn=None, 
                 optimizer=None, 
                 lr_scheduler=None,
                 project_name = 'authorship_attribution',
                 report_to: Optional[Literal['tensorboard', 'wandb']] = None, 
                 early_stopping=True,
                 save_model=False, 
                 tune=False):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.report_to = report_to
        self.args = args
        self.repository_id = repository_id
        self.output_dir = f"{repository_id}/logs"
        self.author_id_map = author_id_map
        self.num_labels = len(self.author_id_map.keys())
        self.early_stopping = early_stopping
        self.save_model = save_model
        self.device = cfg.get_device()
        if not tune:
            self.loss_fn = self.get_loss_fn() if loss_fn is None else loss_fn   
            self.optimizer = self.get_optimizer() if optimizer is None else optimizer
            self.lr_scheduler = self.get_lr_scheduler(self.optimizer) if lr_scheduler is None  else lr_scheduler
            self.loss_fn.to(self.device)

        self.model.to(self.device)
        # print(self.model)
        self.distance_function = cfg.get_distance_function(args.distance_function)
        
        if self.report_to == 'tensorboard':
            self.writer = SummaryWriter(f"{repository_id}/logs")
        elif report_to == 'wandb':
            wandb.init(project=project_name, config=vars(args))
            wandb.watch(model)
        else:
            raise ValueError("Invalid report_to value. Must be 'tensorboard' or 'wandb'")
        
        if self.early_stopping:
            self.early_stopping_model = EarlyStopping(patience=args.early_stopping_patience)
            self.early_stopping_classif = EarlyStopping(patience=self.args.early_stopping_patience)
        
    def train(self, classification_head=False):
        """
        Trains the model for a specified number of epochs and optionally trains a classification head.
        Args:
            classification_head (bool, optional): If True, trains an additional classification head after the main training. Defaults to False.
        Returns:
            model: The trained model.
            classification_model (optional): The trained classification model if `classification_head` is True.
        """
        
        for epoch_n in range(self.args.epochs):
            # train_loss = self.train_model(epoch_n)
            # val_loss = self.validate(epoch_n)
            train_loss = self.train_model(epoch_n, self.train_dataloader, self.optimizer, self.loss_fn, self.lr_scheduler, self.model, self.device)
            val_loss, accuracy = self.validate(epoch_n, self.val_dataloader, self.loss_fn, self.model, self.device)
            if self.early_stopping:
                if self.early_stopping_model.step(val_loss):
                    print("Early stopping triggered")
                    break 
        cfg.save_checkpoint(self.model, self.optimizer, epoch_n, f'{self.repository_id}/final.pth') if self.save_model else None
        
        if classification_head:
            print("Training classification head!")
            if self.args.classification_head == 'gmm':
                embeddings = self.extract_embeddings()
                gmm = self.fit_gmm(embeddings)
                cfg.save_checkpoint(self.model, self.optimizer, epoch_n, f'{self.repository_id}/classification_final.pth') if self.save_model else None
                return self.model, gmm
            
            classification_model = AuthorshipClassificationLLM(self.model, num_labels=self.num_labels, head_type=self.args.classification_head)
            classification_loss = nn.CrossEntropyLoss()
            classification_model.to(self.device)
            classification_loss.to(self.device)
            classification_optimizer = optim.AdamW(classification_model.parameters(), lr=self.args.learning_rate_classification)
            for epoch_n in range(self.args.epochs_classification):
                train_loss = self.train_classification(classification_model, classification_loss, classification_optimizer, epoch_n)
                val_loss = self.validate_classification(classification_model, classification_loss, epoch_n)
                if self.early_stopping:
                    if self.early_stopping_classif.step(val_loss):
                        print("Early stopping triggered")
                        break 
                    
            cfg.save_checkpoint(classification_model, classification_optimizer, epoch_n, f'{self.repository_id}/classification_final.pth') if self.save_model else None
            return self.model, classification_model
        return self.model, None 
    def extract_embeddings(self):
        """
        Extracts embeddings from the model and saves them to a file.
        """
        self.model.eval()
        all_embeddings = []
        with torch.no_grad():
            for i,batch in enumerate(tqdm(self.train_dataloader, desc=f"Extracting embeddings")):
                text = batch['anchor_example']

                embeddings = self.model(text).detach().cpu().numpy()
                all_embeddings.append(embeddings)

        all_embeddings = np.concatenate(all_embeddings, axis=0)
        # print(all_embeddings)
        return all_embeddings
    
    def fit_gmm(self, embeddings):
        """
        Fit the GMM using the embeddings and their corresponding labels.
        """
        gmm = GaussianMixture(n_components=self.num_labels, covariance_type='diag', random_state=self.args.seed)
        gmm.fit(embeddings)
        return gmm

    def train_classification(self, classification_model, loss_fn_classification, optimizer_classification, epoch_n):
        """
        Trains only the cassification head of the model for one epoch.
        Args:
            model (torch.nn.Module): The classification head model to be trained.
            loss_fn (callable): The loss function.
            train_dataloader (torch.utils.data.DataLoader): DataLoader for the training data.
            optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
            device (torch.device): The device to run the model on (e.g., 'cpu' or 'cuda').
            writer (torch.utils.tensorboard.SummaryWriter): TensorBoard writer for logging.
            epoch_n (int): The current epoch number.
            args (argparse.Namespace): Additional arguments, including logging_step.
        Returns:
            float: The average loss over the training dataset.
        """
        
        classification_model.train()
        total_loss = 0
        current_loss = 0.0
        all_embeddings = []
        all_labels = []

        for i,batch in enumerate(tqdm(self.train_dataloader, desc=f"Train Epoch {epoch_n+1}/{self.args.epochs_classification}")):
            text = batch['anchor_example']
            labels = batch['label'].to(self.device)

            out_model = classification_model(text)
                        
            optimizer_classification.zero_grad()
            loss_value = loss_fn_classification(out_model, labels)
            loss_value.backward()
            optimizer_classification.step()
            total_loss += loss_value.item()
            current_loss += loss_value.item()

            if i % self.args.logging_step == self.args.logging_step - 1:
                metrics = {
                    "Loss": current_loss / self.args.logging_step,
                    "epoch": epoch_n,
                    "step": i
                }
                self._log_metrics(metrics, phase='Train_classificaion')
                current_loss = 0.0  

        train_loss = total_loss / len(self.val_dataloader)
        metrics = {
            "Loss": train_loss,
            "epoch": epoch_n
        }
        self._log_metrics(metrics, phase='Train_classificaion_epochs')
        return total_loss / len(self.train_dataloader)
    
    def validate_classification(self, classificaion_model, loss_fn_classification, epoch_n):
        """
        Validates the performance of a classification head on a validation dataset.
        Args:
            model (torch.nn.Module): The model to be validated.
            loss_fn (callable): The loss function used to compute the loss.
            val_dataloader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
            device (torch.device): The device (CPU or GPU) on which the model and data are loaded.
            writer (torch.utils.tensorboard.SummaryWriter): TensorBoard writer for logging validation loss.
            epoch_n (int): The current epoch number.
            args (argparse.Namespace): Additional arguments, including logging_step for logging frequency.
        Returns:
            float: The average validation loss over the entire validation dataset.
        """
        
        classificaion_model.eval()
        total_loss = 0
        current_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for i,batch in enumerate(tqdm(self.val_dataloader, desc=f"Val Epoch {epoch_n+1}/{self.args.epochs_classification}")):
                text = batch['anchor_example']
                labels = batch['label'].to(self.device)

                outputs = classificaion_model(text)
                loss_value = loss_fn_classification(outputs, labels)
                total_loss += loss_value.item()
                current_loss += loss_value.item()
                preds = outputs.argmax(dim=1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

                if i % self.args.logging_step == self.args.logging_step - 1:
                    metrics = {
                        "Loss": current_loss / self.args.logging_step,
                        "epoch": epoch_n,
                        "step": i,
                        "Accuracy": correct / total
                    }
                    self._log_metrics(metrics, phase='Val_classificaion')
                    current_loss = 0.0  
                         
        val_loss = total_loss / len(self.val_dataloader)
        metrics = {
            "Loss": val_loss,
            "epoch": epoch_n,
            "Accuracy": correct / total
        }
        self._log_metrics(metrics, phase='Val_classificaion_epochs')
        return val_loss

    def train_model(self, epoch_n, train_dataloader, optimizer, loss_fn, lr_scheduler, model, device):
        """
        Trains only the the given LLM model using the provided training data.
        Args:
            epoch_n (int): The current epoch number.
        Returns:
            float: The average loss over the training data.
        """
        
        model.train()
        total_loss = 0
        current_loss = 0.0
        correct_count = 0
        iters = len(train_dataloader)
        for i,batch in enumerate(tqdm(train_dataloader, desc=f"Train Epoch {epoch_n+1}/{self.args.epochs}")):
            anchor_example = batch['anchor_example']
            positive_example = batch['positive_example']
            negative_example = batch['negative_example']
            
            optimizer.zero_grad()
            anchor_embeddings = model(anchor_example)
            positive_embeddings = model(positive_example)
            negative_embeddings = model(negative_example)

            loss_value = loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)
            total_loss += loss_value.item()
            current_loss += loss_value.item()
            
            loss_value.backward()
            optimizer.step()
            lr_scheduler.step()
            
            # Accuracy calculation
            distance_positive = self.distance_function(anchor_embeddings, positive_embeddings)
            distance_negative = self.distance_function(anchor_embeddings, negative_embeddings)
            correct_count += torch.sum(distance_positive < distance_negative).item()
            
            if i % self.args.logging_step == self.args.logging_step - 1:
                    metrics = {
                        "Loss": current_loss / self.args.logging_step,
                        "epoch": epoch_n,
                        "step": i
                    }
                    self._log_metrics(metrics, phase='Train')
                    current_loss = 0.0
                
        train_loss = total_loss / len(train_dataloader)
        metrics = {
            "Loss": train_loss,
            "Accuracy": correct_count / len(train_dataloader.dataset),
            "epoch": epoch_n,
        }
        self._log_metrics(metrics, phase='Train_epoch')
        return total_loss / len(train_dataloader)

    def validate(self, epoch_n, val_dataloader, loss_fn, model, device):
        """
        Validates the model on the validation dataset.
        Args:
            epoch_n (int): The current epoch number.
        Returns:
            float: The average validation loss over the entire validation dataset.
        """
        
        model.eval()
        total_loss = 0
        current_loss = 0.0
        correct_count = 0
        total = 0
        with torch.no_grad():
            for i,batch in enumerate(tqdm(val_dataloader, desc=f"Val Epoch {epoch_n+1}/{self.args.epochs}")):
                anchor_example = batch['anchor_example']
                positive_example = batch['positive_example']
                negative_example = batch['negative_example']

                anchor_embeddings = model(anchor_example)
                positive_embeddings = model(positive_example)
                negative_embeddings = model(negative_example)
                loss_value = loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)
                total_loss += loss_value.item()
                current_loss += loss_value.item()
                
                # Accuracy calculation
                distance_positive = self.distance_function(anchor_embeddings, positive_embeddings)
                distance_negative = self.distance_function(anchor_embeddings, negative_embeddings)
                correct_count += torch.sum(distance_positive < distance_negative).item()
                total += len(anchor_example)

                if i % self.args.logging_step == self.args.logging_step - 1:
                    metrics = {
                    "Loss": current_loss / self.args.logging_step,  
                    "Accuracy": correct_count / total,
                    "epoch": epoch_n,
                    "step": i
                    }
                    self._log_metrics(metrics, phase='Val')
                    current_loss = 0.0
                    
        val_loss = total_loss / len(self.val_dataloader)
        metrics = {
                "Loss": val_loss,
                "Accuracy": correct_count / total,
                "epoch": epoch_n,
                }
        self._log_metrics(metrics, phase='Val_epoch')
        return val_loss, correct_count / total
    
    def _log_metrics(self, metrics, phase):
        if self.report_to == 'wandb':
            for key, value in metrics.items():
                wandb.log({f"{phase}/{key}": value}) if (key != 'epoch' and key != 'step') else None
        elif self.report_to == 'tensorboard':
            for key, value in metrics.items():
                if 'epoch' in phase:
                    # print(f"{phase}/{key}, epoch")
                    self.writer.add_scalar(f"{phase}/{key}", value, metrics['epoch']) if (key != 'epoch' and key != 'step') else None
                elif 'Train' in phase:                    
                    self.writer.add_scalar(f"{phase}/{key}", value, metrics['epoch'] * len(self.train_dataloader) + metrics['step']) if (key != 'epoch' and key != 'step') else None
                elif 'Val' in phase:
                    self.writer.add_scalar(f"{phase}/{key}", value, metrics['epoch'] * len(self.val_dataloader) + metrics['step']) if (key != 'epoch' and key != 'step') else None
                else:
                    print("Invalid phase")

    def get_loss_fn(self, margin=None):
        margin = self.args.margin if margin is None else margin
        if self.args.loss_function == 'triplet':
            return TripletLoss(margin=margin, distance_function=self.args.distance_function)
        elif self.args.loss_function == 'contrastive':
            return ContrastiveLoss(margin=margin)
        elif self.args.loss_function == 'cos2':
            return SquaredCosineSimilarityLoss()
        else:
            raise ValueError("Invalid loss_function value. Must be 'triplet', 'contrastive', or 'cos2'")
    
    def get_optimizer(self):
        return optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
    
    def get_lr_scheduler(self, optimizer=None, num_training_steps=None):
        
        num_training_steps = self.args.epochs * len(self.train_dataloader) if num_training_steps is None else num_training_steps
        warmup_steps = int(0.3 * num_training_steps)  
        
        if self.args.lr_scheduler == 'linear_warmup':
            return get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps
                )
        if self.args.lr_scheduler == 'linear':
            return optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1, last_epoch=-1)
        elif self.args.lr_scheduler == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs*len(self.train_dataloader), eta_min=1e-7, last_epoch=-1)
        elif self.args.lr_scheduler == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
        else:
            raise ValueError("Invalid lr_scheduler value. Must be 'linear_warmup', 'linear', or 'cosine'")


def train_tune(config, train_dataset, val_dataset, model, device, args):
    """
    Trains and tunes a model using the provided configuration, datasets, and device.
    Args:
        config (dict): Configuration dictionary containing training parameters such as batch size, learning rate, and number of epochs.
        train_dataset (Dataset): The dataset used for training.
        val_dataset (Dataset): The dataset used for validation.
        model (nn.Module): The model to be trained.
        device (str): The device to run the training on, either 'cpu' or 'cuda'.
    Returns:
        None
    The function performs the following steps:
    1. Sets up the device for training (CPU or GPU).
    2. Initializes the loss function and optimizer.
    3. Loads a checkpoint if available to resume training from a previous state.
    4. Creates data loaders for the training and validation datasets.
    5. Trains the model for the specified number of epochs, performing validation at the end of each epoch.
    6. Implements early stopping based on validation loss to prevent overfitting.
    7. Saves checkpoints during training to allow resuming from the last state.
    Note:
        The function uses a TripletLoss with a fixed margin of 0.5.
    """
    
    batch_size = config["batch_size"]
    learning_rate = config["lr"]
    margin = config["margin"]
    # device = "cpu"
    # if torch.cuda.is_available():
    #     device = "cuda:0"
    #     if torch.cuda.device_count() > 1:
    #         model = nn.DataParallel(model)
    device = cfg.get_device()
    model.to(device)
    loss_fn = TripletLoss(margin=margin)
    loss_fn = loss_fn.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    checkpoint = get_checkpoint()

    distance_function = cfg.get_distance_function(args.distance_function)
    
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            print(f"Resuming from checkpoint in {data_path}")
            with open(data_path, "rb") as fp:
                checkpoint_state = pickle.load(fp)
            start_epoch = checkpoint_state["epoch"]
            model.load_state_dict(checkpoint_state["net_state_dict"])
            optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"]*len(train_dataloader), eta_min=0, last_epoch=-1)
    # early_stopping_tune = EarlyStopping(patience=args.early_stopping_patience)

    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        total_loss = 0.0
        for batch in train_dataloader:
            anchor_example = batch['anchor_example']
            positive_example = batch['positive_example']
            negative_example = batch['negative_example']

            optimizer.zero_grad()
            anchor_embeddings = model(anchor_example)
            positive_embeddings = model(positive_example)
            negative_embeddings = model(negative_example)

            loss_value = loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)
            
            total_loss += loss_value.item()
            loss_value.backward()
            optimizer.step()
            lr_scheduler.step()
            
        # Validate the model
        val_loss = 0.0
        correct_count = 0
        avg_val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                anchor_example = batch['anchor_example']
                positive_example = batch['positive_example']
                negative_example = batch['negative_example']

                anchor_embeddings = model(anchor_example)
                positive_embeddings = model(positive_example)
                negative_embeddings = model(negative_example)

                loss_value = loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)
                val_loss += loss_value.item()

                distance_positive = distance_function(anchor_embeddings, positive_embeddings)
                distance_negative = distance_function(anchor_embeddings, negative_embeddings)
                correct_count += torch.sum(distance_positive < distance_negative).item()

        avg_val_loss = val_loss / len(val_dataloader)
        if epoch % config['epochs']-1 == 0:
            checkpoint_data = {
                "epoch": epoch,
                "net_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            with tempfile.TemporaryDirectory() as checkpoint_dir:
                data_path = Path(checkpoint_dir) / "data.pkl"
                with open(data_path, "wb") as fp:
                    pickle.dump(checkpoint_data, fp)
                checkpoint = Checkpoint.from_directory(checkpoint_dir)
                train.report(
                    {"loss": avg_val_loss, "accuracy": correct_count / len(val_dataloader.dataset)},
                    checkpoint=checkpoint,
                )
        # if early_stopping_tune.step(avg_val_loss):
        #     print("Early stopping triggered")
        #     break 