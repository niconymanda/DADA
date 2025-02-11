import torch
from tqdm import tqdm  
import torch.nn as nn
import torch.optim as optim
from loss_functions import TripletLoss, ContrastiveLoss, SquaredCosineSimilarityLoss, AdaTriplet, TripletLossTemperature
from early_stopping import EarlyStopping
import config as cfg
from model import AuthorshipClassificationLLM
from torch.nn import functional as F
from typing import Optional, Literal
# import wandb
import numpy as np
from transformers import get_linear_schedule_with_warmup
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from logger import Logger
import matplotlib.pyplot as plt
from visualisation import get_tsne_fig


class TrainerAuthorshipAttribution:
    """
    TrainerAuthorshipAttribution is a class designed to handle the training and validation of a model for authorship attribution tasks. It supports early stopping, logging to TensorBoard, and optional training of a classification head.
    Attributes:
        model (torch.nn.Module): The model to be trained.
        criterion (callable): The loss function to be used (e.g. TripletLoss).
        optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for the training data.
        val_dataloader (torch.utils.data.DataLoader): DataLoader for the validation data.
        report_to (str, optional): The logging tool to report to (e.g., 'tensorboard'). Defaults to None.
        early_stopping (EarlyStopping, optional): An instance of EarlyStopping for early stopping. Defaults to None.
        args (argparse.Namespace): Additional arguments, including epochs, and learning_rate.
        repository_id (str): The repository ID for saving logs and checkpoints.
        author_id_map (dict): A mapping of author IDs.
        writer (torch.utils.tensorboard.SummaryWriter, optional): TensorBoard writer for logging. Initialized if report_to is 'tensorboard'.
        output_dir (str): Directory for saving logs and checkpoints.
    
    """
    
    def __init__(self, model, 
                 train_dataloader, 
                 val_dataloaders, 
                 args, 
                 repository_id,
                 author_id_map,
                 criterion=None, 
                 optimizer=None, 
                 lr_scheduler=None,
                 project_name = 'authorship_attribution',
                 report_to: Optional[Literal['tensorboard', 'wandb']] = None, 
                 early_stopping=True,
                 save_model=False, 
                 tune=False, 
                 log_plots=True, 
                 additional_training=False,
                 classification_dataloader=None):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloaders = val_dataloaders 
        self.report_to = report_to
        self.args = args
        self.repository_id = repository_id
        self.output_dir = f"{repository_id}/logs"
        self.author_id_map = author_id_map
        self.early_stopping = early_stopping
        self.save_model = save_model
        self.model_weights = self.args.text_model_path
        self.log_plots = log_plots
        self.additional_training = additional_training
        self.classification_dataloader = classification_dataloader
        self.device = cfg.get_device()
        
        if not tune:
            self.criterion = self.get_criterion() if criterion is None else criterion   
            self.optimizer = self.get_optimizer() if optimizer is None else optimizer
            self.lr_scheduler = self.get_lr_scheduler(self.optimizer) if lr_scheduler is None  else lr_scheduler
            self.criterion.to(self.device)
        self.num_labels = len(self.author_id_map) if self.author_id_map is not None else 0
        self.model.to(self.device)
        self.distance_function = cfg.get_distance_function(args.distance_function)

        self.logger = Logger(args, report_to, project_name, model, self.output_dir)
        
        if self.early_stopping:
            self.early_stopping_model = EarlyStopping(patience=args.early_stopping_patience)
            self.early_stopping_classif = EarlyStopping(patience=self.args.early_stopping_patience)
        
    def train(self, classification_head=False):
        """
        Train the model with optional classification head.
        Parameters:
        classification_head (bool): If True, train a classification head on top of the model.
        Returns:
        tuple: The trained model and the classification head (if any), otherwise None.
        The function performs the following steps:
        1. If custom model weights are provided, load them and optionally perform additional training.
        2. If no custom weights are provided, train the model from scratch.
        3. If classification_head is True, train a classification head using the specified method (GMM, KNN, linear, or MLP).
        The function supports early stopping and saves the model checkpoints if specified.
        """

        if self.model_weights is not None:
            print("Loading custom model weights!...")
            self.model = cfg.load_checkpoint(self.model, self.model_weights)
            self.model.to(self.device)
            if self.additional_training:
                print("Training on paper set with weights!...")
                for epoch_n in range(self.args.epochs):
                    train_loss = self.train_model(epoch_n)
                    val_loss, accuracy = self.validate(epoch_n)
                    if self.early_stopping:
                        if self.early_stopping_model.step(val_loss):
                            print("Early stopping triggered")
                            break 
                cfg.save_checkpoint(self.model, self.optimizer, epoch_n, f'{self.repository_id}/final.pth') if self.save_model else None
        
        else:
            print("Training from scratch!...")
            for epoch_n in range(self.args.epochs):
                train_loss = self.train_model(epoch_n)
                val_loss, accuracy = self.validate(epoch_n)

                if self.early_stopping:
                    if self.early_stopping_model.step(val_loss):
                        print("Early stopping triggered")
                        break 
            cfg.save_checkpoint(self.model, self.optimizer, epoch_n, f'{self.repository_id}/final.pth') if self.save_model else None
        
        if classification_head:
            dataloader = self.classification_dataloader if self.classification_dataloader is not None else self.train_dataloader
            print("Training classification head!")
            # self.model = cfg.load_checkpoint(self.model, self.model_weights)
            # self.model.to(self.device)
            if self.args.classification_head == 'gmm':
                embeddings, _ = self.extract_embeddings(dataloader=dataloader)
                print("Fitting GMM")
                gmm = GaussianMixture(n_components=self.num_labels, covariance_type='diag', random_state=self.args.seed, max_iter=3000, n_init=10)
                gmm.fit(embeddings)
                return self.model, gmm
            elif self.args.classification_head == 'knn':
                embeddings, _= self.extract_embeddings(dataloader=dataloader)
                print("Fitting KNN")
                knn = KNeighborsClassifier(n_neighbors=5)
                knn.fit(embeddings, self.train_dataset.labels)
                return self.model, knn
            elif self.args.classification_head == 'linear' or self.args.classification_head == 'mlp':
                classification_model = AuthorshipClassificationLLM(self.model, num_labels=self.num_labels, head_type=self.args.classification_head)
                classification_loss = nn.CrossEntropyLoss()
                classification_model.to(self.device)
                classification_loss.to(self.device)
                classification_optimizer = optim.AdamW(classification_model.parameters(), lr=self.args.learning_rate_classification)
                for epoch_n in range(self.args.epochs_classification):
                    train_loss = self.train_classification(classification_model, classification_loss, classification_optimizer, epoch_n, dataloader)
                    val_loss = self.validate_classification(classification_model, classification_loss, epoch_n, dataloader)
                    if self.early_stopping:
                        if self.early_stopping_classif.step(val_loss):
                            print("Early stopping triggered")
                            break 
                        
                cfg.save_checkpoint(classification_model, classification_optimizer, epoch_n, f'{self.repository_id}/classification_final.pth') if self.save_model else None
                return self.model, classification_model
        return self.model, None 
    
    def extract_embeddings(self, dataloader=None):
        """
        Extracts embeddings from the model and saves them to a file.
        """
        self.model.eval()
        all_embeddings = []
        all_labels = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting embeddings"):
                embeddings = self.model(batch)
                labels = batch['label']
                anchor_embeddings = embeddings['anchor'].cpu().numpy()
                all_embeddings.append(anchor_embeddings)
                all_labels.extend(labels.cpu().numpy())

        all_embeddings = np.vstack(all_embeddings)
        all_labels = np.array(all_labels)
        return all_embeddings, all_labels
    

    def train_classification(self, classification_model, criterion_classification, optimizer_classification, epoch_n, data_loader):
        """
        Trains only the cassification head of the model for one epoch.
        Args:
            classificaion_model (torch.nn.Module): The classification head model to be validated.
            optimizer_classification (torch.optim.Optimizer): The optimizer for updating model parameters.
            criterion_classification (int): The current epoch number.
            epoch_n (int): The current epoch number.
            data_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        Returns:
            float: The average loss over the training dataset.
        """
        
        classification_model.train()
        total_loss = 0
        correct = 0
        for i,batch in enumerate(tqdm(data_loader, desc=f"Train Epoch {epoch_n+1}/{self.args.epochs_classification}")):
            labels = batch['label'].to(self.device)
            optimizer_classification.zero_grad()
            out_model = classification_model(batch)
            loss_value = criterion_classification(out_model, labels)
            loss_value.backward()
            optimizer_classification.step()
            total_loss += loss_value.item()
            preds = out_model.argmax(dim=1)
            correct += (preds == labels).sum().item()

        train_loss = total_loss / len(data_loader)
        metrics = {
            "Loss": train_loss,
            "epoch": epoch_n, 
            "Accuracy": correct / len(data_loader.dataset)
        }
        self.logger._log_metrics(metrics, phase='Train_classificaion_epochs')
        return total_loss / len(data_loader)
    
    @torch.no_grad()
    def validate_classification(self, classificaion_model, criterion_classification, epoch_n, data_loader):
        """
        Validates the classification head model on the validation dataset.
        Args:
            classificaion_model (torch.nn.Module): The classification head model to be validated.
            criterion_classification (callable): The loss function.
            epoch_n (int): The current epoch number.
            data_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        Returns:
            float: The average loss over the validation dataset.
        """
        
        classificaion_model.eval()
        total_loss = 0
        correct = 0
        with torch.no_grad():
            for i,batch in enumerate(tqdm(self.val_dataloaders[0], desc=f"Val Epoch {epoch_n+1}/{self.args.epochs_classification}")):
                labels = batch['label'].to(self.device)
                outputs = classificaion_model(batch)
                loss_value = criterion_classification(outputs, labels)
                total_loss += loss_value.item()
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                         
        val_loss = total_loss / len(self.val_dataloaders[0])
        metrics = {
            "Loss": val_loss,
            "epoch": epoch_n,
            "Accuracy": correct / len(self.val_dataloaders[0].dataset)
        }
        self.logger._log_metrics(metrics, phase='Val_classificaion_epochs')
        return val_loss

    def train_model(self, epoch_n):
        """
        Trains only the the given LLM model using the provided training data.
        Args:
            epoch_n (int): The current epoch number.
        Returns:
            float: The average loss over the training data.
        """
        
        self.model.train()
        total_loss = 0
        current_loss = 0.0
        correct_count = 0
        all_feats = []
        all_labels =[]        
        iters = len(self.train_dataloader)
        for i,batch in enumerate(tqdm(self.train_dataloader, desc=f"Train Epoch {epoch_n+1}/{self.args.epochs}")):
            labels = batch['label'].to(torch.float32).to(self.device)
            self.optimizer.zero_grad()
            if self.args.loss_function == 'bce':
                embeddings = self.model(batch['text'], mode = 'classification').squeeze()
                loss_value = self.criterion(embeddings, labels)
            else:
                embeddings = self.model(batch, mode = 'triplet')
                all_feats.append(embeddings['anchor'].detach().cpu().numpy())
                all_labels.extend(labels.detach().cpu().numpy())
                loss_value = self.criterion(embeddings['anchor'], embeddings['positive'], embeddings['negative'])
            
            
            loss_value.backward()
            total_loss += loss_value.item()
            current_loss += loss_value.item()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 50)
            
            self.optimizer.step()
            self.lr_scheduler.step(epoch_n + i / iters)
            
            # Accuracy calculation
            if self.args.loss_function == 'bce':
                preds = embeddings.argmax(dim=1)
                correct_count += (preds == labels).sum().item()
            else:
                distance_positive = self.distance_function(embeddings['anchor'], embeddings['positive'])
                distance_negative = self.distance_function(embeddings['anchor'], embeddings['negative'])
                correct_count += torch.sum(distance_positive < distance_negative).item()
            

                
        train_loss = total_loss / len(self.train_dataloader)
        metrics = {
            "Loss": train_loss,
            "Accuracy": correct_count / len(self.train_dataloader.dataset),
            "epoch": epoch_n,
        }
        self.logger._log_metrics(metrics, phase='Train_epoch')
        if self.log_plots and self.args.loss_function != 'bce':
            all_feats = np.vstack(all_feats)
            all_labels = np.array(all_labels)
            print(f"Getting t-SNE plot for train set")
            fig = get_tsne_fig(all_feats, all_labels, f"t-SNE on train : Epoch {epoch_n}")
            plt.savefig(f"{self.repository_id}/tsne_train.png")
            self.logger.log_figure(fig, f"t-SNE on train", epoch_n)
            
        return total_loss / len(self.train_dataloader)
    
    @torch.no_grad()
    def validate(self, epoch_n):
        """
        Validates the model on the validation dataset.
        Args:
            epoch_n (int): The current epoch number.
        Returns:
            float: The average validation loss over the entire validation dataset.
        """
        
        self.model.eval()
        with torch.no_grad():
            for val_dataloader in self.val_dataloaders:
                total_loss = 0
                current_loss = 0.0
                correct_count = 0
                total = 0
                all_feats = []
                all_labels = []
                for i,batch in enumerate(tqdm(val_dataloader, desc=f"Val Epoch {val_dataloader.dataset} {epoch_n+1}/{self.args.epochs}")):
                    labels = batch['label'].to(self.device)
                    if self.args.loss_function == 'bce':
                        embeddings = self.model(batch['text'], mode = 'classification')
                        loss_value = self.criterion(embeddings, labels)
                    else:
                        embeddings = self.model(batch)
                        features = embeddings['anchor']
                        all_feats.append(features)
                        all_labels.extend(labels.detach().cpu().numpy())
                        loss_value = self.criterion(embeddings['anchor'], embeddings['positive'], embeddings['negative'])

                    total_loss += loss_value.item()
                    current_loss += loss_value.item()

                    # Accuracy calculation
                    if self.args.loss_function == 'bce':
                        preds = embeddings.argmax(dim=1)
                        correct_count += (preds == labels).sum().item()
                    else:   
                        distance_positive = self.distance_function(embeddings['anchor'], embeddings['positive'])
                        distance_negative = self.distance_function(embeddings['anchor'], embeddings['negative'])
                        correct_count += torch.sum(distance_positive < distance_negative).item()

                    total += len(batch['label'])

                    
                val_loss = total_loss / len(val_dataloader.dataset)
                metrics = {
                        "Loss": val_loss,
                        "Accuracy": correct_count / total,
                        "epoch": epoch_n,
                        }
                self.logger._log_metrics(metrics, phase=f'Val_epoch_{val_dataloader.dataset}')
                if self.log_plots and self.args.loss_function != 'bce':
                    all_feats = torch.cat(all_feats, dim=0).cpu().numpy()
                    all_labels = np.array(all_labels)
                    print(f"Getting t-SNE plot for validation set")
                    fig = get_tsne_fig(all_feats, all_labels, f"t-SNE on validaation : Epoch {epoch_n}")
                    plt.savefig(f"{self.repository_id}/tsne_validation.png")
                    self.logger.log_figure(fig, f"t-SNE on validation_{val_dataloader.dataset}", epoch_n)
        return val_loss, correct_count / total
    

    def get_criterion(self, margin=0.5):
        """
            Returns the appropriate loss function based on the specified loss function type.
            Args:
                margin (float, optional): The margin value for certain loss functions. Defaults to 0.5.
            Returns:
                callable: The loss function to be used during training.
        """
        margin = self.args.margin if margin is None else margin
        if self.args.loss_function == 'triplet':
            return TripletLoss(margin=margin, distance_function=self.args.distance_function)
        elif self.args.loss_function == 'ada_triplet':
            return AdaTriplet(lambda_=self.args.at_lambda)
        elif self.args.loss_function == 'contrastive':
            return ContrastiveLoss(margin=margin)
        elif self.args.loss_function == 'cos2':
            return SquaredCosineSimilarityLoss()
        elif self.args.loss_function == 'triplet_temperature':
            return TripletLossTemperature()
        elif self.args.loss_function == 'bce':
            return nn.BCELoss()
        else:
            raise ValueError("Invalid loss_function value. Must be 'triplet', 'contrastive', or 'cos2'")
    
    def get_optimizer(self):
        return optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
    
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
        elif self.args.lr_scheduler == 'cosine_warmup':
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-7, last_epoch=-1)
        elif self.args.lr_scheduler == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        else:
            raise ValueError("Invalid lr_scheduler value. Must be 'linear_warmup', 'linear', 'cosine', 'cosine_warmup' or 'plateau'")


