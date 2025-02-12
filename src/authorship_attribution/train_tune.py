from torch.utils.data import DataLoader
from loss_functions import TripletLoss
import torch
import os
import config as cfg
from model import AuthorshipLLM
import torch.optim as optim
from ray.train import Checkpoint, get_checkpoint
from pathlib import Path
import ray.cloudpickle as pickle
import os 
from ray import train
import tempfile

def train_tune(config, train_dataset, val_dataset, device, args):
    """
    Trains and tunes a model using the provided configuration, datasets, and device.
    Args:
        config (dict): Configuration dictionary containing training parameters such as batch size, learning rate, and number of epochs.
        train_dataset (Dataset): The dataset used for training.
        val_dataset (Dataset): The dataset used for validation.
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
    layers = config["layers"]
    mlp_layers = config["mlp_layers"]
    weight_decay = config["weight_decay"]
    num_workers = config["workers"]
    model = AuthorshipLLM(args.model_name)
    device = cfg.get_device()
    model.to(device)
    criterion = TripletLoss(margin=margin)
    criterion = criterion.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
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
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"]*len(train_dataloader), eta_min=0, last_epoch=-1)

    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        total_loss = 0.0
        for batch in train_dataloader:
            embeddings = model(batch)
            optimizer.zero_grad()
            loss = criterion(embeddings['anchor'], embeddings['positive'], embeddings['negative'])
            
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            total_loss += loss_value.item()
            
        # Validate the model
        val_loss = 0.0
        correct_count = 0
        avg_val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                embeddings = model(batch)
                loss_value = criterion(embeddings['anchor'], embeddings['positive'], embeddings['negative'])
                val_loss += loss_value.item()

                distance_positive = distance_function(embeddings['anchor'], embeddings['positive'])
                distance_negative = distance_function(embeddings['anchor'], embeddings['negative'])
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
