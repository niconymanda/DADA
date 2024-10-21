import torch
from tqdm import tqdm  
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import ray
from ray import tune
from loss_functions import TripletLoss

def train(model, loss_fn, train_dataloader, optimizer, device, writer, epoch_n, args):
    model.train()
    total_loss = 0
    current_loss = 0.0
    for i,batch in enumerate(tqdm(train_dataloader)):
        anchor_input_ids = batch['anchor_input_ids'].to(device)
        anchor_attention_mask = batch['anchor_attention_mask'].to(device)
        positive_input_ids = batch['positive_input_ids'].to(device)
        positive_attention_mask = batch['positive_attention_mask'].to(device)
        negative_input_ids = batch['negative_input_ids'].to(device)
        negative_attention_mask = batch['negative_attention_mask'].to(device)
        
        optimizer.zero_grad()
        anchor_embeddins = model(anchor_input_ids, anchor_attention_mask)
        positive_embeddins = model(positive_input_ids, positive_attention_mask)
        negative_embeddins = model(negative_input_ids, negative_attention_mask)
        loss_value = loss_fn(anchor_embeddins, positive_embeddins, negative_embeddins)
        loss_value.backward()
        optimizer.step()
        total_loss += loss_value.item()
        
        current_loss += loss_value.item()
        if i % args.logging_step == args.logging_step - 1:
            writer.add_scalar('Loss/Train', current_loss / args.logging_step, epoch_n * len(train_dataloader) + i)
            current_loss = 0.0
            
    return total_loss / len(train_dataloader)

def validate(model, loss_fn, val_dataloader, device, writer, epoch_n, args):
    model.eval()
    total_loss = 0
    curret_loss = 0.0
    with torch.no_grad():
        for i,batch in enumerate(tqdm(val_dataloader)):
            anchor_input_ids = batch['anchor_input_ids'].to(device)
            anchor_attention_mask = batch['anchor_attention_mask'].to(device)
            positive_input_ids = batch['positive_input_ids'].to(device)
            positive_attention_mask = batch['positive_attention_mask'].to(device)
            negative_input_ids = batch['negative_input_ids'].to(device)
            negative_attention_mask = batch['negative_attention_mask'].to(device)
            
            anchor_embeddins = model(anchor_input_ids, anchor_attention_mask)
            positive_embeddins = model(positive_input_ids, positive_attention_mask)
            negative_embeddins = model(negative_input_ids, negative_attention_mask)
            loss_value = loss_fn(anchor_embeddins, positive_embeddins, negative_embeddins)
            total_loss += loss_value.item()
            curret_loss += loss_value.item()
            if i % args.logging_step == args.logging_step - 1:
                print(f"Validation loss: {curret_loss / args.logging_step}")
                writer.add_scalar('Loss/Val', curret_loss / args.logging_step, epoch_n * len(val_dataloader) + i)
                curret_loss = 0.0       
    return total_loss / len(val_dataloader)



def train_tune(config, train_dataset, val_dataset, model, device):
    batch_size = config["batch_size"]
    learning_rate = config["lr"]
    # margin = config["margin"]
    
    loss_fn = TripletLoss(margin=0.5)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    
    model = model.to(device)
    loss_fn = loss_fn.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    patience = 3
    patience_counter = 0
    
    for epoch in range(config["epochs"]):
        # Training step
        model.train()
        total_loss = 0.0
        for batch in train_dataloader:
            anchor_input_ids = batch['anchor_input_ids'].to(device)
            anchor_attention_mask = batch['anchor_attention_mask'].to(device)
            positive_input_ids = batch['positive_input_ids'].to(device)
            positive_attention_mask = batch['positive_attention_mask'].to(device)
            negative_input_ids = batch['negative_input_ids'].to(device)
            negative_attention_mask = batch['negative_attention_mask'].to(device)

            optimizer.zero_grad()

            anchor_embeddings = model(anchor_input_ids, anchor_attention_mask)
            positive_embeddings = model(positive_input_ids, positive_attention_mask)
            negative_embeddings = model(negative_input_ids, negative_attention_mask)

            loss_value = loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)
            loss_value.backward()

            optimizer.step()
            total_loss += loss_value.item()

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                anchor_input_ids = batch['anchor_input_ids'].to(device)
                anchor_attention_mask = batch['anchor_attention_mask'].to(device)
                positive_input_ids = batch['positive_input_ids'].to(device)
                positive_attention_mask = batch['positive_attention_mask'].to(device)
                negative_input_ids = batch['negative_input_ids'].to(device)
                negative_attention_mask = batch['negative_attention_mask'].to(device)

                anchor_embeddings = model(anchor_input_ids, anchor_attention_mask)
                positive_embeddings = model(positive_input_ids, positive_attention_mask)
                negative_embeddings = model(negative_input_ids, negative_attention_mask)

                loss_value = loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)
                val_loss += loss_value.item()

        avg_val_loss = val_loss / len(val_dataloader)
        tune.report(val_loss=avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0 
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} with best validation loss: {best_val_loss}")
            break


            
