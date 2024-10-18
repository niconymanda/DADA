import torch
from tqdm import tqdm  

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
    current_loss = 0.0  

    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_dataloader)):
            anchor_input_ids = batch['anchor_input_ids'].to(device)
            anchor_attention_mask = batch['anchor_attention_mask'].to(device)
            positive_input_ids = batch['positive_input_ids'].to(device)
            positive_attention_mask = batch['positive_attention_mask'].to(device)
            negative_input_ids = batch['negative_input_ids'].to(device)
            negative_attention_mask = batch['negative_attention_mask'].to(device)

            anchor_embeddings = model(anchor_input_ids, anchor_attention_mask)
            positive_embeddings = model(positive_input_ids, positive_attention_mask)
            negative_embeddings = model(negative_input_ids, negative_attention_mask)

            loss = loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)
            total_loss += loss.item()
            current_loss += loss.item()

            if i % args.logging_step == args.logging_step - 1:
                writer.add_scalar('Loss/Val', current_loss / args.logging_step, epoch_n * len(val_dataloader) + i)
                current_loss = 0.0  
                
    return total_loss / len(val_dataloader)

            
