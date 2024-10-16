import config
from transformers import AutoTokenizer
from dataset import AuthorTripletLossDataset
from sklearn.model_selection import train_test_split
from loss_functions import TripletLoss
from model import AuthorshipLLM
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from early_stopping import EarlyStopping
import os
from torch.nn import functional as F    

def train(model, loss_fn, train_dataloader, optimizer, device, writer, epoch_n):
    model.train()
    total_loss = 0
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
        current_loss = total_loss
        if i % 100 == 99:
            print(f"Epoch {epoch_n + 1}, Batch {i + 1}: Loss: {loss_value.item()}")
            writer.add_scalar('Loss/Train', current_loss/100, epoch_n * len(train_dataloader) + i)
            current_loss = 0.0
    return total_loss / len(train_dataloader)

def validate(model, loss_fn, val_dataloader, device, writer, epoch_n):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i,batch in enumerate(tqdm(val_dataloader)):
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
            current_loss = total_loss
            if i % 100 == 99:
                writer.add_scalar('Loss/Val',
                                current_loss/100,
                                epoch_n * len(val_dataloader) + i)       
                current_loss = 0.0
    return total_loss / len(val_dataloader)
def test_model(model, test_dataloader, device):
    model.eval() 
    correct = 0
    total = 0
    
    with torch.no_grad(): 
        for batch in test_dataloader:
            anchor_input_ids = batch['anchor_input_ids'].to(device)
            anchor_attention_mask = batch['anchor_attention_mask'].to(device)
            positive_input_ids = batch['positive_input_ids'].to(device)
            positive_attention_mask = batch['positive_attention_mask'].to(device)
            negative_input_ids = batch['negative_input_ids'].to(device)
            negative_attention_mask = batch['negative_attention_mask'].to(device)

            anchor_embeddings = model(anchor_input_ids, anchor_attention_mask)
            positive_embeddings = model(positive_input_ids, positive_attention_mask)
            negative_embeddings = model(negative_input_ids, negative_attention_mask)
            pos_distance = F.pairwise_distance(anchor_embeddings, positive_embeddings)
            neg_distance = F.pairwise_distance(anchor_embeddings, negative_embeddings)
            
            correct += (pos_distance < neg_distance).sum().item()
            total += anchor_embeddings.size(0)
    
    accuracy = correct / total
    return accuracy
            
def main(args):
    config.init_env(args)
    data, spoofed_data = config.load_data(args.data)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    train_data, temp_data = train_test_split(data, test_size=0.3, stratify=data['label'])
    val_data, test_data = train_test_split(temp_data, test_size=0.6, stratify=temp_data['label'])

    train_dataset = AuthorTripletLossDataset(train_data, tokenizer)
    val_dataset = AuthorTripletLossDataset(val_data, tokenizer)
    test_dataset = AuthorTripletLossDataset(test_data, tokenizer)
    spoofed_test_dataset = AuthorTripletLossDataset(spoofed_data, tokenizer)
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    spoofed_data_loader = DataLoader(spoofed_test_dataset, batch_size=args.batch_size, shuffle=False)
    
    loss_fn = TripletLoss(margin=0.5)
    model = AuthorshipLLM(args.model_name)
    device = config.get_device()
    model.to(device)
    loss_fn.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    repository_id = f"./output/{args.model_name}_{args.batch_size}_{args.epochs}"
    os.makedirs(repository_id, exist_ok=True)

    writer = SummaryWriter(f"{repository_id}/logs")
    early_stopping = EarlyStopping(patience=args.early_stopping_patience)
    
    for epoch_n in range(args.epochs):
        train_loss = train(model, loss_fn, train_dataloader, optimizer, device, writer, epoch_n)
        writer.add_scalar('Loss/Train_ep', train_loss, epoch_n)
        print(f"Epoch {epoch_n + 1}: Train Loss: {train_loss}")
        val_loss = validate(model, loss_fn, val_dataloader, device, writer, epoch_n)
        writer.add_scalar('Loss/Val_ep', val_loss, epoch_n)
        print(f"Epoch {epoch_n + 1}: Val Loss: {val_loss}")
        if early_stopping.step(val_loss):
            print("Early stopping triggered")
            break 
        torch.save(model.state_dict(), f'{repository_id}/{epoch_n+1}.pth')
    writer.close()
    acc = test_model(model, test_dataloader, device)
    print(f"Test Accuracy : {acc:.4f}")
    acc_sp = test_model(model, spoofed_data_loader, device)
    print(f"Spoofed Test Accuracy : {acc_sp:.4f}")
    
if __name__ == "__main__":
    args = config.get_args()
    main(args)