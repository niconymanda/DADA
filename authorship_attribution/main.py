from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from model import AuthorshipLLM
from loss_functions import TripletLoss
import torch
import os
import config
from dataset import AuthorTripletLossDataset
from train import train, validate
from test import test_model, plot_tsne_for_authors
from early_stopping import EarlyStopping

def main(args):
    config.init_env(args)
    data, spoofed_data = config.load_data(args.data)
    
    train_data, temp_data = train_test_split(data, test_size=0.3, stratify=data['label'])
    val_data, test_data = train_test_split(temp_data, test_size=0.6, stratify=temp_data['label'])
    train_dataset = AuthorTripletLossDataset(train_data, args.model_name, train=True)
    val_dataset = AuthorTripletLossDataset(val_data, args.model_name, train=True)
    test_dataset = AuthorTripletLossDataset(test_data, args.model_name, train=False)
    spoofed_test_dataset = AuthorTripletLossDataset(spoofed_data, args.model_name, train=False)
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
    
    # for epoch_n in range(args.epochs):
    #     train_loss = train(model, loss_fn, train_dataloader, optimizer, device, writer, epoch_n, args)
    #     writer.add_scalar('Loss/Train_ep', train_loss, epoch_n)
    #     print(f"Epoch {epoch_n + 1}: Train Loss: {train_loss}")
    #     val_loss = validate(model, loss_fn, val_dataloader, device, writer, epoch_n, args)
    #     writer.add_scalar('Loss/Val_ep', val_loss, epoch_n)
    #     print(f"Epoch {epoch_n + 1}: Val Loss: {val_loss}")
    #     if early_stopping.step(val_loss):
    #         print("Early stopping triggered")
    #         break 
    #     torch.save(model.state_dict(), f'{repository_id}/{epoch_n+1}.pth')
    # config.save_checkpoint(model, optimizer, epoch_n, f'{repository_id}/final.pth')
    config.load_checkpoint(model, optimizer, f'{repository_id}/final.pth')
    writer.close()
    acc = test_model(model, test_dataloader, device)
    print(f"Test Accuracy : {acc:.4f}")
    plot_tsne_for_authors(model, test_dataloader, device, repository_id)
    # acc_sp = test_model(model, spoofed_data_loader, device)
    # print(f"Spoofed Test Accuracy : {acc_sp:.4f}")
    
if __name__ == "__main__":
    args = config.get_args()
    main(args)