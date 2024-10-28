from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from loss_functions import TripletLoss
import torch
import os
import config
from dataset import AuthorTripletLossDataset
from test_model import TesterAuthorshipAttribution
from train import TrainerAuthorshipAttribution, AuthorshipClassificationLLM
from model import AuthorshipLLM

def main(args):
    config.init_env(args)
    data, spoofed_data, author_id_map = config.load_data(args)
    repository_id = f"./output/n_authors_{len(author_id_map.keys())}/{args.model_name}_{args.batch_size}_{args.epochs}"
    os.makedirs(repository_id, exist_ok=True)
    
    train_data, temp_data = train_test_split(data, test_size=0.3, stratify=data['label'])
    val_data, test_data = train_test_split(temp_data, test_size=0.5, stratify=temp_data['label'])
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
    
    loss_fn = TripletLoss(margin=1.5)
    model = AuthorshipLLM(args.model_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    

    
    trainer = TrainerAuthorshipAttribution(model=model,
                                           loss_fn=loss_fn,  
                                           optimizer=optimizer,
                                           train_dataloader=train_dataloader,
                                           val_dataloader=val_dataloader,
                                           args=args,
                                           repository_id=repository_id,
                                           author_id_map=author_id_map,
                                           report_to='wandb',
                                           early_stopping=True
                                           )
    model, classification_model = trainer.train(classification_head=True)
    # Test ABX accuracy on LLM model
    
    #Load model from checkpoint
    # config.load_checkpoint(model, optimizer, f'{repository_id}/final.pth')
    # classification_model = AuthorshipClassificationLLM(model, num_labels=len(author_id_map.keys()))
    # classification_optimizer = torch.optim.Adam(classification_model.parameters(), lr=args.learning_rate)
    # config.load_checkpoint(classification_model, classification_optimizer, f'{repository_id}/classification_final.pth')
    
    
    tester = TesterAuthorshipAttribution(model=model, 
                    classification_model=classification_model,
                    repository_id=repository_id, 
                    author_id_map=author_id_map)
    acc = tester.test_abx_accuracy(test_dataloader)
    print(f"Test ABX Accuracy : {acc:.4f}")
    tester.plot_tsne_for_authors(test_dataloader)
    acc_sp = tester.test_abx_accuracy(spoofed_data_loader)
    print(f"Spoofed Test ABX Accuracy : {acc_sp:.4f}")
    
    #Test results on classification model
    tester.test_classification(test_dataloader)
    
    
if __name__ == "__main__":
    args = config.get_args()
    main(args)