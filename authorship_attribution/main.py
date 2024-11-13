from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from loss_functions import TripletLoss
import torch
import os
import config
from dataset import AuthorTripletLossDataset
from test_model import TesterAuthorshipAttribution
from train import TrainerAuthorshipAttribution
from model import AuthorshipLLM, AuthorshipClassificationLLM
import time

def main(args):
    config.init_env(args)
    data, spoofed_data, author_id_map = config.load_data(args)
    current_time = time.strftime("%Y%m%d-%H%M%S")
    repository_id = f"/data/iivanova-23/output_data/n_authors_{len(author_id_map.keys())}/{args.model_name}_{args.batch_size}_{args.epochs}_{current_time}"
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
            
    num_workers = 0
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
    spoofed_data_loader = DataLoader(spoofed_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
    
    model = AuthorshipLLM(args.model_name)

    trainer = TrainerAuthorshipAttribution(model=model,
                                           train_dataloader=train_dataloader,
                                           val_dataloader=val_dataloader,
                                           args=args,
                                           repository_id=repository_id,
                                           author_id_map=author_id_map,
                                           report_to='tensorboard',
                                           early_stopping=True,
                                           save_model=False,
                                           )
    model, classification_model = trainer.train(classification_head=False)
    
    #Load model from checkpoint
    # repo = '/home/infres/iivanova-23/DADA/output/n_authors_3/FacebookAI/roberta-large_32_20'
    # model = AuthorshipLLM(args.model_name)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    # config.load_checkpoint(model, optimizer, f'{repo}/final.pth')
    # classification_model = AuthorshipClassificationLLM(model, num_labels=len(author_id_map.keys()))
    # classification_optimizer = torch.optim.Adam(classification_model.parameters(), lr=args.learning_rate)
    # config.load_checkpoint(classification_model, classification_optimizer, f'{repo}/classification_final.pth')
    

    tester = TesterAuthorshipAttribution(model=model, 
                    classification_model=classification_model,
                    repository_id=repository_id, 
                    author_id_map=author_id_map,
                    args=args)
    
    tester.test(test_dataloader, spoofed_data_loader)


if __name__ == "__main__":
    args = config.get_args()
    main(args)


