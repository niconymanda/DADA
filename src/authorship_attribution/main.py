from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from loss_functions import TripletLoss
import torch
import os
import config as cfg
from dataset import AuthorTripletLossDataset
from test_model import TesterAuthorshipAttribution
from train import TrainerAuthorshipAttribution
from model import AuthorshipLLM
from peft import LoraConfig, get_peft_model
import time
import pandas as pd

def main(args):
    cfg.init_env(args)
    data, spoofed_data, rest_of_data, author_id_map = cfg.load_data(args)
    current_time = time.strftime("%Y%m%d-%H%M%S")
    repository_id = f"/data/iivanova-23/output/train_in_the_wild/n_authors_{len(author_id_map.keys())}/{args.model_name}_{args.batch_size}_{args.epochs}_{current_time}"
    os.makedirs(repository_id, exist_ok=True)
    print(f"Repository ID: {repository_id}")
    # index_set = "/home/infres/iivanova-23/DADA/Data/index_10_authors.json"
    # _, val_rest = train_test_split(rest_of_data, test_size=0.3, stratify=rest_of_data['label'], random_state=args.seed)
    # train_data, val_data = train_test_split(data, test_size=0.3, stratify=data['label'], random_state=args.seed)
    # train_dataset = AuthorTripletLossDataset(train_data, args.model_name, train=True, predefined_set=None)
    # # val_data = pd.concat([val_data, val_rest])
    # val_dataset = AuthorTripletLossDataset(val_data, args.model_name, train=True, predefined_set=None)
    # train_dataset = AuthorTripletLossDataset(data, args.model_name, train=True, predefined_set=index_set)
    # val_dataset = AuthorTripletLossDataset(data, args.model_name, train=False, predefined_set=index_set)
    in_the_wild = pd.read_csv("/data/iivanova-23/DataDADA/train_merged.csv")
    in_the_wild = in_the_wild[in_the_wild['label']<args.authors_to_train]
    index_set = '/data/iivanova-23/DataDADA/index_10_authors_merged.json'
    train_dataset = AuthorTripletLossDataset(in_the_wild, args.model_name, train=True, predefined_set=index_set)
    val_dataset = AuthorTripletLossDataset(in_the_wild, args.model_name, train=False, predefined_set=index_set)
    
    spoofed_test_dataset = AuthorTripletLossDataset(in_the_wild, args.model_name, mode = 'spoof')
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
            
    num_workers = 2
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    spoofed_data_loader = DataLoader(spoofed_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
    # real_in_the_wild_loader = DataLoader(real_in_the_wild, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
    # spoof_in_the_wild_loader = DataLoader(spoof_in_the_wild, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
    # lora_config = LoraConfig(
    #     r=4,  
    #     lora_alpha=16,  
    #     lora_dropout=0.1,
    #     target_modules=['query_proj', 'value_proj'],
    # )

    model = AuthorshipLLM(args.model_name, 
                          dropout_rate=0.1, 
                          out_features=1024, 
                          max_length=64, 
                          num_layers=args.mlp_layers, 
                          freeze_encoder=False, 
                          use_layers=args.hidden_layers)
    # model = get_peft_model(model, lora_config)
    # path = '/home/infres/iivanova-23/DADA/output/n_authors_3/microsoft/deberta-v3-large_16_14_20241204-171443/final.pth'
    # path='/home/infres/iivanova-23/DADA/output/n_authors_6/answerdotai/ModernBERT-large_16_14_20250109-092429/final.pth'
    # path='/data/iivanova-23/output/n_authors_6/answerdotai/ModernBERT-large_16_14_20250109-095045/final.pth'
    # path = '/data/iivanova-23/output/margin_0.4/n_authors_10/google-t5/t5-large_16_20_20250120-113103/final.pth'
    trainer = TrainerAuthorshipAttribution(model=model,
                                           train_dataloader=train_dataloader,
                                           val_dataloader=val_dataloader,
                                           args=args,
                                           repository_id=repository_id,
                                           author_id_map=author_id_map,
                                           report_to='tensorboard',
                                           early_stopping=True,
                                           save_model=True,
                                           model_weights=None
                                           )
    model, classification_model = trainer.train(classification_head=False)
    # loaded_model.load_state_dict(torch.load("output/n_authors_3/microsoft/deberta-v3-small_16_10_20241128-150757/final.pth"))
    print("Training finished")
    tester = TesterAuthorshipAttribution(model=model, 
                    classification_model=classification_model,
                    repository_id=repository_id, 
                    author_id_map=author_id_map,
                    args=args)
    
    tester.test(val_dataloader, spoofed_data_loader)
    # print("Test In the Wild")
    # tester.test(real_in_the_wild_loader, spoof_in_the_wild_loader, 'In the Wild')

if __name__ == "__main__":
    args = cfg.get_args()
    main(args)


