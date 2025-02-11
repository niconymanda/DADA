from torch.utils.data import DataLoader, ConcatDataset
from sklearn.model_selection import train_test_split
from loss_functions import TripletLoss
import torch
import os
import config as cfg
from dataset import AuthorTripletLossDataset, AuthorClassificationDataset
from SpeechCLR.utils.datasets import VoxCeleb2Dataset
from test_model import TesterAuthorshipAttribution
from train import TrainerAuthorshipAttribution
from model import AuthorshipLLM
import time
import pandas as pd


def main(args):
    cfg.init_env(args)
    # data, rest_of_data, author_id_map = cfg.load_data(args)
    current_time = time.strftime("%Y%m%d-%H%M%S")
    # repository_id = f"/data/iivanova-23/output/wiki/n_authors_{len(author_id_map.keys())}/{args.model_name}_{args.batch_size}_{args.epochs}_{current_time}"
    repository_id = f"/data/iivanova-23/output/inthewild/{args.model_name}_{args.batch_size}_{args.epochs}_{current_time}"
    os.makedirs(repository_id, exist_ok=True)
    print(f"Repository ID: {repository_id}")
    # index_set = "/data/iivanova-23/data/wiki/meta_wiki.json"
    # in_the_wild = pd.read_csv("/data/iivanova-23/data/inthewild/inthewild_transcriptions_final.csv")
    # in_the_wild_real = in_the_wild[in_the_wild['type']!='spoof']
    # # author_id_map = {author: i for i, author in enumerate(in_the_wild_real['label'].unique())}
    # author_id_map = in_the_wild[['label', 'author_name']].drop_duplicates().set_index('label').to_dict()['author_name']
    
    # print(f"Number of in the wild samples: {len(in_the_wild_real)}")
    data = pd.read_csv(args.data[0])
    train, val = train_test_split(data, test_size=0.3, stratify=data['label'], random_state=args.seed, shuffle=True)
    author_id_map = data[['label', 'author_name']].drop_duplicates().set_index('label').to_dict()['author_name']
    
    train_dataset = AuthorTripletLossDataset(train, train=True)
    val_dataset = AuthorTripletLossDataset(val, train=True)
    
    train_dataset_vox = VoxCeleb2Dataset(args.data[1], split="train", text_only=True, max_samples=5000, mode='triplet')
    val_dataset_vox = VoxCeleb2Dataset(args.data[1], split="val", text_only=True, max_samples=2000, mode='triplet')
    train_concat = ConcatDataset([train_dataset, train_dataset_vox])
    # val_dataset = VoxCeleb2Dataset(args.data, split="val", text_only=True, max_samples=2000, mode='triplet')
    # author_id_map = train_dataset.id_to_author
    # in_the_wild = pd.read_csv("/data/iivanova-23/data/wild_transcription_meta.csv")
    # index_wild = "/data/iivanova-23/data/index_10_authors_wild.json"
    # in_the_wild = in_the_wild[in_the_wild['label']<args.authors_to_train]

           
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    
    train_dataloader = DataLoader(train_concat, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataloader_vox = DataLoader(val_dataset_vox, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    train_dataloader_classification = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    # in_the_wild_real_loader = DataLoader(in_the_wild_real_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    # spoofed_data_loader = DataLoader(spoofed_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    # real_in_the_wild_loader = DataLoader(real_in_the_wild, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    # spoof_in_the_wild_loader = DataLoader(spoof_in_the_wild, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


    model = AuthorshipLLM(args.model_name, 
                          dropout_rate=0.1, 
                          out_features=512, 
                          max_length=64, 
                          num_layers=args.mlp_layers, 
                          freeze_encoder=False, 
                          use_layers=args.hidden_layers)
    
    trainer = TrainerAuthorshipAttribution(model=model,
                                           train_dataloader=train_dataloader,
                                           val_dataloaders=[val_dataloader, val_dataloader_vox],
                                           args=args,
                                           repository_id=repository_id,
                                           author_id_map=author_id_map,
                                           report_to='tensorboard',
                                           early_stopping=True,
                                           save_model=True,
                                           additional_training=False,
                                           log_plots=False, 
                                           classification_dataloader=train_dataloader_classification)
    
    model, classification_model = trainer.train(classification_head=True)
    print("Training finished")
    tester = TesterAuthorshipAttribution(model=model, 
                    classification_model=classification_model,
                    repository_id=repository_id, 
                    author_id_map=author_id_map,
                    args=args, 
                    classification=False)
    
    tester.test(val_dataloader, 'Wiki')
    tester.test(val_dataloader_vox, 'VoxCeleb2')  

if __name__ == "__main__":
    args = cfg.get_args()
    main(args)


