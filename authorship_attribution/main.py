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
from transformers import get_scheduler, get_cosine_with_hard_restarts_schedule_with_warmup, get_linear_schedule_with_warmup
import time

def main(args):
    config.init_env(args)
    data, spoofed_data, author_id_map = config.load_data(args)
    current_time = time.strftime("%Y%m%d-%H%M%S")
    repository_id = f"./output/n_authors_{len(author_id_map.keys())}/{args.model_name}_{args.batch_size}_{args.epochs}_{current_time}"
    os.makedirs(repository_id, exist_ok=True)
    
    train_data, temp_data = train_test_split(data, test_size=0.4, stratify=data['label'])
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
    
    loss_fn = TripletLoss(margin=0.1, distance_function=args.distance_function)
    # loss_fn = torch.nn.TripletMarginLoss(margin=1.0)
    model = AuthorshipLLM(args.model_name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # num_training_steps = args.epochs * len(train_dataloader)
    # # lr_scheduler = get_scheduler(
    # #     "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    # # )
    # # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, eta_min=1e-4)
    num_training_steps = args.epochs * (len(train_dataset) // args.batch_size) 
    warmup_steps = int(0.1 * num_training_steps)  
    # # lr_scheduler =    (
    # #     optimizer,
    # #     num_warmup_steps=warmup_steps,
    # #     num_training_steps=num_training_steps,
    # #     num_cycles=5
    # # )

    # lr_scheduler = get_linear_schedule_with_warmup(
    #     optimizer=optimizer,
    #     num_warmup_steps=warmup_steps,
    #     num_training_steps=num_training_steps
    # )
    trainer = TrainerAuthorshipAttribution(model=model,
                                            loss_fn=loss_fn,  
                                            optimizer=optimizer,
                                            lr_scheduler=args.lr_scheduler,
                                            train_dataloader=train_dataloader,
                                            val_dataloader=val_dataloader,
                                            args=args,
                                            repository_id=repository_id,
                                            author_id_map=author_id_map,
                                            num_training_steps=num_training_steps,
                                            warmup_steps=warmup_steps,
                                            report_to='wandb',
                                            early_stopping=True,
                                            save_model=False,
                                            distance_function=args.distance_function)
    model, classification_model = trainer.train(classification_head=True)
    
    #Load model from checkpoint
    # repo = '/home/infres/iivanova-23/DADA/output/n_authors_3/FacebookAI/roberta-large_32_20'
    # model = AuthorshipLLM(args.model_name)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    # config.load_checkpoint(model, optimizer, f'{repo}/final.pth')
    # classification_model = AuthorshipClassificationLLM(model, num_labels=len(author_id_map.keys()))
    # classification_optimizer = torch.optim.Adam(classification_model.parameters(), lr=args.learning_rate)
    # config.load_checkpoint(classification_model, classification_optimizer, f'{repo}/classification_final.pth')
    
    results = {}
    results_spoofed = {}
    tester = TesterAuthorshipAttribution(model=model, 
                    classification_model=classification_model,
                    repository_id=repository_id, 
                    author_id_map=author_id_map,
                    distance_function=args.distance_function)
    acc = tester.test_abx_accuracy(test_dataloader)
    results['abx_accuracy'] = acc
    print(f"Test ABX Accuracy : {acc:.4f}")
    tester.plot_tsne_for_authors(test_dataloader)
    tester.plot_cosine_distence_distribution()
    acc_sp = tester.test_abx_accuracy(spoofed_data_loader)
    print(f"Spoofed Test ABX Accuracy : {acc_sp:.4f}")
    results_spoofed['abx_accuracy'] = acc_sp
    config.save_model_config(args,output_path=f"{repository_id}/model_config.json")
    #Test results on classification model
    classif_results = tester.test_classification(test_dataloader)
    results.update(classif_results)
    config.write_results_to_file(results, './output/results.txt', args)
    
    classif_results_spoofed = tester.test_classification(spoofed_data_loader)
    results_spoofed.update(classif_results_spoofed)
    config.write_results_to_file(results_spoofed, './output/results_spoofed.txt', args)
    
if __name__ == "__main__":
    args = config.get_args()
    main(args)