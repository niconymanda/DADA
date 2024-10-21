import config
from sklearn.model_selection import train_test_split
from dataset import AuthorTripletLossDataset
from model import AuthorshipLLM
from loss_functions import TripletLoss
from train import train_tune
from ray.tune.schedulers import ASHAScheduler
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import torch
import test
from torch.utils.data import DataLoader

def main(args):
    config.init_env(args)
    data, spoofed_data, author_id_map = config.load_data(args)
    repository_id = f"./output/n_authors_{len(author_id_map.keys())}/{args.model_name}_tune"
    
    train_data, temp_data = train_test_split(data, test_size=0.3, stratify=data['label'])
    val_data, test_data = train_test_split(temp_data, test_size=0.6, stratify=temp_data['label'])
    train_dataset = AuthorTripletLossDataset(train_data, args.model_name, train=True)
    val_dataset = AuthorTripletLossDataset(val_data, args.model_name, train=True)
    test_dataset = AuthorTripletLossDataset(test_data, args.model_name, train=False)
    spoofed_test_dataset = AuthorTripletLossDataset(spoofed_data, args.model_name, train=False)
    
    model = AuthorshipLLM(args.model_name)
    device = config.get_device()
    
    config_tune = {
        "lr": tune.loguniform(1e-7, 1e-3),  
        "batch_size": tune.choice([4, 8, 16]), 
        # "margin": tune.uniform(0.1, 2.0),  
        "epochs": 10
    }

    scheduler = ASHAScheduler(
        metric="val_loss",  
        mode="min",         
        max_t=20,           
        grace_period=1,     
        reduction_factor=2  
    )
    
    analysis = tune.run(
    tune.with_parameters(train_tune, train_dataset=train_dataset, val_dataset=val_dataset, model=model, device=device),
    resources_per_trial={"gpu": 1},  
    config=config_tune,
    num_samples=10,  
    scheduler=scheduler,
    progress_reporter=CLIReporter(
        metric_columns=["val_loss", "training_iteration"]
    )
)  

    best_config = analysis.best_config
    print("Best hyperparameters found were: ", best_config)
    best_trial = analysis.get_best_trial("val_loss", mode="min", scope="all")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.last_result["val_loss"]))
    torch.save(model.state_dict(), f"{repository_id}/best_model.pth")
    
    test_dataloader = DataLoader(test_dataset, batch_size=best_config['batch_size'], shuffle=False)
    spoofed_data_loader = DataLoader(spoofed_test_dataset, batch_size=best_config['batch_size'], shuffle=False)

    acc = test.test_model(model, test_dataloader, device)
    print(f"Test Accuracy : {acc:.4f}")
    test.plot_tsne_for_authors(model, test_dataloader, device, repository_id, author_id_map)
    acc_sp = test.test_model(model, spoofed_data_loader, device)
    print(f"Spoofed Test Accuracy : {acc_sp:.4f}")
if __name__ == '__main__':
    args = config.get_args()
    main(args)