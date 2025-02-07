import config
from sklearn.model_selection import train_test_split
from dataset import AuthorTripletLossDataset
from model import AuthorshipLLM
from ray.tune.schedulers import ASHAScheduler
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import torch
from torch.utils.data import DataLoader
from test_model import TesterAuthorshipAttribution
import os
import pickle
from pathlib import Path
from train import train_tune

def main(args):
    config.init_env(args)
    data, spoofed_data, author_id_map = config.load_data(args)
    repository_id = f"./output/n_authors_{len(author_id_map.keys())}_tune/{args.model_name}"
    os.makedirs(repository_id, exist_ok=True)
    
    train_data, temp_data = train_test_split(data, test_size=0.3, stratify=data['label'])
    val_data, test_data = train_test_split(temp_data, test_size=0.5, stratify=temp_data['label'])
    train_dataset = AuthorTripletLossDataset(train_data, args.model_name, train=True)
    val_dataset = AuthorTripletLossDataset(val_data, args.model_name, train=True)
    test_dataset = AuthorTripletLossDataset(test_data, args.model_name, train=False)
    spoofed_test_dataset = AuthorTripletLossDataset(spoofed_data, args.model_name, train=False)
    
    model = AuthorshipLLM(args.model_name)
    device = config.get_device()
    
    config_tune = {
        "lr": tune.loguniform(1e-8, 1e-2),  
        "batch_size": tune.choice([8, 16, 32, 64]), 
        "margin": tune.choice([0.1, 0.5, 1.0, 1.5, 2.0]),	
        "epochs": 20
    }

    scheduler = ASHAScheduler(
        metric="loss",  
        mode="min",         
        max_t=20,           
        grace_period=1,     
        reduction_factor=2  
    )

    analysis = tune.run(
    tune.with_parameters(train_tune, train_dataset=train_dataset, val_dataset=val_dataset, model=model, device=device, args=args),
    resources_per_trial={"cpu":64, "gpu": 1},  
    config=config_tune,
    num_samples=10,  
    scheduler=scheduler,
    progress_reporter=CLIReporter(
        metric_columns=["loss", "training_iteration"]
    )
)  

    best_trial = analysis.get_best_trial("loss", mode="min", scope="all")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    torch.save(model.state_dict(), f"{repository_id}/best_model.pth")
    best_trained_model = AuthorshipLLM(args.model_name)
    best_trained_model.to(device)
    best_checkpoint = analysis.get_best_checkpoint(trial=best_trial, metric="accuracy", mode="max")
    with best_checkpoint.as_directory() as checkpoint_dir:
        data_path = Path(checkpoint_dir) / "data.pkl"
        with open(data_path, "rb") as fp:
            best_checkpoint_data = pickle.load(fp)

        best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])
        test_dataloader = DataLoader(test_dataset, batch_size=best_trial.config['batch_size'], shuffle=False)
        spoofed_data_loader = DataLoader(spoofed_test_dataset, batch_size=best_trial.config['batch_size'], shuffle=False)

        tester = TesterAuthorshipAttribution(model=best_trained_model,
                    repository_id=repository_id, 
                    author_id_map=author_id_map)
        acc = tester.test_abx_accuracy(test_dataloader)
        print("Best trial test set accuracy: {}".format(acc))
        tester.plot_tsne_for_authors(test_dataloader)
        acc_sp = tester.test_abx_accuracy(spoofed_data_loader)
        print(f"Spoofed Test ABX Accuracy : {acc_sp:.4f}")

    
    #Test results on classification model
    # tester.test_classification(test_dataloader)
if __name__ == '__main__':
    args = config.get_args()
    main(args)