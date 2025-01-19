import numpy as np
from torch.utils.tensorboard import SummaryWriter
import wandb

class Logger:
    def __init__(self, args, report_to, project_name, model, repository_id, train_dataloader=None, val_dataloader=None):
        self.args = args
        self.report_to = report_to
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        if self.report_to == 'tensorboard':
            self.writer = SummaryWriter(f"{repository_id}/logs")
        elif self.report_to == 'wandb':
            wandb.init(project=project_name, config=vars(args))
            wandb.watch(model)
        else:
            raise ValueError("Invalid report_to value. Must be 'tensorboard' or 'wandb'")
        
    def _log_metrics(self, metrics, phase):
        if self.report_to == 'wandb':
            for key, value in metrics.items():
                wandb.log({f"{phase}/{key}": value}) if (key != 'epoch' and key != 'step') else None
        elif self.report_to == 'tensorboard':
            for key, value in metrics.items():
                if 'epoch' in phase:
                    # print(f"{phse}/{key}, epoch")
                    self.writer.add_scalar(f"{phase}/{key}", value, metrics['epoch']) if (key != 'epoch' and key != 'step') else None
                elif 'Train' in phase:                    
                    self.writer.add_scalar(f"{phase}/{key}", value, metrics['epoch'] * len(self.train_dataloader) + metrics['step']) if (key != 'epoch' and key != 'step') else None
                elif 'Val' in phase:
                    self.writer.add_scalar(f"{phase}/{key}", value, metrics['epoch'] * len(self.val_dataloader) + metrics['step']) if (key != 'epoch' and key != 'step') else None
                else:
                    print("Invalid phase")
                    
    def log_figure(self, figure, name, epoch):
        if self.report_to == 'wandb':
            wandb.log({name: figure})
        elif self.report_to == 'tensorboard':
            self.writer.add_figure(name, figure, global_step=epoch)
            
    