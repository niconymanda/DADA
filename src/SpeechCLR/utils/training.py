import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from utils.datasets import InTheWildDataset, ASVSpoof21Dataset
from torch.utils.tensorboard import SummaryWriter

from transformers import Trainer, TrainingArguments
from transformers import TrainerCallback


class StageOneTrainer():
    pass

class StageTwoTrainer():
    pass