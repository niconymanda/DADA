"""
Training Module for Context Aware Audio Spoof Detection with Mid-Fusion

Metrics:
    - EER
    - F1
    - Accuracy ?
    - t-DCF

    
Losses:
    - CrossEntropyLoss
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm
from torch import autograd
from torch.utils.data import DataLoader
from mid_fusion.utils.datasets import InTheWildDataset
from SpeechCLR.utils.logging import Logger
from SpeechCLR.models import SpeechEmbedder
from authorship_attribution.model import AuthorshipLLM

class MidFusionTrainer():
    def __init__(self, args):
        pass

    def train_epoch(self):
        pass

    def train(self):
        pass

    def evaluate(self):
        pass

    def save(self):
        pass
