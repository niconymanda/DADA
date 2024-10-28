import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    """
    Implements the triplet loss for text embeddings.
    
    The triplet loss is defined as:
    L(A,P,N) = max(||f(A) - f(P)||₂ - ||f(A) - f(N)||₂ + α, 0)
    
    where:
    - A: Anchor embedding
    - P: Positive embedding (same class as anchor)
    - N: Negative embedding (different class from anchor)
    - α: Margin parameter
    - ||.||₂: L2 norm (Euclidean distance)
    Args:
        margin (float, optional): The margin value for the triplet loss. Default is 1.0.
    Methods:
        forward(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
            Computes the triplet loss given anchor, positive, and negative tensors.
            Args:
                anchor (torch.Tensor): The anchor tensor.
                positive (torch.Tensor): The positive tensor (same author as anchor).
                negative (torch.Tensor): The negative tensor (another random auhhor different then anchor).
            Returns:
                torch.Tensor: The computed triplet loss.
    """
    
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        # distance_positive = F.pairwise_distance(anchor, positive, p=2)
        # distance_negative = F.pairwise_distance(anchor, negative, p=2)
        # loss = torch.clamp(distance_positive - distance_negative + self.margin, min=0.0)
        # return loss.mean()
        distance_positive = torch.norm(anchor - positive, dim=1, p=2) 
        distance_negative = torch.norm(anchor - negative, dim=1, p=2) 
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return torch.mean(losses)
    
class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss function.
    Args:
        margin (float, optional): Margin for contrastive loss. Default is 0.5.
    Methods:
        forward(anchor: torch.Tensor, positive: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            Computes the contrastive loss between the anchor and positive tensors.
            Args:
                anchor (torch.Tensor): The anchor tensor.
                positive (torch.Tensor): The positive tensor.
                target (torch.Tensor): The target tensor indicating whether pairs are similar (1) or dissimilar (0).
            Returns:
                torch.Tensor: The computed contrastive loss.
    """

    def __init__(self, margin=0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        distance = F.pairwise_distance(anchor, positive, p=2)
        loss = torch.where(target == 1, distance.pow(2), torch.clamp(self.margin - distance, min=0.0).pow(2))

        return loss.mean()