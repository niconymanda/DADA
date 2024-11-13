import torch
import torch.nn as nn
import config

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
    
    def __init__(self, margin=0.1, distance_function='l2', reduction='mean'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction
        
        self.distance_function = config.get_distance_function(distance_function)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        
        distance_positive = self.distance_function(anchor, positive) 
        distance_negative = self.distance_function(anchor, negative) 
        # print(f"distance_positive: {distance_positive}")
        # print(f"distance_negative: {distance_negative}")
        loss = torch.clamp_min(distance_positive - distance_negative + self.margin, 0)
        if self.reduction == "sum":
            return torch.sum(loss)
        elif self.reduction == "mean":
            return torch.mean(loss)
        else:  # reduction == "none"
            return loss
    
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

    def __init__(self, margin=0.5, distance_function='l2'):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.distance_function = config.get_distance_function(distance_function)
        
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        distance = self.distance_function(anchor, positive) 
        loss = torch.where(target == 1, distance.pow(2), torch.clamp(self.margin - distance, min=0.0).pow(2))

        return loss.mean()
    
class SquaredCosineSimilarityLoss(nn.Module):
    """
    Squared Cosine Similarity Loss function.
    Loss_cos_cos2 (A, B) = ((1 − cos(YA, YB))/2 if same
                            cos^2(YA, YB) if different
        forward(anchor: torch.Tensor, positive: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            Computes the squared cosine similarity loss between the anchor and positive tensors.
            Args:
                anchor (torch.Tensor): The anchor tensor.
                positive (torch.Tensor): The positive tensor.
                negative (torch.Tensor): The negative tensor.
            Returns:
                torch.Tensor: The computed squared cosine similarity loss.
    """

    def __init__(self, reduction='mean'):
        super(SquaredCosineSimilarityLoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        positive_similarity = torch.nn.functional.cosine_similarity(anchor, positive)
        negative_similarity = torch.nn.functional.cosine_similarity(anchor, negative)

        loss_same = (1 - positive_similarity)/2
        loss_diff = negative_similarity.pow(2)
        
        # Radomly select the target for the loss if same or different
        target = torch.randint(0, 2, (anchor.size(0),), dtype=torch.float32, device=anchor.device)
        loss = torch.where(target == 1, loss_same, loss_diff)
       
        if self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "mean":
            return loss.mean()
        else:  # reduction == "none"
            return loss 
        
