import torch
import torch.nn as nn
import config
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity

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
        positive_similarity = F.cosine_similarity(anchor, positive)
        negative_similarity = F.cosine_similarity(anchor, negative)

        loss_same = (1 - positive_similarity)/2
        loss_diff = negative_similarity.pow(2)
        
        # Randomly select the target for the loss if same or different
        # target = torch.randint(0, 2, (anchor.size(0),), dtype=torch.float32, device=anchor.device)
        # loss = torch.where(target == 1, loss_same, loss_diff)
        loss = loss_same + loss_diff
       
        if self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "mean":
            return loss.mean()
        else:  # reduction == "none"
            return loss 
        
class AdaTriplet(nn.Module):
    """
    Adaptive Triplet Loss from
    Nyugen et al. 'AdaTriplet: Adaptive Gradient Triplet Loss with Automatic Margin Learning
                   for Forensic Medical Image Matching'
    """

    def __init__(self, K_d=2, K_an=2, eps=0, beta=0, lambda_=1, reduction="mean"):
        super(AdaTriplet, self).__init__()
        self.K_d = K_d
        self.K_an = K_an
        self.eps = eps
        self.beta = beta
        self.lambda_ = lambda_

        # stats, init?
        self.mu_d = 0
        self.mu_an = 0
        self.counter = 0

        self.reduction = reduction

    def reset(self):
        self.mu_d = 0
        self.mu_an = 0
        self.counter = 0

    def update_stats(self, phi_ap, phi_an):
        delta = phi_ap - phi_an
        self.mu_d = (self.counter * self.mu_d + delta.mean()) / (self.counter + 1)
        self.mu_an = (self.counter * self.mu_an + phi_an.mean()) / (self.counter + 1)
        self.counter = self.counter + 1

    def update_margins(self):
        self.eps = self.mu_d / self.K_d
        self.beta = self.mu_an / self.K_an
        print(f"Updated eps: {self.eps}, beta: {self.beta}")

    def __repr__(self):
        return f"AdaTriplet(K_d={self.K_d}, K_an={self.K_an}, eps={self.eps}, beta={self.beta}, lambda_={self.lambda_})"

    def forward(self, anchor, positive, negative, eval=False, **kwargs):
        phi_ap = cosine_similarity(anchor, positive)
        phi_an = cosine_similarity(anchor, negative)

        if not eval:
            with torch.no_grad():
                self.update_stats(phi_ap, phi_an)
                self.update_margins()

        loss = torch.clamp_min(phi_an - phi_ap + self.eps, 0)
        loss = loss + self.lambda_ * torch.clamp_min(phi_an - self.beta, 0)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()

        return loss
    
class TripletLossTemperature(nn.Module):
    def __init__(self, temperature=0.01):
        """
        Contrastive loss implementation based on equation 3 in the paper: Louis, A., van Dijck, G., &
        Spanakis, G. (2024). Interpretable Long-Form Legal Question Answering
        with Retrieval-Augmented Large Language Models. Proceedings of the AAAI
        Conference on Artificial Intelligence, 38(20), 22266-22275..
        
        Args:
            temperature (float): Temperature parameter (tau in the equation).
        """
        super(TripletLossTemperature, self).__init__()
        self.temperature = temperature

    def forward(self, anchor, positive, negative):
        """
        Forward pass for the contrastive loss.
        
        Args:
            query (torch.Tensor): Query embeddings, shape (batch_size, embedding_dim).
            positive (torch.Tensor): Positive embeddings, shape (batch_size, embedding_dim).
            negative (torch.Tensor): Negative embeddings, shape (batch_size, embedding_dim).
            
        Returns:
            torch.Tensor: Scalar contrastive loss.
        """
        #TODO: See if we can have multiple negatives as in the original paper
        pos_scores = cosine_similarity(anchor, positive)  
        neg_scores = cosine_similarity(anchor, negative)

        pos_scores /= self.temperature
        neg_scores /= self.temperature

        exp_pos_scores = torch.exp(pos_scores)  
        exp_neg_scores = torch.exp(neg_scores)  

        loss = -torch.log(exp_pos_scores / exp_neg_scores).mean() 

        return loss
