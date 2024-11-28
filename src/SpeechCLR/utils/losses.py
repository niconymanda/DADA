import torch
import torch.nn.functional as F
from torch.nn import TripletMarginLoss, TripletMarginWithDistanceLoss
import os

import torch.nn as nn
from torch.nn.functional import cosine_similarity


class SquaredSimilarity(nn.Module):
    """
    Squared Similarity Loss from
    G. Synnaeve, T. Schatz and E. Dupoux, 'Phonetics embedding learning with side information'
    """

    def __init__(self, reduction="mean"):
        super(SquaredSimilarity, self).__init__()
        self.reduction = reduction
        self.sq_cos = lambda x, y: F.cosine_similarity(x, y) ** 2

    def forward(self, x, y, x_label, y_label, **kwargs):
        batch_size = x.size(0)

        same_mask = (x_label.squeeze() == y_label.squeeze()).float()
        diff_mask = ~same_mask

        loss = torch.zeros(batch_size, device=x.device)
        loss[same_mask] = self.sq_cos(x[same_mask], y[same_mask])
        loss[diff_mask] = 1 - self.sq_cos(x[diff_mask], y[diff_mask])

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()

        return loss


class NormalisedEuclideanDistance(nn.Module):  # Doesn't work, may not need to implement
    def __init__(self):
        super(NormalisedEuclideanDistance, self).__init__()

    def forward(self, x, y):
        pass


class TripletMarginCosineLoss(nn.Module):
    def __init__(self, margin=1.0, reduction="mean"):
        super(TripletMarginCosineLoss, self).__init__()
        self.margin = margin
        self.counter = 0
        self.reduction = reduction

    def update(self):
        self.counter += 1

    def forward(self, anchor, positive, negative, **kwargs):
        pos_dist = 1 - F.cosine_similarity(anchor, positive)
        neg_dist = 1 - F.cosine_similarity(anchor, negative)

        loss = F.relu(pos_dist - neg_dist + self.margin)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
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


class CosineDistance(torch.nn.Module):
    def __init__(self):
        super(CosineDistance, self).__init__()

    def forward(self, x, y):
        return 1 - F.cosine_similarity(x, y)


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    a = torch.randn(10, 200)
    p = torch.randn(10, 200)
    n = torch.randn(10, 200)

    criterion = TripletMarginLoss(margin=1.0)
    criterion2 = TripletMarginWithDistanceLoss(
        margin=1.0, distance_function=CosineDistance()
    )

    loss = criterion(a, p, n)
    loss2 = criterion2(a, p, n)

    print(f"Loss 1: {loss}, Loss 2: {loss2}")
