import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity


class ABXAccuracy(nn.Module):
    def __init__(self, distance_function="cosine"):
        super(ABXAccuracy, self).__init__()
        self.distance = distance_function

        assert self.distance in [
            "cosine",
            "euclidean",
        ], "Distance function not supported.\nSupported distance functions: cosine, euclidean"

        if self.distance == "cosine":
            self.distance_func = lambda x, y: 1 - cosine_similarity(x, y, dim=1)
        elif self.distance == "euclidean":
            self.distance_func = lambda x, y: torch.norm(x - y, p=2, dim=1)

    def forward(self, anchor, positive, negative):
        """
        Computes the ABX accuracy.

        Args:
            anchor (Tensor): Anchor embeddings of shape (batch_size, embedding_dim).
            positive (Tensor): Positive embeddings of shape (batch_size, embedding_dim).
            negative (Tensor): Negative embeddings of shape (batch_size, embedding_dim).

        Returns:
            float: ABX accuracy.
        """
        pos_dist = self.distance_func(anchor, positive).squeeze()
        neg_dist = self.distance_func(anchor, negative).squeeze()

        correct = (pos_dist < neg_dist).float()
        accuracy = correct.mean().item()

        return accuracy
