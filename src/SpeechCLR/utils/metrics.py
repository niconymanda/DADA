import torch
import torch.nn as nn


class ABXAccuracy(nn.Module):
    def __init__(self):
        super(ABXAccuracy, self).__init__()

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
        pos_dist = torch.norm(anchor - positive, dim=1)
        neg_dist = torch.norm(anchor - negative, dim=1)

        correct = (pos_dist < neg_dist).float()
        accuracy = correct.mean().item()

        return accuracy
