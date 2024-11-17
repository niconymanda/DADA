import torch
import torch.nn.functional as F
from torch.nn import TripletMarginLoss, TripletMarginWithDistanceLoss
import os


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
