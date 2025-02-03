import torch
import torch.nn as nn

class SelfContrastiveLoss(nn.Module):
    def __init__(self, lambda_=0.1):
        super(SelfContrastiveLoss, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x_style, x_linguistics):
        # Define the custom loss computation
        # Normalize embeddings
        B = x_style.size(0)
        x_style_norm = nn.BatchNorm1d(x_style, affine=False) / B
        x_linguistics_norm = nn.BatchNorm1d(x_linguistics, affine=False) / B

        # Compute cross-subspace loss
        D = torch.linalg.norm(x_style_norm - x_linguistics_norm, ord="fro")
        D = torch.pow(D, 2)

        # Compute intra-subspace redundancy loss
        v_linguistic = torch.mm(x_linguistics_norm.T, x_linguistics_norm)
        C_linguistic = torch.linalg.norm(v_linguistic - torch.eye(v_linguistic.size(1)))
        C_linguistic = torch.pow(C_linguistic, 2)

        v_style = torch.mm(x_style_norm.T, x_style_norm)
        C_style = torch.linalg.norm(v_style - torch.eye(v_style.size(1)))
        C_style = torch.pow(C_style, 2)

        # Composite Loss
        loss = D = self.lambda_ * (C_linguistic + C_style)

        return loss
