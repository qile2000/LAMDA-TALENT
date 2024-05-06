import torch.nn as nn
import torch.nn.functional as F

class MarginLoss(nn.Module):
    def __init__(self):
        super(MarginLoss, self).__init__()

    def forward(self, pred, labels):
        b = labels.shape[0]
        pred = F.softmax(pred, dim=-1)
        left = F.relu(0.9 - pred, inplace=True) ** 2
        right = F.relu(pred - 0.1, inplace=True) ** 2

        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = (margin_loss.sum()) / b
        return margin_loss
