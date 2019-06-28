import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DiceScoreWithLogits(nn.Module):

    def __init__(self, threshold=0.5, smooth_factor=0.001):
        super(DiceScoreWithLogits, self).__init__()
        self.threshold = threshold
        self.smooth_factor = smooth_factor

    def forward(self, logits, targets):
        proba = torch.sigmoid(logits)
        num = targets.size(0)
        predict = (proba.view(num, -1) > self.threshold).float()
        targets = targets.view(num, -1)
        intersection = predict * targets
        score = (2.0 * intersection.sum(1) + self.smooth_factor) / (
                predict.sum(1) + targets.sum(1) + self.smooth_factor)
        return score.mean()