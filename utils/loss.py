import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BCEWithLogitsLoss2d(nn.Module):

    def __init__(self, reduction='mean', **kwargs):
        """
        this class computes the binary cross entropy loss for a 2d image

        :param reduction: see docs of nn.BCEWithLogitsLoss on pytorch website
        """
        super(BCEWithLogitsLoss2d, self).__init__()
        self.loss = nn.BCEWithLogitsLoss(reduction=reduction, **kwargs)

    def forward(self, logits, targets):
        """
        forward function

        :param logits: (Tensor) - the logits of each pixel
        :param targets: (Tensor) - the ground truth label of each pixel
        :return: scalar. If reduction is 'none', then (N,*), same as logits input.
        """
        logits = logits.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        return self.loss(logits, targets)


class SoftDiceLoss(nn.Module):

    def __init__(self, smooth_factor=1):
        super(SoftDiceLoss, self).__init__()
        self.smooth_factor = smooth_factor

    def forward(self, logits, targets):
        proba = torch.sigmoid(logits)
        num = targets.size(0)

        proba = proba.view(num, -1)
        targets = targets.view(num, -1)

        intersection = proba * targets
        # use 1 as smooth factor for back propagation
        score = 2. * (intersection.sum(1) + self.smooth_factor) / (
            proba.sum(1) + targets.sum(1) + self.smooth_factor)
        # average over batch size and compute the loss
        score = 1 - score.sum() / num
        return score


class CombinedLoss(nn.Module):
    def __init__(self, smooth_factor=1, reduction='mean'):
        super(CombinedLoss, self).__init__()
        self.dice_loss = SoftDiceLoss(smooth_factor=smooth_factor)
        self.bce_loss = BCEWithLogitsLoss2d(reduction=reduction)

    def forward(self, logits, targets):
        return self.bce_loss(logits, targets) + self.dice_loss(logits, targets)