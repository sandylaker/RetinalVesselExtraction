import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BCEWithLogitsLoss2d(nn.Module):

    def __init__(self, reduction='mean', pos_weight=None):
        """
        this class computes the binary cross entropy loss for a 2d image

        :param reduction: see docs of nn.BCEWithLogitsLoss on pytorch website
        :param pos_weight: see docs of nn.BCEWithLogitsLoss on pytorch website
        """
        super(BCEWithLogitsLoss2d, self).__init__()
        self.loss = nn.BCEWithLogitsLoss(reduction=reduction, pos_weight=pos_weight)

    def forward(self, logits, targets, weights=None):
        """
        forward function

        :param logits: (Tensor) - the logits of each pixel
        :param targets: (Tensor) - the ground truth label of each pixel
        :return: scalar. If reduction is 'none', then (N,*), same as logits input.
        """
        # TODO add weight for each classification map
        logits = logits.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        return self.loss(logits, targets)


class SoftDiceLoss(nn.Module):

    def __init__(self):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets, weights=None):
        proba = torch.sigmoid(logits)
        num = targets.size(0)
        if weights is None:
            weights = torch.ones_like(targets)
        else:
            weights = weights
        targets = weights * targets
        proba = weights * proba
        proba = proba.view(num, -1)
        targets = targets.view(num, -1)

        intersection = proba * targets
        # use 1 as smooth factor for back propagation
        score = 2. * (intersection.sum(1) + 1) / (
                proba.sum(1) + targets.sum(1) + 1)
        # average over batch size and compute the loss
        score = 1 - score.sum() / num
        return score

