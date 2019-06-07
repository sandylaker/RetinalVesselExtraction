import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BCEWithLogitsLoss2d(nn.Module):

    def __init__(self, weight=None, reduction='mean', pos_weight=None):
        """
        this class computes the binary cross entropy loss for a 2d image

        :param weight: (Tensor, optional) - 2d weight maps, has to be a Tensor of size n-batch.
        :param reduction: see docs of nn.BCEWithLogitsLoss on pytorch website
        :param pos_weight: see docs of nn.BCEWithLogitsLoss on pytorch website
        """
        super(BCEWithLogitsLoss2d, self).__init__()
        if weight is None:
            self.weight = None
        else:
            self.weight = weight.view(-1)
        self.loss = nn.BCEWithLogitsLoss(self.weight, reduction=reduction, pos_weight=pos_weight)

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

    def __init__(self, weight=None):
        super(SoftDiceLoss, self).__init__()
        self.weight = weight

    def forward(self, logits, targets):
        proba = torch.sigmoid(logits)
        num = targets.size(0)
        if not self.weight:
            weight = torch.ones_like(targets)
        else:
            weight = self.weight
        targets = weight * targets
        proba = weight * proba
        proba = proba.view(num, -1)
        targets = targets.view(num, -1)

        intersection = proba * targets
        # use 1 as smooth factor for back propagation
        score = 2. * ((intersection).sum(1) + 1) / (
                proba.sum(1) + targets.sum(1) + 1)
        # average over batch size and compute the loss
        score = 1 - score.sum() / num
        return score

