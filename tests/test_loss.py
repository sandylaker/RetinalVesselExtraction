import torch
from utils import *


def test_dice_loss():
    target_true = torch.ones(200, 1024, 1024)
    logits = torch.ones(200, 1024, 1024) * 1000

    criterion = SoftDiceLoss()
    loss = criterion(logits, target_true)
    print(loss.item())


def test_BCEWithLogitsLoss2d():
    logits = torch.zeros(10, 1, 584, 565)
    targets = torch.ones(10, 1, 584, 565)

    # weight_map = torch.rand_like(logits)
    weight_map = torch.ones_like(logits)
    # weight_map = weight_map/torch.sum(weight_map, dim=(2, 3), keepdim=True)
    # weight_map = None

    criterion = BCEWithLogitsLoss2d(weight=weight_map)
    loss = criterion(logits, targets)
    print(loss.item())