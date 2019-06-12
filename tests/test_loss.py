import torch
from utils import *


def test_dice_loss_1():
    target_true = torch.tensor(np.random.uniform(0, 1, (2, 1, 10, 10)) > 0.5, dtype=torch.float)
    logits = torch.ones(2, 1, 10, 10) * 50
    weights_1 = None
    weights_2 = torch.ones_like(logits)
    weights_2[0, :, 0:8, 0:8] = 4

    criterion = SoftDiceLoss(smooth_factor=1)
    loss_1 = criterion(logits, target_true, weights_1)
    loss_2 = criterion(logits, target_true, weights_2)
    print(loss_1.item())
    print(loss_2.item())

def test_dice_loss_2():
    targets = torch.tensor(np.array([[0, 1, 1, 1, 0]]), dtype=torch.float)
    logits = torch.tensor(np.array([[-1, -1, 1, -1, 1]]) * 100, dtype=torch.float)
    weights_1 = torch.ones_like(logits)
    weights_2 = torch.tensor(np.array([[1, 2, 3, 1, 1]]), dtype=torch.float)

    criterion = SoftDiceLoss()
    loss_1 = criterion(logits, targets, weights_1)
    loss_2 = criterion(logits, targets, weights_2)
    print(loss_1.item())
    print(loss_2.item())


def test_BCEWithLogitsLoss2d():
    logits = torch.zeros(10, 1, 584, 565)
    targets = torch.ones(10, 1, 584, 565)

    criterion = BCEWithLogitsLoss2d()
    loss = criterion(logits, targets)
    print(loss.item())