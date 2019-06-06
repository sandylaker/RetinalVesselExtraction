import torch

from utils import DiceScoreWithLogits

def test_DiceScoreWithLogits():
    size= (10, 1, 100, 100)
    logits_list = [torch.ones(size=size),
                   torch.zeros(size),
                   torch.rand(size),
                   ]
    targets = torch.ones(size)
    score = DiceScoreWithLogits()
    for i, logits in enumerate(logits_list):
        print('{}: score: {}'.format(i, score(logits, targets)))