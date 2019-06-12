import numpy as np
import os
import skimage
from skimage.feature import canny
from skimage import io
import torch
from torch import Tensor
import torchvision.transforms as transforms
from warnings import filterwarnings
filterwarnings('ignore')


def canny_weight_map(targets: Tensor,
                     factor,
                     sigma=1,
                     low_threshold=0.2,
                     high_threshold=0.8) -> Tensor:
    if targets.is_cuda:
        targets = targets.cpu()
    assert len(targets.size()) == 4

    targets = targets.numpy()
    N, C, H, W = targets.shape
    for i in range(N):
        for j in range(C):
            edge = canny(targets[i, j, :, :], sigma=sigma, low_threshold=low_threshold,
                         high_threshold=high_threshold)
            edge = edge * factor
            edge[edge == 0] = 1
            targets[i, j, :, :] = edge
    return torch.tensor(targets)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    a = torch.tensor(
        np.array(
           [[[[0, 0, 0, 0, 0],
             [0, 1, 1, 1, 0],
             [0, 1, 0, 1, 0],
             [0, 1, 1, 1, 0],
             [0, 0, 0, 0, 0]]],
            [[[0, 0, 0, 0, 1],
              [0, 0, 0, 1, 0],
              [0, 0, 1, 0, 0],
              [0, 1, 0, 0, 0],
              [1, 0, 0, 0, 0]]]]
    ))
    a_canny = canny_weight_map(a, 3)
    b = io.imread(os.path.join('../data/training/1st_manual', '{}_manual1.gif'.format(22)))
    b = torch.tensor(b[np.newaxis, np.newaxis, :, :])
    b_canny = canny_weight_map(b, 3).numpy()

    plt.imshow(b_canny[0, 0], cmap='RdBu')
    plt.colorbar()
    plt.show()




