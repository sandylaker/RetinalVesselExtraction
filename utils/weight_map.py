import numpy as np
import os
from scipy.ndimage import distance_transform_edt as edt
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

    targets = targets.numpy().astype(float)
    N, C, H, W = targets.shape
    for i in range(N):
        for j in range(C):
            edge = canny(targets[i,
                                 j,
                                 :,
                                 :],
                         sigma=sigma,
                         low_threshold=low_threshold,
                         high_threshold=high_threshold)
            edge = edge * factor
            edge[edge == 0] = 1
            targets[i, j, :, :] = edge
    return torch.tensor(targets)


def edt_weight_map(targets: Tensor, sigma=1) -> Tensor:
    if targets.is_cuda:
        targets = targets.cpu()
    assert len(targets.size()) == 4

    targets = targets.numpy().astype(float)
    N, C, H, W = targets.shape
    for i in range(N):
        for j in range(C):
            # distances of front pixels (non-zeros) to the closest zeros.
            d_front = edt(targets[i, j, :, :])
            front_ind = np.nonzero(targets[i, j, :, :])
            # reverse the front and background, now the background are
            # non-zeros
            rev_map = np.ones_like(d_front)
            rev_map[front_ind] = 0
            d_back = edt(rev_map)
            d_map = d_front + d_back
            # apply gaussian function
            targets[i, j, :, :] = np.exp(- d_map ** 2 / (2 * sigma ** 2))
    return torch.tensor(targets, dtype=torch.float)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    a = torch.tensor(
        np.array(([0, 1, 1, 1, 1],
                  [0, 0, 1, 1, 1],
                  [0, 1, 1, 1, 1],
                  [0, 1, 1, 1, 0],
                  [0, 1, 1, 0, 0]))
    )
    a_edt = edt_weight_map(a.view(1, 1, 5, 5), sigma=2)
    print(a_edt)
    b = io.imread(
        os.path.join(
            '../data/training/1st_manual',
            '{}_manual1.gif'.format(23)))
    b = torch.tensor(b[np.newaxis, np.newaxis, :, :])
    b_edt = edt_weight_map(b, 5).numpy()

    plt.imshow(b_edt[0, 0], cmap='RdBu')
    plt.colorbar()
    plt.show()
