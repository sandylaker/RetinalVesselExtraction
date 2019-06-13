import torch
from src.unet import UNet
from src import UNetPlusPlus
import os
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

def test_unet_forward():
    a = torch.ones(2, 3, 584, 565)

    unet = UNet(in_channels=3, n_classes=2, padding=(117, 118, 108, 108))
    a_out = unet(a)
    print(a_out.size())

def test_unet_2():
    """
    test additional output layers
    """
    a = torch.ones(2, 3, 584, 565)
    unet = UNet(in_channels=3, n_classes=2, padding=(117, 118, 108, 108), add_out_layers=True)
    a_out = unet(a)
    for i, out in enumerate(a_out):
        print('1/{} resolution output size:'.format(2 ** i), a_out[i].size())

def test_unet_plusplus():
    """
    test UNetPlusPlus
    """
    a = torch.ones(2, 3, 584, 565)
    model = UNetPlusPlus(3, 2, padding=(117, 118, 108, 108))
    a_out = model(a, train_mode=False, prune_level=3)
    print(a_out.size())

def test_unet_plusplus_2():
    img = Image.open(os.path.join('../data/training/images/21_training.tif'))
    to_tensor = transforms.ToTensor()
    to_PIL = transforms.ToPILImage()
    img = to_tensor(img)[None, ...]
    model = UNetPlusPlus(3, 2, padding=(117, 118, 108, 108))
    img_out = model(img, train_mode=False, prune_level=4)
    # output is a 4d tensor
    img_out = to_PIL(img_out[0])
    plt.imshow(img_out)
    plt.show()
