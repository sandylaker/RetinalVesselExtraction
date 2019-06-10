import torch
from src.unet import UNet

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
