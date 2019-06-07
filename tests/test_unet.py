import torch
from src.unet import UNet

def test_unet_forward():
    a = torch.ones(2, 3, 584, 565)

    unet = UNet(in_channels=3, n_classes=2, padding=(117, 118, 108, 108))
    a_out = unet(a)
    print(a_out.size())