import torch
import torch.nn as nn
import torch.nn.functional as F
from src.components import *


class UNet(nn.Module):
    def __init__(self, in_channels, n_classes=2, return_logits=True, padding=None, pad_value=0):
        """

        :param in_channels: (int) - Number of input channels
        :param n_classes: (int) - Number of classes to be predicted,
                                if n_class > 2, the number of output maps equals n_classes,
                                if n_class = 2, there will be only one output map
        :param return_logits: (boolean, optional) - whether to return logits. If false, return the
                                predicted probabilities computed by the softmax function
        :param padding: (int, tuple, optional) - The size of padding. (padding_left,
                            padding_right, padding_top, padding_bottom)
        :param pad_value: (int, float, optional) - the constant value to be padded
        """
        super(UNet, self).__init__()
        if n_classes <= 1:
            raise ValueError('Number of classes must be at least 2')
        self.n_classes = n_classes
        if padding:
            self.pad = nn.ConstantPad2d(padding, pad_value)
        else:
            self.pad = None

        self.conv_in = InputConv(in_channels, 64)

        self.down1 = DownLayer(64, 128)
        self.down2 = DownLayer(128, 256)
        self.down3 = DownLayer(256, 512)
        self.down4 = DownLayer(512, 1024)

        self.up1 = UpLayer(1024, 512)
        self.up2 = UpLayer(512, 256)
        self.up3 = UpLayer(256, 128)
        self.up4 = UpLayer(128, 64)

        if self.n_classes > 2:
            self.conv_out = OutLayer(64, self.n_classes)
        else:
            self.conv_out = OutLayer(64, n_classes-1)

        self.return_logits = return_logits

    def forward(self, x):
        H, W = x.size()[-2:]
        if self.pad:
            x = self.pad(x)
            left_pad_len, top_pad_len = self.pad.padding[0], self.pad.padding[2]
        x1 = self.conv_in(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        x = self.conv_out(x)

        if self.pad:
            x = x[..., top_pad_len: top_pad_len + H, left_pad_len: left_pad_len + W]

        if self.return_logits:
            return x
        else:
            return F.softmax(x, dim=1)