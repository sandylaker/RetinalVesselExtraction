import torch
import torch.nn as nn
import torch.nn.functional as F
from src.components import *


class UNet(nn.Module):
    def __init__(self,
                 in_channels,
                 n_classes=2,
                 return_logits=True,
                 padding=None,
                 pad_value=0,
                 add_out_layers=False):
        """
        The U-Net Model

        :param in_channels: (int) - Number of input channels
        :param n_classes: (int) - Number of classes to be predicted,
                                if n_class > 2, the number of output maps equals n_classes,
                                if n_class = 2, there will be only one output map
        :param return_logits: (boolean, optional) - whether to return logits. If false, return the
                                predicted probabilities computed by the softmax function
        :param padding: (int, tuple, optional) - The size of padding. (padding_left,
                            padding_right, padding_top, padding_bottom)
        :param pad_value: (int, float, optional) - the constant value to be padded
        :param add_out_layers: (bool, optional) - If True, include additional output layers at low
        resolutions, and forward function will return a tuple of (out_1, out_2, out_4,
        out_8, out_16), where out_i indicating the output layer at low-resolution /i with respect to
        the highest resolution (out_1). Caution: the additional outputs comes from padded inputs.
        So when computing losses for these layers, first pad the targets and down sample them.
        If False, only keep the output layer at highest resolution.
        """
        super(UNet, self).__init__()
        if n_classes <= 1:
            raise ValueError('Number of classes must be at least 2')
        self.n_classes = n_classes
        if padding:
            self.pad = nn.ConstantPad2d(padding, pad_value)
        else:
            self.pad = None
        self.add_out_layers = add_out_layers

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
            out_channels = self.n_classes
        else:
            # if binary classification, output only one classification map
            out_channels = self.n_classes - 1
        self.conv_out = OutLayer(64, out_channels)

        if self.add_out_layers:
            # output layers at low-res /2, /4, /8 and /16
            self.conv_out_2 = nn.Conv2d(
                128, out_channels, kernel_size=3, padding=1)
            self.conv_out_4 = nn.Conv2d(
                256, out_channels, kernel_size=3, padding=1)
            self.conv_out_8 = nn.Conv2d(
                512, out_channels, kernel_size=3, padding=1)
            self.conv_out_16 = nn.Conv2d(
                1024, out_channels, kernel_size=3, padding=1)

        self.return_logits = return_logits

    def forward(self, x):
        H, W = x.size()[-2:]
        if self.pad:
            x = self.pad(x)
            left_pad_len, top_pad_len = self.pad.padding[0], self.pad.padding[2]
        else:
            top_pad_len = None
            left_pad_len = None
        x1 = self.conv_in(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # additional output at low-res layers /16
        if self.add_out_layers:
            x_16 = self.conv_out_16(x5)
        else:
            x_16 = None
        x = self.up1(x5, x4)
        # additional output at low-res layers /8
        if self.add_out_layers:
            x_8 = self.conv_out_8(x)
        else:
            x_8 = None
        x = self.up2(x, x3)
        # additional output at low-res layers /4
        if self.add_out_layers:
            x_4 = self.conv_out_4(x)
        else:
            x_4 = None
        x = self.up3(x, x2)
        # additional output at low-res layers /2
        if self.add_out_layers:
            x_2 = self.conv_out_2(x)
        else:
            x_2 = 0
        x = self.up4(x, x1)

        x = self.conv_out(x)

        if self.pad:
            x = x[..., top_pad_len: top_pad_len +
                  H, left_pad_len: left_pad_len + W]

        if self.return_logits:
            if self.add_out_layers:
                return [x, x_2, x_4, x_8, x_16]
            else:
                return x
        else:
            if self.add_out_layers:
                return [
                    F.softmax(x),
                    F.softmax(x_2),
                    F.softmax(x_4),
                    F.softmax(x_8),
                    F.softmax(x_16)]
