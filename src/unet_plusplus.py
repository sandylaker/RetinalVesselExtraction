import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from src.components import *


class UNetPlusPlus(nn.Module):
    def __init__(self,
                 in_channels,
                 n_classes=2,
                 return_logits=True,
                 padding=None,
                 pad_value=0):
        super(UNetPlusPlus, self).__init__()
        if n_classes <= 1:
            raise ValueError('Number of classes must be at least 2')
        self.n_classes = n_classes
        if padding:
            self.pad = nn.ConstantPad2d(padding, pad_value)
        else:
            self.pad = None
        self.return_logits = return_logits

        self.conv0_0 = TwoLayerConv(in_channels, 32)
        self.down1_0 = DownLayer(32, 64)
        self.down2_0 = DownLayer(64, 128)
        self.down3_0 = DownLayer(128, 256)
        self.down4_0 = DownLayer(256, 512)

        self.up0_1 = UpLayerPlusPlus(64, 32, 2)
        self.up1_1 = UpLayerPlusPlus(128, 64, 2)
        self.up2_1 = UpLayerPlusPlus(256, 128, 2)
        self.up3_1 = UpLayerPlusPlus(512, 256, 2)
        self.up0_2 = UpLayerPlusPlus(64, 32, 3)
        self.up1_2 = UpLayerPlusPlus(128, 64, 3)
        self.up2_2 = UpLayerPlusPlus(256, 128, 3)
        self.up0_3 = UpLayerPlusPlus(64, 32, 4)
        self.up1_3 = UpLayerPlusPlus(128, 64, 4)
        self.up0_4 = UpLayerPlusPlus(64, 32, 5)

        if self.n_classes > 2:
            out_channels = self.n_classes
        else:
            # if binary classification, output only one classification map
            out_channels = self.n_classes - 1
        self.conv_out0_4 = OutLayer(32, out_channels)
        self.conv_out0_3 = OutLayer(32, out_channels)
        self.conv_out0_2 = OutLayer(32, out_channels)
        self.conv_out0_1 = OutLayer(32, out_channels)

    def forward(self, x: Tensor, train_mode=True, prune_level=4):
        """
        forward propagation
        :param x: input tensor
        :param train_mode: (bool, Optional) - if True, return the list [output0_1, output0_2,
        output0_3, output0_4]
        :param prune_level: (int, Optional) - 4: return output0_4
                                              3: return output0_3
                                              2: return output0_2
                                              1: return output0_1
                                              the prune level is effective only when train_mode
                                              is False
        :return:
        """
        assert (prune_level > 0) & (prune_level <= 4)
        H, W = x.size()[-2:]
        if self.pad:
            x = self.pad(x)
            left_pad_len, top_pad_len = self.pad.padding[0], self.pad.padding[2]
        else:
            top_pad_len = None
            left_pad_len = None
        x0_0 = self.conv0_0(x)

        x1_0 = self.down1_0(x0_0)
        x0_1 = self.up0_1(x1_0, (x0_0,))

        x2_0 = self.down2_0(x1_0)
        x1_1 = self.up1_1(x2_0, (x1_0,))
        x0_2 = self.up0_2(x1_1, (x0_0, x0_1))

        x3_0 = self.down3_0(x2_0)
        x2_1 = self.up2_1(x3_0, (x2_0,))
        x1_2 = self.up1_2(x2_1, (x1_0, x1_1))
        x0_3 = self.up0_3(x1_2, (x0_0, x0_1, x0_2))

        x4_0 = self.down4_0(x3_0)
        x3_1 = self.up3_1(x4_0, (x3_0,))
        x2_2 = self.up2_2(x3_1, (x2_0, x2_1))
        x1_3 = self.up1_3(x2_2, (x1_0, x1_1, x1_2))
        x0_4 = self.up0_4(x1_3, (x0_0, x0_1, x0_2, x0_3))

        x_out0_4 = self.conv_out0_4(x0_4)
        x_out0_3 = self.conv_out0_3(x0_3)
        x_out0_2 = self.conv_out0_2(x0_2)
        x_out0_1 = self.conv_out0_1(x0_1)

        if self.pad:
            x_out0_4 = self._crop_output(x_out0_4, top_pad_len, left_pad_len, H, W)
            x_out0_3 = self._crop_output(x_out0_3, top_pad_len, left_pad_len, H, W)
            x_out0_2 = self._crop_output(x_out0_2, top_pad_len, left_pad_len, H, W)
            x_out0_1 = self._crop_output(x_out0_1, top_pad_len, left_pad_len, H, W)

        if self.return_logits:
            output_list = [x_out0_1, x_out0_2, x_out0_3, x_out0_4]
            if train_mode:
                return output_list
            else:
                return output_list[prune_level-1]
        else:
            output_list = [F.softmax(x_out0_1),
                           F.softmax(x_out0_2),
                           F.softmax(x_out0_3),
                           F.softmax(x_out0_4)]
            if train_mode:
                return output_list
            else:
                return output_list[prune_level-1]

    @staticmethod
    def _crop_output(x, top_pad_len, left_pad_len, H, W):
        x = x[..., top_pad_len: top_pad_len + H, left_pad_len: left_pad_len + W]
        return x


