import torch
import torch.nn as nn


class TwoLayerConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TwoLayerConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class InputConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InputConv, self).__init__()
        self.conv = TwoLayerConv(in_channels, out_channels)

    def forward(self, x):
        x = self.conv(x)
        return x


class DownLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownLayer, self).__init__()
        self.max_and_conv = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            TwoLayerConv(in_channels, out_channels)
        )

    def forward(self, x):
        x = self.max_and_conv(x)
        return x


class UpLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpLayer, self).__init__()
        self.up_layer = nn.ConvTranspose2d(in_channels, in_channels//2, 2, stride=2)
        self.conv = TwoLayerConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up_layer(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x