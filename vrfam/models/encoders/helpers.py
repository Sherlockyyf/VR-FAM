"""IR-SE ResNet building blocks.

ArcFace implementation from TreB1eN/InsightFace_Pytorch (MIT License), as used by
pixel2style2pixel and encoder4editing (MIT License).
"""
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch.nn import AdaptiveAvgPool2d, BatchNorm2d, Conv2d, MaxPool2d, Module, PReLU, ReLU, Sequential, Sigmoid


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    return torch.div(input, norm)


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    """A named tuple describing a ResNet block."""


def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for _ in range(num_units - 1)]


def get_blocks(num_layers):
    if num_layers == 50:
        units = (3, 4, 14, 3)
    elif num_layers == 100:
        units = (3, 13, 30, 3)
    elif num_layers == 152:
        units = (3, 8, 36, 3)
    else:
        raise ValueError(f'Invalid number of layers: {num_layers}. Must be one of [50, 100, 152]')
    return [
        get_block(in_channel=64, depth=64, num_units=units[0]),
        get_block(in_channel=64, depth=128, num_units=units[1]),
        get_block(in_channel=128, depth=256, num_units=units[2]),
        get_block(in_channel=256, depth=512, num_units=units[3]),
    ]


class SEModule(Module):
    def __init__(self, channels, reduction):
        super().__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return module_input * x


class bottleneck_IR(Module):
    def __init__(self, in_channel, depth, stride):
        super().__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(Conv2d(in_channel, depth, (1, 1), stride, bias=False), BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False), PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False), BatchNorm2d(depth))

    def forward(self, x):
        return self.res_layer(x) + self.shortcut_layer(x)


class bottleneck_IR_SE(Module):
    def __init__(self, in_channel, depth, stride):
        super().__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(Conv2d(in_channel, depth, (1, 1), stride, bias=False), BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth),
            SEModule(depth, 16))

    def forward(self, x):
        return self.res_layer(x) + self.shortcut_layer(x)


def _upsample_add(x, y):
    """Upsample ``x`` to the spatial size of ``y`` and add them (FPN lateral connection)."""
    _, _, H, W = y.size()
    return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y
