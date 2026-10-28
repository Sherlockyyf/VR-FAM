"""IR-SE-50 ArcFace backbone used by the identity loss and the ID metric.

Modified Backbone implementation from TreB1eN/InsightFace_Pytorch (MIT License).
"""
from torch.nn import BatchNorm1d, BatchNorm2d, Conv2d, Dropout, Linear, Module, PReLU, Sequential

from .helpers import Flatten, bottleneck_IR, bottleneck_IR_SE, get_blocks, l2_norm


class Backbone(Module):
    def __init__(self, input_size, num_layers, mode='ir', drop_ratio=0.4, affine=True):
        super().__init__()
        assert input_size in [112, 224], 'input_size should be 112 or 224'
        assert num_layers in [50, 100, 152], 'num_layers should be 50, 100 or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        unit_module = bottleneck_IR if mode == 'ir' else bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False), BatchNorm2d(64), PReLU(64))
        spatial = 7 if input_size == 112 else 14
        self.output_layer = Sequential(BatchNorm2d(512), Dropout(drop_ratio), Flatten(),
                                       Linear(512 * spatial * spatial, 512), BatchNorm1d(512, affine=affine))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel, bottleneck.depth, bottleneck.stride))
        self.body = Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return l2_norm(x)
