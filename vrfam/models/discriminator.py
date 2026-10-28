"""Latent codes discriminator D_W (Eq. 2-3 of the VR-FAM paper).

Same architecture as in encoder4editing (MIT License): an MLP that tells codes produced by
StyleGAN's mapping network (real) from codes produced by the encoder (fake).
"""
from torch import nn


class LatentCodesDiscriminator(nn.Module):
    def __init__(self, style_dim=512, n_mlp=4):
        super().__init__()
        self.style_dim = style_dim
        layers = []
        for _ in range(n_mlp - 1):
            layers.append(nn.Linear(style_dim, style_dim))
            layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Linear(style_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, w):
        return self.mlp(w)
