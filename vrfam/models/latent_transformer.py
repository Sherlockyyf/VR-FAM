"""Nonlinear latent transformation network T (Eq. 4 of the VR-FAM paper).

    T(w+, alpha) = w+ + alpha * f(w+)

``f`` is applied independently to every style layer of the W+ code. Each per-layer branch
is a small MLP; with ``depth=3`` (default, as in the paper) the transformation is nonlinear,
while ``depth=1`` reduces to the linear per-layer latent transformer of Yao et al. (ICCV 2021).
One network is trained per attribute.
"""
import torch
from torch import nn


class LayerwiseMLP(nn.Module):
    def __init__(self, dim=512, depth=3, hidden_dim=512, negative_slope=0.2):
        super().__init__()
        layers = []
        in_dim = dim
        for i in range(depth):
            out_dim = dim if i == depth - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < depth - 1:
                layers.append(nn.LeakyReLU(negative_slope))
            in_dim = out_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class LatentTransformer(nn.Module):
    def __init__(self, n_styles=18, style_dim=512, depth=3, hidden_dim=512):
        super().__init__()
        self.n_styles = n_styles
        self.style_dim = style_dim
        self.layers = nn.ModuleList(LayerwiseMLP(style_dim, depth, hidden_dim) for _ in range(n_styles))

    def direction(self, w):
        """f(w+): the (input dependent) editing direction, shape [B, n_styles, style_dim]."""
        w = w.view(w.size(0), self.n_styles, self.style_dim)
        return torch.stack([layer(w[:, i]) for i, layer in enumerate(self.layers)], dim=1)

    def forward(self, w, alpha):
        """Edit ``w`` ([B, n_styles, 512]) with strength ``alpha`` (scalar or [B])."""
        w = w.view(w.size(0), self.n_styles, self.style_dim)
        alpha = torch.as_tensor(alpha, dtype=w.dtype, device=w.device).reshape(-1, 1, 1)
        return w + alpha * self.direction(w)
