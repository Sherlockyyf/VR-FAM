"""Latent attribute classifier C: W+ -> {0, 1}^N (Sec. 2.2 of the VR-FAM paper).

A three-layer MLP on the flattened W+ code (18 x 512 = 9216 dims) that predicts the logits
of the 40 CelebA attributes. Parameter names (``fcs.*``) follow the latent classifier
released with "A Latent Transformer for Disentangled Face Editing" (Yao et al., ICCV 2021),
so that their pretrained classifier can be reused.
"""
import torch
from torch import nn

from ..io import torch_load


class LatentClassifier(nn.Module):
    def __init__(self, fmaps=(9216, 2048, 512, 40), activation='leakyrelu'):
        super().__init__()
        self.fcs = nn.ModuleList(nn.Linear(fmaps[i], fmaps[i + 1]) for i in range(len(fmaps) - 1))
        if activation == 'relu':
            self.activ = nn.ReLU()
        elif activation == 'leakyrelu':
            self.activ = nn.LeakyReLU(0.2)
        else:
            raise ValueError(f'Unsupported activation: {activation}')

    def forward(self, w):
        """``w``: [B, 18, 512] or [B, 9216] -> attribute logits [B, N]."""
        x = w.flatten(1)
        for layer in self.fcs[:-1]:
            x = self.activ(layer(x))
        return self.fcs[-1](x)


def load_classifier(path, activation='leakyrelu', map_location='cpu'):
    """Build a :class:`LatentClassifier` whose layer sizes are read from the checkpoint itself."""
    state_dict = torch_load(path, map_location=map_location)
    weights = [state_dict[f'fcs.{i}.weight'] for i in range(len(state_dict) // 2)]
    fmaps = [weights[0].shape[1]] + [w.shape[0] for w in weights]
    model = LatentClassifier(fmaps, activation)
    model.load_state_dict(state_dict)
    return model.eval()
