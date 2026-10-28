"""Loading latent transformers and editing W+ codes with them."""
import glob
import math
import os

import torch

from .attributes import attribute_index, canonical_name
from .models import LatentTransformer
from .io import torch_load


def build_transformer(cfg):
    return LatentTransformer(n_styles=cfg.get('n_styles', 18), style_dim=cfg.get('style_dim', 512),
                             depth=cfg.get('depth', 3), hidden_dim=cfg.get('hidden_dim', 512))


def _convert_legacy_state_dict(state_dict, style_dim=512):
    """Weights of the original linear latent transformer (``F_mapping``: ``dense.{i}.weight/bias``).

    Those layers apply equalised learning rate at runtime (weight * 1/sqrt(fan_in)), which we fold
    into the weights of an equivalent ``depth=1`` :class:`LatentTransformer`.
    """
    scale = 1 / math.sqrt(style_dim)
    converted = {}
    for key, value in state_dict.items():
        _, idx, name = key.split('.')
        converted[f'layers.{idx}.net.0.{name}'] = value * scale if name == 'weight' else value
    n_styles = len([k for k in state_dict if k.endswith('.weight')])
    return converted, {'n_styles': n_styles, 'style_dim': style_dim, 'depth': 1}


def load_transformer(path, map_location='cpu'):
    ckpt = torch_load(path, map_location=map_location)
    if 'state_dict' in ckpt:  # checkpoint written by scripts/train_transformer.py
        state_dict, cfg = ckpt['state_dict'], ckpt.get('model', {})
    else:
        state_dict = ckpt.get('T_net_state_dict', ckpt)
        if any(k.startswith('dense.') for k in state_dict):
            state_dict, cfg = _convert_legacy_state_dict(state_dict)
        else:
            cfg = {}
    net = build_transformer(cfg)
    net.load_state_dict(state_dict)
    return net.eval()


def find_transformer_checkpoint(model_dir, attr):
    """Look for ``tnet_<Attribute>.pth`` (new) or ``tnet_<index>.pth.tar`` (original code)."""
    name = canonical_name(attr)
    candidates = [os.path.join(model_dir, f'tnet_{name}.pth'),
                  os.path.join(model_dir, f'tnet_{attribute_index(name)}.pth.tar')]
    candidates += sorted(glob.glob(os.path.join(model_dir, f'*{name}*.pth*')))
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f'No latent transformer for "{name}" in {model_dir}')


class LatentEditor:
    """Applies one or several attribute edits to W+ codes.

    Only the first ``edit_layers`` style layers are modified; the remaining (fine) layers are copied
    from the input code so that colour / texture details are preserved (as in the original code).
    """

    def __init__(self, model_dir, device='cpu', edit_layers=11):
        self.model_dir = model_dir
        self.device = device
        self.edit_layers = edit_layers
        self._cache = {}

    def transformer(self, attr):
        name = canonical_name(attr)
        if name not in self._cache:
            path = find_transformer_checkpoint(self.model_dir, name)
            self._cache[name] = load_transformer(path).to(self.device)
        return self._cache[name]

    @torch.no_grad()
    def edit(self, w, attr, alpha):
        w_edit = self.transformer(attr)(w, alpha)
        if self.edit_layers is not None:
            w_edit = torch.cat([w_edit[:, :self.edit_layers], w[:, self.edit_layers:]], dim=1)
        return w_edit

    @torch.no_grad()
    def sequential_edit(self, w, edits):
        """``edits``: iterable of ``(attribute, alpha)`` applied one after another."""
        w_edit = w
        for attr, alpha in edits:
            w_edit = self.transformer(attr)(w_edit, alpha)
            if self.edit_layers is not None:
                w_edit = torch.cat([w_edit[:, :self.edit_layers], w[:, self.edit_layers:]], dim=1)
        return w_edit


def parse_edits(spec):
    """``"Smiling=1.0,Eyeglasses=1.5"`` -> ``[("Smiling", 1.0), ("Eyeglasses", 1.5)]``."""
    edits = []
    for item in spec.split(','):
        item = item.strip()
        if not item:
            continue
        name, value = item.split('=')
        edits.append((name.strip(), float(value)))
    return edits
