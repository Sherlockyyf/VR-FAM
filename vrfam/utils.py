"""Small helpers shared by the scripts."""
import argparse
import os
import random

import numpy as np
import torch
import yaml
from PIL import Image

from .io import torch_load


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device=None):
    if device:
        return torch.device(device)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_config(path, overrides=None):
    """Load a YAML config and apply ``key=value`` overrides (dotted keys for nested entries)."""
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    for item in overrides or []:
        key, value = item.split('=', 1)
        node = cfg
        *parents, leaf = key.split('.')
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(value)
    return cfg


def add_config_args(parser: argparse.ArgumentParser, default_config):
    parser.add_argument('--config', default=default_config, help='YAML config file')
    parser.add_argument('--opts', nargs='*', default=[], metavar='KEY=VALUE',
                        help='override config entries, e.g. --opts batch_size=16 lambda.attr=5')


def tensor2im(x):
    """[-1, 1] CHW tensor -> uint8 PIL image."""
    x = (x.detach().float().cpu().clamp(-1, 1) + 1) / 2
    return Image.fromarray((x.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))


def make_row(images, size=256):
    """Concatenate a list of [-1, 1] CHW tensors (or PIL images) horizontally."""
    pils = [im if isinstance(im, Image.Image) else tensor2im(im) for im in images]
    pils = [im.resize((size, size), Image.BICUBIC) for im in pils]
    row = Image.new('RGB', (size * len(pils), size))
    for i, im in enumerate(pils):
        row.paste(im, (i * size, 0))
    return row


def save_latent(w, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torch.save(w.detach().cpu(), path)


def load_latent(path, device='cpu'):
    w = np.load(path) if path.endswith('.npy') else torch_load(path, map_location='cpu')
    w = torch.as_tensor(np.asarray(w) if not torch.is_tensor(w) else w, dtype=torch.float32)
    return w.reshape(1, -1, 512).to(device)


def count_parameters(module):
    return sum(p.numel() for p in module.parameters())
