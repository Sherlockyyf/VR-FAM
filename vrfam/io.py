"""Checkpoint loading that works across PyTorch versions."""
import torch


def torch_load(path, map_location='cpu'):
    """``torch.load`` for the (trusted) checkpoints used in this project.

    The official StyleGAN2 / pSp / e4e / HyperStyle checkpoints and the latent codes of the original
    release contain non-tensor objects (argparse namespaces, numpy arrays) that PyTorch >= 2.6 refuses
    to unpickle by default (``weights_only=True``). Only load files that you trust.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch < 1.13 has no ``weights_only`` argument
        return torch.load(path, map_location=map_location)
