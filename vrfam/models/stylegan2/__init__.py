from .model import EqualLinear, Generator
from ...io import torch_load


def load_generator(path, size=None):
    """Load a StyleGAN2 generator from an official checkpoint (``g_ema``) or from the ``decoder``
    of a pSp / e4e / VR-FAM / HyperStyle checkpoint. The resolution is inferred when ``size`` is None."""
    ckpt = torch_load(path, map_location='cpu')
    if 'g_ema' in ckpt:
        state = ckpt['g_ema']
    else:
        state = ckpt.get('state_dict', ckpt)
        state = {k[len('decoder.'):]: v for k, v in state.items() if k.startswith('decoder.')}
    if size is None:
        n_to_rgbs = len({k.split('.')[1] for k in state if k.startswith('to_rgbs.')})
        size = 2 ** (n_to_rgbs + 2)
    generator = Generator(size, 512, 8)
    generator.load_state_dict(state)
    return generator.eval()


__all__ = ['Generator', 'EqualLinear', 'load_generator']
