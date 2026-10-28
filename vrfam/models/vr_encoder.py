"""Variance-reduced encoder (Sec. 2.1 of the VR-FAM paper).

``VREncoder`` bundles the progressive e4e-style encoder with a frozen StyleGAN2
generator. The encoder first learns a single style code ``w`` and then the residual
terms ``delta_i`` one after another; together with the delta L2 regulariser and the
latent discriminator this keeps the inverted codes close to StyleGAN's W space.

Checkpoints use the same layout as pSp / e4e (``state_dict`` with ``encoder.*`` and
``decoder.*`` keys, plus ``latent_avg``), so the official e4e FFHQ encoder can also be
loaded with :meth:`VREncoder.from_checkpoint` for comparison.
"""
import math

import torch
from torch import nn

from .encoders import Encoder4Editing
from .stylegan2 import Generator
from ..io import torch_load


def _filter_keys(state_dict, prefix):
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    return {k[len(prefix) + 1:]: v for k, v in state_dict.items() if k.startswith(prefix + '.')}


class VREncoder(nn.Module):
    def __init__(self, stylegan_size=1024, encoder_layers=50):
        super().__init__()
        self.stylegan_size = stylegan_size
        self.n_styles = int(math.log(stylegan_size, 2)) * 2 - 2
        self.encoder = Encoder4Editing(encoder_layers, 'ir_se', stylegan_size)
        self.decoder = Generator(stylegan_size, 512, 8, channel_multiplier=2)
        self.face_pool = nn.AdaptiveAvgPool2d((256, 256))
        # Not part of the state dict, so that official pSp/e4e checkpoints load with strict=True.
        self.register_buffer('latent_avg', torch.zeros(self.n_styles, 512), persistent=False)

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_pretrained_backbones(cls, stylegan_ckpt, irse50_ckpt=None, stylegan_size=1024):
        """Initialise a fresh encoder for training: IR-SE-50 (ImageNet/ArcFace) + pretrained StyleGAN2."""
        net = cls(stylegan_size=stylegan_size)
        if irse50_ckpt is not None:
            net.encoder.load_state_dict(torch_load(irse50_ckpt, map_location='cpu'), strict=False)
        ckpt = torch_load(stylegan_ckpt, map_location='cpu')
        net.decoder.load_state_dict(ckpt['g_ema'], strict=False)
        if 'latent_avg' in ckpt:
            net.latent_avg.copy_(ckpt['latent_avg'].view(-1, 512).expand(net.n_styles, 512))
        else:
            with torch.no_grad():
                net.latent_avg.copy_(net.decoder.mean_latent(int(1e5))[0].expand(net.n_styles, 512))
        return net

    @classmethod
    def from_checkpoint(cls, checkpoint_path, stylegan_size=None):
        """Load a trained VR-FAM encoder (or an official pSp/e4e checkpoint with the same layout)."""
        ckpt = torch_load(checkpoint_path, map_location='cpu')
        if stylegan_size is None:
            stylegan_size = ckpt.get('opts', {}).get('stylegan_size', 1024)
        net = cls(stylegan_size=stylegan_size)
        net.encoder.load_state_dict(_filter_keys(ckpt, 'encoder'), strict=True)
        net.decoder.load_state_dict(_filter_keys(ckpt, 'decoder'), strict=True)
        if ckpt.get('latent_avg') is not None:
            net.latent_avg.copy_(ckpt['latent_avg'].view(-1, 512).expand(net.n_styles, 512))
        return net

    def checkpoint_dict(self, **extra):
        return {'state_dict': self.state_dict(), 'latent_avg': self.latent_avg.cpu(),
                'opts': {'stylegan_size': self.stylegan_size}, **extra}

    # ---------------------------------------------------------------- inference
    def encode(self, x):
        """Images in [-1, 1] at 256x256 -> W+ codes of shape [B, n_styles, 512]."""
        return self.encoder(x) + self.latent_avg.unsqueeze(0)

    def synthesize(self, codes, resize=True, randomize_noise=False):
        images, _ = self.decoder([codes], input_is_latent=True, randomize_noise=randomize_noise)
        return self.face_pool(images) if resize else images

    def forward(self, x, resize=True, randomize_noise=False, return_latents=False):
        codes = self.encode(x)
        images = self.synthesize(codes, resize=resize, randomize_noise=randomize_noise)
        return (images, codes) if return_latents else images
