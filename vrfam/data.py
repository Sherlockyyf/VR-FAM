"""Datasets: aligned face images for the encoder, (latent code, attribute label) pairs for the rest."""
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .io import torch_load

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.tif', '.tiff', '.webp')


def list_images(root):
    if os.path.isfile(root):
        return [root]
    paths = []
    for dirpath, _, fnames in sorted(os.walk(root)):
        for fname in sorted(fnames):
            if fname.lower().endswith(IMG_EXTENSIONS):
                paths.append(os.path.join(dirpath, fname))
    if not paths:
        raise FileNotFoundError(f'No images found in {root}')
    return paths


def encoder_transforms(train=True, size=256):
    ops = [transforms.Resize((size, size))]
    if train:
        ops.append(transforms.RandomHorizontalFlip(0.5))
    ops += [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    return transforms.Compose(ops)


class ImageFolderDataset(Dataset):
    """Returns ``(image, image)`` pairs (input and reconstruction target) in [-1, 1]."""

    def __init__(self, root, transform=None, return_path=False):
        self.paths = list_images(root)
        self.transform = transform or encoder_transforms(train=False)
        self.return_path = return_path

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = self.transform(Image.open(self.paths[idx]).convert('RGB'))
        if self.return_path:
            return image, self.paths[idx]
        return image, image


def load_latents(path):
    """Load stacked W+ codes ([N, 18, 512]) from ``.npy`` / ``.pt`` or a folder of per-image files."""
    if os.path.isdir(path):
        files = sorted(f for f in os.listdir(path) if f.endswith(('.npy', '.pt')))
        return np.stack([np.asarray(load_latents(os.path.join(path, f))).reshape(-1, 512) for f in files])
    if path.endswith('.npy'):
        return np.load(path)
    latents = torch_load(path, map_location='cpu')
    return latents.numpy() if torch.is_tensor(latents) else np.asarray(latents)


def load_labels(path):
    """Load the [N, 40] binary attribute matrix produced by ``scripts/prepare_celebahq_labels.py``."""
    if path.endswith('.npy'):
        return np.load(path)
    labels = torch_load(path, map_location='cpu')
    if isinstance(labels, dict):  # {image_name: label}
        labels = [labels[k] for k in sorted(labels)]
    return np.asarray(labels)


class LatentAttributeDataset(Dataset):
    """``(w+, labels)`` pairs. The first ``train_ratio`` of the samples form the training split."""

    def __init__(self, latent_path, label_path, split='train', train_ratio=0.9):
        latents = load_latents(latent_path)
        labels = load_labels(label_path)
        if len(latents) != len(labels):
            raise ValueError(f'{len(latents)} latent codes but {len(labels)} labels')
        n_train = int(train_ratio * len(labels))
        sl = slice(0, n_train) if split == 'train' else slice(n_train, None)
        self.latents = torch.as_tensor(np.asarray(latents[sl]), dtype=torch.float32)
        self.labels = torch.as_tensor(np.asarray(labels[sl]), dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.latents[idx], self.labels[idx]
