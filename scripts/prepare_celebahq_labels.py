"""Convert the CelebAMask-HQ attribute annotation into a [N, 40] {0, 1} label matrix.

The rows follow the *sorted* image files of ``--images_dir`` (the same order used by
``scripts/invert.py`` when it stacks latent codes), so latents and labels stay aligned.
Without ``--images_dir`` all 30,000 images are written in index order.

Example:
    python scripts/prepare_celebahq_labels.py \
        --anno data/CelebAMask-HQ/CelebAMask-HQ-attribute-anno.txt \
        --images_dir data/celeba_hq/train --output data/celeba_hq/train_labels.npy
"""
import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vrfam.attributes import CELEBA_ATTRIBUTES  # noqa: E402
from vrfam.data import list_images  # noqa: E402


def read_annotation(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = [line.split() for line in f if line.strip()]
    names = lines[1]
    assert names == CELEBA_ATTRIBUTES, 'Unexpected attribute order in the annotation file'
    table = {}
    for row in lines[2:]:
        index = int(re.findall(r'\d+', row[0])[0])
        table[index] = (np.array(row[1:], dtype=np.int64) > 0).astype(np.float32)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--anno', required=True, help='CelebAMask-HQ-attribute-anno.txt')
    parser.add_argument('--images_dir', default=None, help='use the order of the sorted images in this folder')
    parser.add_argument('--output', required=True, help='output .npy file')
    args = parser.parse_args()

    table = read_annotation(args.anno)
    if args.images_dir:
        indices = [int(re.findall(r'\d+', os.path.basename(p))[-1]) for p in list_images(args.images_dir)]
    else:
        indices = sorted(table)
    labels = np.stack([table[i] for i in indices])
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    np.save(args.output, labels)
    print(f'Saved labels of shape {labels.shape} to {args.output}')
    print('Positive ratio per attribute:')
    for name, ratio in zip(CELEBA_ATTRIBUTES, labels.mean(0)):
        print(f'  {name:<20s} {ratio:.3f}')


if __name__ == '__main__':
    main()
