"""Invert face images into the W+ space with the (variance-reduced) encoder.

For every image ``<name>.jpg`` this writes ``<output>/latents/<name>.pt`` ([1, 18, 512]); all codes
are additionally stacked into ``<output>/latents.npy`` (sorted by file name, the order expected by
``prepare_celebahq_labels.py`` / ``train_classifier.py``). Reconstructions are optional.

Example:
    python scripts/invert.py --checkpoint pretrained_models/vr_encoder_celebahq.pt \
        --input data/celeba_hq/train --output data/latents/celeba_hq_train --batch_size 8
"""
import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vrfam.data import ImageFolderDataset, encoder_transforms, list_images  # noqa: E402
from vrfam.models import VREncoder  # noqa: E402
from vrfam.utils import get_device, make_row  # noqa: E402


def align_inputs(input_path, predictor_path, out_dir):
    """Align in-the-wild photos (dlib) before inversion; returns the folder with aligned faces."""
    import dlib

    from vrfam.alignment import align_face

    predictor = dlib.shape_predictor(predictor_path)
    os.makedirs(out_dir, exist_ok=True)
    for path in tqdm(list_images(input_path), desc='Aligning'):
        try:
            align_face(path, predictor).save(os.path.join(out_dir, os.path.splitext(os.path.basename(path))[0] + '.png'))
        except RuntimeError as err:
            print(f'Skip {path}: {err}')
    return out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True, help='VR-FAM encoder (or an official e4e/pSp checkpoint)')
    parser.add_argument('--input', required=True, help='image file or folder')
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--save_reconstruction', action='store_true')
    parser.add_argument('--align', action='store_true', help='align faces with dlib first')
    parser.add_argument('--shape_predictor', default='pretrained_models/shape_predictor_68_face_landmarks.dat')
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    device = get_device(args.device)
    input_path = args.input
    if args.align:
        input_path = align_inputs(input_path, args.shape_predictor, os.path.join(args.output, 'aligned'))

    net = VREncoder.from_checkpoint(args.checkpoint).to(device).eval()
    dataset = ImageFolderDataset(input_path, encoder_transforms(train=False), return_path=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    latent_dir = os.path.join(args.output, 'latents')
    recon_dir = os.path.join(args.output, 'reconstructions')
    os.makedirs(latent_dir, exist_ok=True)
    if args.save_reconstruction:
        os.makedirs(recon_dir, exist_ok=True)

    all_codes, names = [], []
    with torch.no_grad():
        for images, paths in tqdm(loader, desc='Inverting'):
            images = images.to(device)
            codes = net.encode(images)
            if args.save_reconstruction:
                recon = net.synthesize(codes, resize=False)
            for i, path in enumerate(paths):
                name = os.path.splitext(os.path.basename(path))[0]
                torch.save(codes[i:i + 1].cpu(), os.path.join(latent_dir, f'{name}.pt'))
                all_codes.append(codes[i].cpu().numpy())
                names.append(name)
                if args.save_reconstruction:
                    make_row([images[i], recon[i]]).save(os.path.join(recon_dir, f'{name}.jpg'))

    np.save(os.path.join(args.output, 'latents.npy'), np.stack(all_codes).astype(np.float32))
    with open(os.path.join(args.output, 'names.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(names))
    print(f'Saved {len(names)} latent codes to {args.output}')


if __name__ == '__main__':
    main()
