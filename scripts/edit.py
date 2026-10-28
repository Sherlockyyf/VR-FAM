"""Edit facial attributes of real images (or pre-computed W+ codes) with the latent transformers.

Single attribute, several strengths (one row per image: input | reconstruction | edits):
    python scripts/edit.py --encoder pretrained_models/vr_encoder_celebahq.pt \
        --transformers pretrained_models/transformers --images examples \
        --attr Smiling --alphas -1.5 -1 -0.5 0.5 1 1.5 --output outputs/smiling

Sequential editing (each column adds one more attribute):
    python scripts/edit.py --encoder pretrained_models/vr_encoder_celebahq.pt \
        --transformers pretrained_models/transformers --images examples \
        --edits "Smiling=1,Arched_Eyebrows=1,Young=-1,Eyeglasses=1.5" --output outputs/sequential

Add ``--save_individual`` to also dump every edited image into ``<output>/<attr>_<alpha>/`` (the
layout expected by ``scripts/evaluate.py``).
"""
import argparse
import os
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vrfam.data import ImageFolderDataset, encoder_transforms  # noqa: E402
from vrfam.editor import LatentEditor, parse_edits  # noqa: E402
from vrfam.models import VREncoder, load_generator  # noqa: E402
from vrfam.utils import get_device, load_latent, make_row, tensor2im  # noqa: E402


def iterate_inputs(args, net, device):
    """Yields ``(name, input_image_or_None, w)``."""
    if args.latents:
        files = [args.latents] if os.path.isfile(args.latents) else sorted(
            os.path.join(args.latents, f) for f in os.listdir(args.latents) if f.endswith(('.pt', '.npy')))
        for path in files:
            yield os.path.splitext(os.path.basename(path))[0], None, load_latent(path, device)
    else:
        dataset = ImageFolderDataset(args.images, encoder_transforms(train=False), return_path=True)
        for image, path in dataset:
            image = image.unsqueeze(0).to(device)
            with torch.no_grad():
                w = net.encode(image)
            yield os.path.splitext(os.path.basename(path))[0], image[0], w


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--images', help='aligned face image or folder (requires --encoder)')
    src.add_argument('--latents', help='W+ code (.pt/.npy) or folder of codes')
    parser.add_argument('--encoder', default=None, help='VR-FAM encoder checkpoint (its decoder is used for synthesis)')
    parser.add_argument('--stylegan', default='pretrained_models/stylegan2-ffhq-config-f.pt',
                        help='generator used with --latents when no --encoder is given')
    parser.add_argument('--transformers', required=True, help='folder with tnet_<Attribute>.pth files')
    parser.add_argument('--attr', default=None, help='attribute to edit, e.g. Smiling / Male / Eyeglasses / Young')
    parser.add_argument('--alphas', type=float, nargs='+', default=[-1.5, -1.0, -0.5, 0.5, 1.0, 1.5])
    parser.add_argument('--edits', default=None, help='sequential edits, e.g. "Smiling=1,Eyeglasses=1.5"')
    parser.add_argument('--edit_layers', type=int, default=11, help='only the first N style layers are edited')
    parser.add_argument('--output', required=True)
    parser.add_argument('--save_individual', action='store_true')
    parser.add_argument('--device', default=None)
    args = parser.parse_args()
    if (args.attr is None) == (args.edits is None):
        parser.error('specify exactly one of --attr or --edits')

    device = get_device(args.device)
    if args.encoder:
        net = VREncoder.from_checkpoint(args.encoder).to(device).eval()
        generator = net.decoder
    else:
        if args.images:
            parser.error('--images requires --encoder')
        net, generator = None, load_generator(args.stylegan).to(device)

    editor = LatentEditor(args.transformers, device=device, edit_layers=args.edit_layers)
    os.makedirs(args.output, exist_ok=True)

    def synth(w):
        with torch.no_grad():
            return generator([w], input_is_latent=True, randomize_noise=False)[0][0]

    for name, image, w in tqdm(list(iterate_inputs(args, net, device)), desc='Editing'):
        row = ([image] if image is not None else []) + [synth(w)]
        if args.attr:
            for alpha in args.alphas:
                edited = synth(editor.edit(w, args.attr, alpha))
                row.append(edited)
                if args.save_individual:
                    out_dir = os.path.join(args.output, f'{args.attr}_{alpha:g}')
                    os.makedirs(out_dir, exist_ok=True)
                    tensor2im(edited).save(os.path.join(out_dir, f'{name}.jpg'))
        else:
            edits = parse_edits(args.edits)
            for k in range(1, len(edits) + 1):
                row.append(synth(editor.sequential_edit(w, edits[:k])))
        make_row(row).save(os.path.join(args.output, f'{name}.jpg'))
    print(f'Results saved to {args.output}')


if __name__ == '__main__':
    main()
