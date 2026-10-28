"""Train one nonlinear latent transformer per attribute.

Example:
    python scripts/train_transformer.py --latents data/latents/celeba_hq/latents.npy \
        --labels data/celeba_hq/labels.npy --classifier pretrained_models/latent_classifier.pth \
        --exp_dir experiments/transformers --attributes Smiling Eyeglasses Male Young
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vrfam.data import LatentAttributeDataset  # noqa: E402
from vrfam.models import load_classifier, load_generator  # noqa: E402
from vrfam.trainers.transformer_trainer import TransformerTrainer  # noqa: E402
from vrfam.utils import add_config_args, get_device, load_config, seed_everything  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser, 'configs/transformer.yaml')
    parser.add_argument('--latents', required=True)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--classifier', required=True, help='pretrained latent classifier (.pth)')
    parser.add_argument('--exp_dir', required=True)
    parser.add_argument('--attributes', nargs='*', default=None, help='overrides the list in the config')
    parser.add_argument('--generator', default=None,
                        help='StyleGAN2 / encoder checkpoint; needed for lambda.id > 0 and sample images')
    parser.add_argument('--ir_se50', default='pretrained_models/model_ir_se50.pth')
    parser.add_argument('--device', default=None)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config, args.opts)
    seed_everything(args.seed)
    device = get_device(args.device)

    dataset = LatentAttributeDataset(args.latents, args.labels, 'train', cfg['train_ratio'])
    print(f'{len(dataset)} training codes')

    classifier = load_classifier(args.classifier).to(device)

    generator, id_loss = None, None
    if args.generator:
        generator = load_generator(args.generator).to(device)
        for p in generator.parameters():
            p.requires_grad_(False)
    if cfg['lambda'].get('id', 0) > 0:
        if generator is None:
            raise ValueError('lambda.id > 0 requires --generator')
        from vrfam.losses import IDLoss
        id_loss = IDLoss(args.ir_se50).to(device)

    sample_w = dataset.latents[:1]
    for attr in args.attributes or cfg['attributes']:
        trainer = TransformerTrainer(cfg, attr, classifier, dataset.labels.numpy(), args.exp_dir, device,
                                     generator=generator, id_loss=id_loss)
        trainer.train(dataset, sample_w=sample_w)


if __name__ == '__main__':
    main()
