"""Train the variance-reduced encoder.

Example:
    python scripts/train_encoder.py --config configs/encoder.yaml --exp_dir experiments/vr_encoder
    python scripts/train_encoder.py --config configs/encoder.yaml --exp_dir experiments/vr_encoder \
        --resume experiments/vr_encoder/checkpoints/iteration_50000.pt
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vrfam.trainers.encoder_trainer import EncoderTrainer  # noqa: E402
from vrfam.utils import add_config_args, get_device, load_config, seed_everything  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser, 'configs/encoder.yaml')
    parser.add_argument('--exp_dir', required=True, help='output folder (checkpoints, images, logs)')
    parser.add_argument('--resume', default=None, help='checkpoint to resume training from')
    parser.add_argument('--device', default=None)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config, args.opts)
    seed_everything(args.seed)
    trainer = EncoderTrainer(cfg, args.exp_dir, get_device(args.device), resume=args.resume)
    trainer.train()


if __name__ == '__main__':
    main()
