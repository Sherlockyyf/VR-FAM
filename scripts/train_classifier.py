"""Pre-train the latent attribute classifier C on (W+ code, CelebA attribute) pairs.

Example:
    python scripts/train_classifier.py --latents data/latents/celeba_hq/latents.npy \
        --labels data/celeba_hq/labels.npy --exp_dir experiments/classifier
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vrfam.attributes import CELEBA_ATTRIBUTES  # noqa: E402
from vrfam.data import LatentAttributeDataset  # noqa: E402
from vrfam.models import LatentClassifier  # noqa: E402
from vrfam.utils import add_config_args, get_device, load_config, seed_everything  # noqa: E402


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total, loss = 0, 0, 0.0
    for w, y in loader:
        w, y = w.to(device), y.to(device)
        logits = model(w)
        loss += F.binary_cross_entropy_with_logits(logits, y, reduction='sum').item()
        correct = correct + ((logits > 0).float() == y).float().sum(0)
        total += y.shape[0]
    model.train()
    return loss / (total * y.shape[1]), (correct / total).cpu()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser, 'configs/classifier.yaml')
    parser.add_argument('--latents', required=True, help='latents.npy / .pt or a folder of per-image codes')
    parser.add_argument('--labels', required=True, help='[N, 40] label matrix (.npy)')
    parser.add_argument('--exp_dir', required=True)
    parser.add_argument('--device', default=None)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config, args.opts)
    seed_everything(args.seed)
    device = get_device(args.device)
    os.makedirs(args.exp_dir, exist_ok=True)

    train_set = LatentAttributeDataset(args.latents, args.labels, 'train', cfg['train_ratio'])
    val_set = LatentAttributeDataset(args.latents, args.labels, 'val', cfg['train_ratio'])
    train_loader = DataLoader(train_set, batch_size=cfg['batch_size'], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg['batch_size']) if len(val_set) else None
    print(f'{len(train_set)} training / {len(val_set)} validation codes')

    model = LatentClassifier(cfg['fmaps'], cfg['activation']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg['step_size'], gamma=cfg['gamma'])

    for epoch in range(cfg['epochs']):
        running = 0.0
        for w, y in train_loader:
            w, y = w.to(device), y.to(device)
            loss = F.binary_cross_entropy_with_logits(model(w), y)  # Eq. (1) of DO-FAM / Sec. 2.2 of VR-FAM
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()
        scheduler.step()
        msg = f'epoch {epoch + 1:3d} | train loss {running / len(train_loader):.4f}'
        if val_loader is not None:
            val_loss, acc = evaluate(model, val_loader, device)
            msg += f' | val loss {val_loss:.4f} | mean acc {acc.mean():.4f}'
        print(msg, flush=True)
        torch.save(model.state_dict(), os.path.join(args.exp_dir, f'latent_classifier_epoch_{epoch + 1}.pth'))

    torch.save(model.state_dict(), os.path.join(args.exp_dir, 'latent_classifier.pth'))
    if val_loader is not None:
        _, acc = evaluate(model, val_loader, device)
        print('Validation accuracy per attribute:')
        for name, a in zip(CELEBA_ATTRIBUTES, acc.tolist()):
            print(f'  {name:<20s} {a:.4f}')


if __name__ == '__main__':
    main()
