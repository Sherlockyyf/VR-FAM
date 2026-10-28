"""Compare edited / reconstructed images with the source images.

Images are matched by file name (without extension) and resized to 256x256 for the paired metrics.

Example:
    python scripts/evaluate.py --source_dir data/ffhq/test --result_dir outputs/smiling/Smiling_1 \
        --metrics l2 lpips ssim id fid --id_weights pretrained_models/model_ir_se50.pth
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vrfam.data import list_images  # noqa: E402
from vrfam.metrics import IdentitySimilarity, fid_between_folders, ms_ssim, ssim  # noqa: E402
from vrfam.utils import get_device  # noqa: E402

TO_TENSOR = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])


def paired_files(source_dir, result_dir):
    sources = {os.path.splitext(os.path.basename(p))[0]: p for p in list_images(source_dir)}
    pairs = []
    for path in list_images(result_dir):
        name = os.path.splitext(os.path.basename(path))[0]
        if name in sources:
            pairs.append((sources[name], path))
    if not pairs:
        raise RuntimeError('No matching file names between the two folders')
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source_dir', required=True)
    parser.add_argument('--result_dir', required=True)
    parser.add_argument('--metrics', nargs='+', default=['l2', 'lpips', 'ssim', 'msssim'],
                        choices=['l2', 'lpips', 'ssim', 'msssim', 'id', 'fid'])
    parser.add_argument('--id_model', default='ir_se50', choices=['ir_se50', 'curricular'])
    parser.add_argument('--id_weights', default='pretrained_models/model_ir_se50.pth')
    parser.add_argument('--output', default=None, help='optional json file for the scores')
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    device = get_device(args.device)
    paired = [m for m in args.metrics if m != 'fid']
    lpips_fn = None
    if 'lpips' in paired:
        import lpips
        lpips_fn = lpips.LPIPS(net='alex', verbose=False).to(device).eval()
    id_fn = IdentitySimilarity(args.id_weights, args.id_model).to(device) if 'id' in paired else None

    scores = {m: [] for m in paired}
    if paired:
        for src_path, res_path in tqdm(paired_files(args.source_dir, args.result_dir), desc='Evaluating'):
            x = TO_TENSOR(Image.open(src_path).convert('RGB')).unsqueeze(0).to(device)  # [0, 1]
            y = TO_TENSOR(Image.open(res_path).convert('RGB')).unsqueeze(0).to(device)
            with torch.no_grad():
                if 'l2' in scores:
                    scores['l2'].append(torch.mean((x - y) ** 2).item())
                if 'lpips' in scores:
                    scores['lpips'].append(lpips_fn(x * 2 - 1, y * 2 - 1).item())
                if 'ssim' in scores:
                    scores['ssim'].append(ssim(x, y).item())
                if 'msssim' in scores:
                    scores['msssim'].append(ms_ssim(x, y).item())
                if 'id' in scores:
                    scores['id'].append(id_fn(x * 2 - 1, y * 2 - 1).item())

    summary = {m: {'mean': float(np.mean(v)), 'std': float(np.std(v)), 'n': len(v)} for m, v in scores.items()}
    if 'fid' in args.metrics:
        summary['fid'] = {'mean': float(fid_between_folders(args.source_dir, args.result_dir, device))}
    for m, s in summary.items():
        extra = f" +- {s['std']:.4f} (n={s['n']})" if 'std' in s else ''
        print(f'{m:>7s}: {s["mean"]:.4f}{extra}')
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
