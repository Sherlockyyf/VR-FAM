<div align="center">

# VR-FAM: Variance-Reduced Encoder with Nonlinear Transformation for Facial Attribute Manipulation

[**Yifan Yuan**](https://yifanyuan.com) · **Siteng Ma** · **Junping Zhang**<sup>&dagger;</sup>

Shanghai Key Lab of Intelligent Information Processing, School of Computer Science, Fudan University
<br>
<sup>&dagger;</sup>Corresponding author

**ICASSP 2022**

<a href="https://doi.org/10.1109/ICASSP43922.2022.9746046"><img src="https://img.shields.io/badge/Paper-ICASSP%202022-blue" alt="Paper"></a>
<a href="https://github.com/Sherlockyyf/DO-FAM"><img src="https://img.shields.io/badge/Follow--up-DO--FAM%20(ICASSP%202023)-orange" alt="DO-FAM"></a>
<a href="https://pytorch.org"><img src="https://img.shields.io/badge/PyTorch-%E2%89%A51.8-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-lightgrey" alt="License"></a>

[English](README.md) · [简体中文](README_zh.md)

</div>

<p align="center"><img src="assets/sequential_editing.jpg" width="85%"><br><em>Sequential facial attribute editing on real images.</em></p>

**TL;DR:** A StyleGAN-based facial attribute manipulation (FAM) framework that balances the three requirements of FAM — *editing ability*, *distortion* and *fidelity*. A **variance-reduced encoder** keeps the inverted W+ codes close to StyleGAN's W space, and a **nonlinear latent transformation network** moves them towards the desired attribute while keeping identity and the other attributes unchanged.

## 🔍 Method

<p align="center"><img src="assets/framework.png" width="100%"></p>

VR-FAM consists of a variance-reduced encoder, a nonlinear latent transformation network and a StyleGAN2 generator:

- **Variance-reduced encoder (Sec. 2.1).** The encoder (pSp / e4e architecture) outputs $w^+ = (w, w+\Delta_1, \dots, w+\Delta_{L-1})$. It first learns the single code $w$ and then the residuals $\Delta_i$ one by one. An L2 penalty on the residuals, $\mathcal{L}_{reg} = \sum_i \lVert\Delta_i\rVert_2$, reduces the variance between the style codes, and a latent discriminator $D_W$ (non-saturating loss + R1) keeps them inside the distribution of $W$. Reconstruction is supervised with $\mathcal{L}_2$, LPIPS and an ArcFace identity loss.
- **Nonlinear latent transformation (Sec. 2.2).** For every attribute $a_k$, a network $T_k(w^+, \alpha) = w^+ + \alpha \cdot f(w^+)$ with a nonlinear $f$ (three fully-connected layers per style layer) edits the code. It is trained with a pre-trained latent attribute classifier $C$: a classification loss on the target attribute and an attribute-preservation loss on the others, weighted by $1-|\gamma_{ik}|$ (attribute correlation).

## 🛠️ Installation

```bash
git clone https://github.com/Sherlockyyf/VR-FAM.git
cd VR-FAM
conda env create -f environment.yml && conda activate vrfam      # or: pip install -r requirements.txt
```

The StyleGAN2 operators are implemented in pure PyTorch, so nothing needs to be compiled.

## 📦 Pretrained Models

Put the files into `pretrained_models/`.

| Model | Description | Download |
|:-|:-|:-:|
| `stylegan2-ffhq-config-f.pt` | StyleGAN2 FFHQ 1024×1024 generator (rosinality format) | [pSp model zoo](https://github.com/eladrich/pixel2style2pixel#pretrained-models) |
| `model_ir_se50.pth` | IR-SE50 ArcFace, for the identity loss / metric | [pSp model zoo](https://github.com/eladrich/pixel2style2pixel#pretrained-models) |
| `vr_encoder_celebahq.pt` | VR-FAM variance-reduced encoder | coming soon |
| `latent_classifier.pth` | latent attribute classifier (40 CelebA attributes) | coming soon |
| `transformers/tnet_<Attribute>.pth` | nonlinear latent transformers | coming soon |
| `shape_predictor_68_face_landmarks.dat` | dlib landmarks, only for aligning in-the-wild photos | [dlib](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) |

The official [e4e encoder](https://github.com/omertov/encoder4editing) and the latent classifier of [Latent Transformer](https://github.com/InterDigitalInc/latent-transformer) can be loaded directly as well. Checkpoints of the original linear latent transformer (`tnet_<index>.pth.tar`) are converted on the fly.

## 🚀 Quick Start

Edit aligned face images (FFHQ alignment; add `--align` to `invert.py` for in-the-wild photos):

```bash
# one attribute, several strengths: input | reconstruction | edits
python scripts/edit.py --encoder pretrained_models/vr_encoder_celebahq.pt --transformers pretrained_models/transformers \
    --images path/to/aligned_faces --attr Smiling --alphas -1.5 -1 -0.5 0.5 1 1.5 --output outputs/smiling

# sequential editing: every column adds one attribute
python scripts/edit.py --encoder pretrained_models/vr_encoder_celebahq.pt --transformers pretrained_models/transformers \
    --images path/to/aligned_faces --edits "Smiling=1,Arched_Eyebrows=1,Young=-1,Eyeglasses=1.5" --output outputs/sequential
```

<details>
<summary><b>Python API</b></summary>

```python
import torch
from vrfam.models import VREncoder
from vrfam.editor import LatentEditor
from vrfam.data import encoder_transforms
from vrfam.utils import tensor2im
from PIL import Image

net = VREncoder.from_checkpoint('pretrained_models/vr_encoder_celebahq.pt').cuda().eval()
editor = LatentEditor('pretrained_models/transformers', device='cuda', edit_layers=11)

x = encoder_transforms(train=False)(Image.open('face.png').convert('RGB')).unsqueeze(0).cuda()
with torch.no_grad():
    w = net.encode(x)                                        # [1, 18, 512]
    w_edit = editor.sequential_edit(w, [('Smiling', 1.0), ('Eyeglasses', 1.5)])
    image = net.synthesize(w_edit, resize=False)[0]          # 1024 x 1024
tensor2im(image).save('edited.png')
```
</details>

Attributes follow the CelebA names (`Smiling`, `Male`, `Young`, `Eyeglasses`, `Bald`, `Bangs`, ...); `Gender` and `Age` are accepted as aliases of `Male` and `Young`.

## 🏋️ Training

**1. Data.** [CelebA-HQ](https://github.com/tkarras/progressive_growing_of_gans) (training, with the attribute annotation of [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ)) and [FFHQ](https://github.com/NVlabs/ffhq-dataset) (test), aligned like FFHQ.

```text
data/
├── celeba_hq/train/          # 0.jpg ... (named by CelebA-HQ index)
├── ffhq/test/
└── CelebAMask-HQ/CelebAMask-HQ-attribute-anno.txt
```

**2. Variance-reduced encoder.** `w` alone for 20K steps, then one residual every 2K steps (batch 4, Adam 1e-4, λ<sub>ℓ2</sub>=1, λ<sub>perp</sub>=0.8, λ<sub>ID</sub>=0.1, λ<sub>reg</sub>=2e-4, λ<sub>adv</sub>=0.1, see `configs/encoder.yaml`):

```bash
python scripts/train_encoder.py --config configs/encoder.yaml --exp_dir experiments/vr_encoder
```

**3. Latent codes and labels.**

```bash
python scripts/invert.py --checkpoint experiments/vr_encoder/checkpoints/best_model.pt \
    --input data/celeba_hq/train --output inversion/celeba_hq_train
python scripts/prepare_celebahq_labels.py --anno data/CelebAMask-HQ/CelebAMask-HQ-attribute-anno.txt \
    --images_dir data/celeba_hq/train --output inversion/celeba_hq_train/labels.npy
```

**4. Latent classifier** (9216 → 2048 → 512 → 40):

```bash
python scripts/train_classifier.py --latents inversion/celeba_hq_train/latents.npy \
    --labels inversion/celeba_hq_train/labels.npy --exp_dir experiments/classifier
```

**5. Nonlinear latent transformers** (one per attribute, `configs/transformer.yaml`):

```bash
python scripts/train_transformer.py --latents inversion/celeba_hq_train/latents.npy \
    --labels inversion/celeba_hq_train/labels.npy --classifier experiments/classifier/latent_classifier.pth \
    --exp_dir experiments/transformers --attributes Smiling Male Young Eyeglasses
```

The identity loss of Eq. (8) requires synthesising images during training; enable it with `--opts lambda.id=1.0 --generator pretrained_models/stylegan2-ffhq-config-f.pt`.

## 📊 Evaluation

```bash
python scripts/edit.py ... --attr Smiling --alphas 1.0 --save_individual --output outputs/eval   # writes outputs/eval/Smiling_1/
python scripts/evaluate.py --source_dir data/ffhq/test --result_dir outputs/eval/Smiling_1 --metrics l2 lpips msssim id fid
```

| Method | Distortion (L2) ↓ | Perception (FID) ↓ | Editability (FID) ↓ |
|:-|:-:|:-:|:-:|
| InterFaceGAN | 0.049 | 74.92 | 130.23 |
| e4e | 0.053 | **30.96** | **81.08** |
| **VR-FAM (ours)** | **0.041** | 32.21 | 83.75 |

<p align="center"><img src="assets/comparison.jpg" width="60%"><br><em>Comparison with InterFaceGAN and e4e.</em></p>

## 📁 Repository Structure

```text
vrfam/
├── models/
│   ├── stylegan2/             # StyleGAN2 generator (pure PyTorch ops)
│   ├── encoders/              # IR-SE50 backbone, progressive e4e-style encoder
│   ├── vr_encoder.py          # variance-reduced encoder (encoder + generator + latent average)
│   ├── discriminator.py       # latent discriminator D_W
│   ├── latent_classifier.py   # latent attribute classifier C
│   └── latent_transformer.py  # nonlinear latent transformation network T
├── trainers/                  # encoder / transformer training loops
├── losses.py · editor.py · metrics.py · data.py · alignment.py
scripts/                       # prepare_celebahq_labels · train_encoder · invert · train_classifier
                               # train_transformer · edit · evaluate
configs/                       # encoder.yaml · classifier.yaml · transformer.yaml
```

## ✅ TODO

- [x] Training and inference code
- [ ] Pretrained checkpoints

## 📝 Citation

```bibtex
@inproceedings{yuan2022vrfam,
  title     = {{VR-FAM}: Variance-Reduced Encoder with Nonlinear Transformation for Facial Attribute Manipulation},
  author    = {Yuan, Yifan and Ma, Siteng and Zhang, Junping},
  booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages     = {1755--1759},
  year      = {2022},
  doi       = {10.1109/ICASSP43922.2022.9746046}
}
```

## 📄 License

[MIT](LICENSE). Third-party code keeps its original license (see `licenses/`).
