"""Loss functions of VR-FAM.

* Encoder (Eq. 1-3, 5, 9): pixel L2, LPIPS, identity, delta regularisation and the
  non-saturating latent adversarial loss with R1 penalty.
* Latent transformer (Eq. 6-8, 10): attribute classification, attribute preservation,
  latent regularisation and (optional) identity loss.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import autograd, nn

from .models.encoders import Backbone
from .io import torch_load


class IDLoss(nn.Module):
    """1 - cos(R(x), R(y)) with the IR-SE-50 ArcFace network R (Eq. 5 and Eq. 8)."""

    def __init__(self, ir_se50_path):
        super().__init__()
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch_load(ir_se50_path, map_location='cpu'))
        self.facenet.eval()
        self.face_pool = nn.AdaptiveAvgPool2d((112, 112))
        for p in self.facenet.parameters():
            p.requires_grad_(False)

    def train(self, mode=True):  # the face recogniser always stays in eval mode
        return super().train(False)

    def extract_feats(self, x):
        if x.shape[-1] != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        x = x[:, :, 35:223, 32:220]  # crop the face region of an FFHQ-aligned 256x256 image
        return self.facenet(self.face_pool(x))

    def similarity(self, x, y):
        return (self.extract_feats(x) * self.extract_feats(y)).sum(dim=1)

    def forward(self, y_hat, y):
        return (1 - self.similarity(y_hat, y.detach())).mean()


def delta_regularization(codes, num_deltas, norm=2):
    """Eq. (1): sum_i ||delta_i||, delta_i = w_i - w_0, over the residuals enabled so far."""
    first_w = codes[:, 0, :]
    total = codes.new_zeros(())
    for i in range(1, num_deltas + 1):
        total = total + torch.norm(codes[:, i, :] - first_w, norm, dim=1).mean()
    return total


def g_nonsaturating_loss(fake_pred):
    """Eq. (3): -log D(E(x))."""
    return F.softplus(-fake_pred).mean()


def d_logistic_loss(real_pred, fake_pred):
    """Eq. (2) without the R1 term."""
    return F.softplus(-real_pred).mean() + F.softplus(fake_pred).mean()


def d_r1_loss(real_pred, real_w):
    (grad_real,) = autograd.grad(outputs=real_pred.sum(), inputs=real_w, create_graph=True)
    return grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()


# --------------------------------------------------------------------------- transformer
def editing_targets(prob):
    """Flip every sample's attribute (as in the latent transformer of Yao et al.).

    Returns the target label (0/1) and the scaling factor ``alpha`` that moves the predicted
    probability to the opposite side: ``alpha = -p`` if the attribute is present, ``1 - p`` otherwise.
    """
    present = prob > 0.5
    target = (~present).to(prob.dtype)
    alpha = torch.where(present, -prob, 1 - prob)
    return target, alpha


def classification_loss(logits, target):
    """Eq. (6), class-balanced binary cross entropy on the edited attribute."""
    n = target.numel()
    pos_weight = n / (target.sum() + 1e-8)
    neg_weight = n / (n - target.sum() + 1e-8)
    weight = pos_weight * target + neg_weight * (1 - target)
    return (F.binary_cross_entropy_with_logits(logits, target, reduction='none') * weight).mean()


def attribute_preservation_loss(logits_edit, logits_orig, mask):
    """Eq. (7): keep the logits of the attributes that are not correlated with the edited one."""
    return F.mse_loss(logits_edit * mask, logits_orig * mask)


def correlation_mask(labels, attr_index, threshold=1.0):
    """``1 - |rho_ik|`` from the attribute correlation matrix; attributes with |rho| >= threshold are ignored."""
    corr = np.corrcoef(np.asarray(labels, dtype=np.float64).T)
    corr[np.isnan(corr)] = 0
    rho = np.abs(corr[attr_index]).copy()
    rho[rho >= threshold] = 1
    return torch.from_numpy(1 - rho).float()
