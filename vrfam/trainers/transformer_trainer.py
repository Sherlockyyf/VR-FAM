"""Training loop of the nonlinear latent transformation network (Sec. 2.2, Eq. 6-8 and 10)."""
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..attributes import attribute_index, canonical_name
from ..editor import build_transformer
from ..losses import (attribute_preservation_loss, classification_loss, correlation_mask,
                      editing_targets)
from ..utils import make_row

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


class TransformerTrainer:
    """Trains one latent transformer T_k for attribute ``attr``.

    Args:
        classifier: frozen :class:`LatentClassifier`.
        train_labels: [N, 40] label matrix used to compute the attribute correlation mask.
        generator / id_loss: only needed for the optional identity loss (``lambda.id > 0``) and
            for visualising samples during training.
    """

    def __init__(self, cfg, attr, classifier, train_labels, exp_dir, device, generator=None, id_loss=None):
        self.cfg = cfg
        self.attr = canonical_name(attr)
        self.k = attribute_index(self.attr)
        self.exp_dir = exp_dir
        self.device = device
        self.classifier = classifier.eval()
        for p in self.classifier.parameters():
            p.requires_grad_(False)
        self.generator = generator
        self.id_loss = id_loss

        self.model_cfg = cfg['model']
        self.net = build_transformer(self.model_cfg).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=cfg['lr'], betas=tuple(cfg['betas']),
                                          weight_decay=cfg['weight_decay'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=cfg['step_size'],
                                                         gamma=cfg['gamma'])
        self.mask = correlation_mask(train_labels, self.k, cfg['corr_threshold']).to(device)
        self.writer = SummaryWriter(os.path.join(exp_dir, 'logs', self.attr)) if SummaryWriter else None
        os.makedirs(os.path.join(exp_dir, 'images', self.attr), exist_ok=True)

    def compute_loss(self, w):
        lam = self.cfg['lambda']
        logits0 = self.classifier(w)
        prob0 = torch.sigmoid(logits0[:, self.k])
        target, alpha = editing_targets(prob0)
        if not self.cfg.get('adaptive_alpha', True):
            alpha = 2 * target - 1

        w_edit = self.net(w, alpha)
        logits1 = self.classifier(w_edit)

        logs = {}
        loss_cls = classification_loss(logits1[:, self.k], target)
        loss_attr = attribute_preservation_loss(logits1, logits0, self.mask)
        loss = lam['cls'] * loss_cls + lam['attr'] * loss_attr
        logs.update(loss_cls=loss_cls.item(), loss_attr=loss_attr.item())
        if lam.get('reg', 0) > 0:
            loss_reg = F.mse_loss(w_edit, w)
            loss = loss + lam['reg'] * loss_reg
            logs['loss_reg'] = loss_reg.item()
        if lam.get('id', 0) > 0:
            loss_id = self._identity_loss(w, w_edit)
            loss = loss + lam['id'] * loss_id
            logs['loss_id'] = loss_id.item()
        logs['loss'] = loss.item()
        return loss, logs

    def _identity_loss(self, w, w_edit):
        """Eq. (8): 1 - <R(G(w)), R(G(T(w)))> on a random subset of the batch (synthesis is expensive)."""
        idx = torch.randperm(w.shape[0], device=w.device)[:self.cfg.get('id_subset', 4)]
        with torch.no_grad():
            x, _ = self.generator([w[idx]], input_is_latent=True, randomize_noise=False)
        x_edit, _ = self.generator([w_edit[idx]], input_is_latent=True, randomize_noise=False)
        return self.id_loss(F.interpolate(x_edit, size=256, mode='area'), F.interpolate(x, size=256, mode='area'))

    def train(self, dataset, sample_w=None):
        cfg = self.cfg
        loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, drop_last=True,
                            num_workers=cfg.get('workers', 0))
        step = 0
        tic = time.time()
        for epoch in range(cfg['epochs']):
            for w, _ in loader:
                w = w.to(self.device)
                loss, logs = self.compute_loss(w)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if step % cfg['log_interval'] == 0:
                    msg = ', '.join(f'{k}={v:.4f}' for k, v in logs.items())
                    print(f'[{self.attr} | epoch {epoch} | step {step} | {time.time() - tic:.0f}s] {msg}', flush=True)
                    if self.writer:
                        for k, v in logs.items():
                            self.writer.add_scalar(k, v, step)
                if sample_w is not None and self.generator is not None and step % cfg['image_interval'] == 0:
                    self.log_samples(sample_w, step)
                step += 1
            self.scheduler.step()
        return self.save()

    @torch.no_grad()
    def log_samples(self, w, step, alphas=(-1.0, -0.5, 0.5, 1.0)):
        self.net.eval()
        w = w.to(self.device)
        images = [self.generator([w], input_is_latent=True, randomize_noise=False)[0][0]]
        for a in alphas:
            w_edit = self.net(w, a)
            images.append(self.generator([w_edit], input_is_latent=True, randomize_noise=False)[0][0])
        make_row(images).save(os.path.join(self.exp_dir, 'images', self.attr, f'step_{step:06d}.jpg'))
        self.net.train()

    def save(self):
        path = os.path.join(self.exp_dir, 'checkpoints', f'tnet_{self.attr}.pth')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({'state_dict': self.net.state_dict(), 'attr': self.attr, 'model': self.model_cfg}, path)
        print(f'Saved {path}')
        return path
