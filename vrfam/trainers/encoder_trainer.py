"""Training loop of the variance-reduced encoder (Sec. 2.1, Eq. 1-3 and 9 of the paper).

Schedule: the single style code ``w`` is trained alone for ``progressive_start`` steps, then
one residual ``delta_i`` is switched on every ``progressive_step_every`` steps. The generator
stays frozen; only the encoder (and the latent discriminator) are optimised.
"""
import json
import os
import time

import lpips
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from ..data import ImageFolderDataset, encoder_transforms
from ..losses import IDLoss, d_logistic_loss, d_r1_loss, delta_regularization, g_nonsaturating_loss
from ..models import LatentCodesDiscriminator, VREncoder
from ..models.encoders import ProgressiveStage
from ..utils import make_row
from ..io import torch_load

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # tensorboard is optional
    SummaryWriter = None


def requires_grad(module, flag=True):
    for p in module.parameters():
        p.requires_grad_(flag)


class EncoderTrainer:
    def __init__(self, cfg, exp_dir, device, resume=None):
        self.cfg = cfg
        self.exp_dir = exp_dir
        self.device = device
        os.makedirs(os.path.join(exp_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(exp_dir, 'images'), exist_ok=True)
        with open(os.path.join(exp_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(cfg, f, indent=2)

        # ------------------------------------------------------------- networks
        self.net = VREncoder.from_pretrained_backbones(cfg['stylegan_weights'], cfg.get('ir_se50'),
                                                       stylegan_size=cfg['stylegan_size']).to(device)
        requires_grad(self.net.decoder, False)
        self.net.decoder.eval()
        self.n_styles = self.net.n_styles

        betas = tuple(cfg.get('betas', (0.95, 0.999)))
        self.optimizer = torch.optim.Adam(self.net.encoder.parameters(), lr=cfg['lr'], betas=betas)

        self.use_adv = cfg['lambda_adv'] > 0
        if self.use_adv:
            self.discriminator = LatentCodesDiscriminator(512, 4).to(device)
            self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=cfg['d_lr'])

        # ---------------------------------------------------------------- losses
        self.lpips = lpips.LPIPS(net='alex', verbose=False).to(device).eval() if cfg['lambda_lpips'] > 0 else None
        self.id_loss = IDLoss(cfg['ir_se50']).to(device).eval() if cfg['lambda_id'] > 0 else None

        # ------------------------------------------------------------------ data
        train_set = ImageFolderDataset(cfg['train_dir'], encoder_transforms(train=True))
        test_set = ImageFolderDataset(cfg['test_dir'], encoder_transforms(train=False))
        self.train_loader = DataLoader(train_set, batch_size=cfg['batch_size'], shuffle=True,
                                       num_workers=cfg['workers'], drop_last=True)
        self.test_loader = DataLoader(test_set, batch_size=cfg['test_batch_size'], shuffle=False,
                                      num_workers=cfg['workers'], drop_last=False)

        # ---------------------------------------------------- progressive schedule
        start, every = cfg['progressive_start'], cfg['progressive_step_every']
        self.progressive_steps = [0] + [start + i * every for i in range(self.n_styles - 1)]

        self.global_step = 0
        self.best_val_loss = None
        if resume:
            self._load_training_state(resume)
        self._update_progressive_stage(resumed=resume is not None)

        self.writer = SummaryWriter(os.path.join(exp_dir, 'logs')) if SummaryWriter else None

    # ------------------------------------------------------------------ schedule
    def _update_progressive_stage(self, resumed=False):
        for i, step in enumerate(self.progressive_steps):
            if self.global_step == step or (resumed and self.global_step >= step):
                self.net.encoder.set_progressive_stage(ProgressiveStage(i))

    @property
    def stage(self):
        return self.net.encoder.progressive_stage.value

    def _dims_to_discriminate(self):
        return list(range(min(self.stage, self.n_styles - 1) + 1))

    # -------------------------------------------------------------------- losses
    def compute_loss(self, x, y, y_hat, codes):
        cfg = self.cfg
        logs = {}
        loss = 0.0
        if self.use_adv:
            dims = self._dims_to_discriminate()
            loss_adv = sum(g_nonsaturating_loss(self.discriminator(codes[:, i])) for i in dims) / len(dims)
            logs['loss_adv'] = loss_adv.item()
            loss = loss + cfg['lambda_adv'] * loss_adv
        if self.stage < ProgressiveStage.Inference.value:
            loss_reg = delta_regularization(codes, min(self.stage, self.n_styles - 1), cfg['delta_norm'])
            logs['loss_reg'] = float(loss_reg)
            loss = loss + cfg['lambda_reg'] * loss_reg
        if self.id_loss is not None:
            loss_id = self.id_loss(y_hat, y)
            logs['loss_id'] = loss_id.item()
            loss = loss + cfg['lambda_id'] * loss_id
        if cfg['lambda_l2'] > 0:
            loss_l2 = F.mse_loss(y_hat, y)
            logs['loss_l2'] = loss_l2.item()
            loss = loss + cfg['lambda_l2'] * loss_l2
        if self.lpips is not None:
            loss_lpips = self.lpips(y_hat, y).mean()
            logs['loss_lpips'] = loss_lpips.item()
            loss = loss + cfg['lambda_lpips'] * loss_lpips
        logs['loss'] = float(loss)
        return loss, logs

    def train_discriminator(self, x):
        cfg = self.cfg
        requires_grad(self.discriminator, True)
        with torch.no_grad():
            real_w = self.net.decoder.get_latent(torch.randn(x.shape[0], 512, device=self.device))
            fake_w = self.net.encode(x)[:, 0]  # as in e4e, D is trained on the main code w
        loss = d_logistic_loss(self.discriminator(real_w), self.discriminator(fake_w))
        self.d_optimizer.zero_grad()
        loss.backward()
        self.d_optimizer.step()
        logs = {'d_loss': loss.item()}

        if self.global_step % cfg['d_reg_every'] == 0:  # lazy R1 regularisation
            real_w = real_w.detach().requires_grad_(True)
            real_pred = self.discriminator(real_w)
            r1 = d_r1_loss(real_pred, real_w)
            self.d_optimizer.zero_grad()
            (cfg['r1'] / 2 * r1 * cfg['d_reg_every'] + 0 * real_pred[0]).backward()
            self.d_optimizer.step()
            logs['d_r1'] = r1.item()
        requires_grad(self.discriminator, False)
        return logs

    # ---------------------------------------------------------------------- loop
    def train(self):
        cfg = self.cfg
        self.net.encoder.train()
        tic = time.time()
        while self.global_step < cfg['max_steps']:
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                logs = self.train_discriminator(x) if self.use_adv else {}

                codes = self.net.encode(x)
                y_hat = self.net.synthesize(codes, resize=True)
                loss, loss_logs = self.compute_loss(x, y, y_hat, codes)
                logs.update(loss_logs)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                step = self.global_step
                if step % cfg['board_interval'] == 0:
                    msg = ', '.join(f'{k}={v:.4f}' for k, v in logs.items())
                    print(f'[step {step:06d} | stage {self.stage:2d} | {time.time() - tic:.0f}s] {msg}', flush=True)
                    if self.writer:
                        for k, v in logs.items():
                            self.writer.add_scalar(f'train/{k}', v, step)
                if step % cfg['image_interval'] == 0:
                    self._log_images(x, y_hat, 'train')

                val_logs = None
                if step > 0 and (step % cfg['val_interval'] == 0 or step == cfg['max_steps'] - 1):
                    val_logs = self.validate()
                    if self.best_val_loss is None or val_logs['loss'] < self.best_val_loss:
                        self.best_val_loss = val_logs['loss']
                        self.save('best_model.pt', val_logs)
                if step > 0 and step % cfg['save_interval'] == 0:
                    self.save(f'iteration_{step}.pt', val_logs)

                self.global_step += 1
                self._update_progressive_stage()
                if self.global_step >= cfg['max_steps']:
                    break
        self.save('last_model.pt')
        print('Training finished.')

    @torch.no_grad()
    def validate(self, max_batches=50):
        self.net.encoder.eval()
        agg = {}
        n = 0
        for i, (x, y) in enumerate(self.test_loader):
            if i >= max_batches:
                break
            x, y = x.to(self.device), y.to(self.device)
            codes = self.net.encode(x)
            y_hat = self.net.synthesize(codes, resize=True)
            _, logs = self.compute_loss(x, y, y_hat, codes)
            for k, v in logs.items():
                agg[k] = agg.get(k, 0.0) + v
            n += 1
            if i == 0:
                self._log_images(x, y_hat, 'test')
        self.net.encoder.train()
        agg = {k: v / max(n, 1) for k, v in agg.items()}
        print('[val] ' + ', '.join(f'{k}={v:.4f}' for k, v in agg.items()), flush=True)
        if self.writer:
            for k, v in agg.items():
                self.writer.add_scalar(f'test/{k}', v, self.global_step)
        return agg

    def _log_images(self, x, y_hat, name, n=4):
        rows = [make_row([x[i], y_hat[i]]) for i in range(min(n, x.shape[0]))]
        canvas = Image.new('RGB', (rows[0].width, rows[0].height * len(rows)))
        for i, r in enumerate(rows):
            canvas.paste(r, (0, i * r.height))
        canvas.save(os.path.join(self.exp_dir, 'images', f'{name}_{self.global_step:06d}.jpg'))

    # --------------------------------------------------------------- checkpoints
    def save(self, name, val_logs=None):
        state = self.net.checkpoint_dict(global_step=self.global_step, val_logs=val_logs,
                                         optimizer=self.optimizer.state_dict())
        if self.use_adv:
            state['discriminator'] = self.discriminator.state_dict()
            state['d_optimizer'] = self.d_optimizer.state_dict()
        torch.save(state, os.path.join(self.exp_dir, 'checkpoints', name))

    def _load_training_state(self, path):
        ckpt = torch_load(path, map_location='cpu')
        self.net.load_state_dict(ckpt['state_dict'])
        self.net.latent_avg.copy_(ckpt['latent_avg'].view(-1, 512).expand(self.n_styles, 512))
        if 'optimizer' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer'])
        if self.use_adv and 'discriminator' in ckpt:
            self.discriminator.load_state_dict(ckpt['discriminator'])
            self.d_optimizer.load_state_dict(ckpt['d_optimizer'])
        self.global_step = ckpt.get('global_step', 0) + 1
        print(f'Resumed from {path} at step {self.global_step}')
