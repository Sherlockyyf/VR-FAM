"""Image-space metrics used in the papers: L2, LPIPS, (MS-)SSIM, identity similarity and FID."""
import torch
import torch.nn.functional as F

from .models.encoders import Backbone
from .io import torch_load

_MS_SSIM_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)


def _gaussian_window(size=11, sigma=1.5, channels=3, device='cpu'):
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)
    window = (g.t() @ g).expand(channels, 1, size, size).contiguous()
    return window


def _ssim_components(x, y, window, data_range=1.0):
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    channels = x.shape[1]
    mu_x = F.conv2d(x, window, groups=channels)
    mu_y = F.conv2d(y, window, groups=channels)
    sigma_x = F.conv2d(x * x, window, groups=channels) - mu_x ** 2
    sigma_y = F.conv2d(y * y, window, groups=channels) - mu_y ** 2
    sigma_xy = F.conv2d(x * y, window, groups=channels) - mu_x * mu_y
    cs = (2 * sigma_xy + c2) / (sigma_x + sigma_y + c2)
    ssim_map = ((2 * mu_x * mu_y + c1) / (mu_x ** 2 + mu_y ** 2 + c1)) * cs
    return ssim_map.flatten(1).mean(1), cs.flatten(1).mean(1)


def ssim(x, y):
    """SSIM between images in [0, 1] (Gaussian window 11x11, sigma 1.5). Returns [B]."""
    window = _gaussian_window(channels=x.shape[1], device=x.device)
    return _ssim_components(x, y, window)[0]


def ms_ssim(x, y):
    """Multi-scale SSIM (Wang et al., 2003) between images in [0, 1]. Returns [B]."""
    window = _gaussian_window(channels=x.shape[1], device=x.device)
    weights = torch.tensor(_MS_SSIM_WEIGHTS, device=x.device)
    values = []
    for i in range(len(weights)):
        s, cs = _ssim_components(x, y, window)
        values.append(torch.relu(s if i == len(weights) - 1 else cs))
        if i < len(weights) - 1:
            x, y = F.avg_pool2d(x, 2), F.avg_pool2d(y, 2)
    values = torch.stack(values, dim=1)
    return torch.prod(values ** weights, dim=1)


class IdentitySimilarity(torch.nn.Module):
    """Cosine similarity of face-recognition embeddings.

    ``model='ir_se50'``: ArcFace IR-SE-50 (``model_ir_se50.pth``, as in pSp / e4e).
    ``model='curricular'``: CurricularFace IR-101 (``CurricularFace_Backbone.pth``, as in HyperStyle).
    Inputs are FFHQ-aligned faces in [-1, 1].
    """

    def __init__(self, weights, model='ir_se50'):
        super().__init__()
        if model == 'ir_se50':
            self.net = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        elif model == 'curricular':
            self.net = Backbone(input_size=112, num_layers=100, drop_ratio=0.4, mode='ir', affine=False)
        else:
            raise ValueError(model)
        self.net.load_state_dict(torch_load(weights, map_location='cpu'))
        self.net.eval()

    @torch.no_grad()
    def embed(self, x):
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        x = F.adaptive_avg_pool2d(x[:, :, 35:223, 32:220], (112, 112))
        return self.net(x)

    @torch.no_grad()
    def forward(self, x, y):
        return (self.embed(x) * self.embed(y)).sum(1)


def fid_between_folders(dir_a, dir_b, device='cuda', batch_size=50):
    try:
        from pytorch_fid.fid_score import calculate_fid_given_paths
    except ImportError as err:
        raise ImportError('FID needs `pip install pytorch-fid`') from err
    return calculate_fid_given_paths([dir_a, dir_b], batch_size, str(device), 2048)
