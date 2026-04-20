"""
Observation augmentations for sim-to-real transfer.

Applied to replay buffer samples during world-model training (DrQ-style):
encoder sees augmented views → learns invariant representations.
Live rollout observations are NOT augmented.

Input/output: float32 tensor [T, B, C, H, W] in range [-0.5, 0.5].

Augmentations (all randomised per batch):
  - Random brightness & contrast  (lighting fluctuations)
  - Synthetic shadow               (random dark horizontal band)
  - Gaussian blur                  (motion blur / cheap optics)
  - Gaussian noise                 (camera sensor noise)
  - Random gamma                   (exposure variation / auto-exposure)
  - Random erasing / cutout        (occlusions, insects on lens)
  - Random crop + resize           (camera vibration, minor FOV shift)

No horizontal flip (would require flipping steering labels).
No rotation (distorts road geometry the policy relies on).

All ops are fully vectorised — no Python loops over samples.
"""

import torch
import torch.nn.functional as F


class Augmenter:
    def __init__(
        self,
        brightness,
        contrast,
        shadow_prob,
        shadow_intensity,
        blur_prob,
        blur_kernel,
        noise_std,
        gamma_range,
        erase_prob,
        erase_max_frac,
        crop_frac,
        device,
    ):
        self.brightness       = brightness
        self.contrast         = contrast
        self.shadow_prob      = shadow_prob
        self.shadow_intensity = shadow_intensity
        self.blur_prob        = blur_prob
        self.blur_kernel      = blur_kernel
        self.noise_std        = noise_std
        self.gamma_range      = gamma_range
        self.erase_prob       = erase_prob
        self.erase_max_frac   = erase_max_frac
        self.crop_frac        = crop_frac
        self.device           = device

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: [T, B, C, H, W] float32 in [-0.5, 0.5]"""
        T, B, C, H, W = obs.shape
        x = obs.reshape(T * B, C, H, W)

        x = self._brightness_contrast(x)
        x = self._gamma(x)
        x = self._gaussian_noise(x)
        x = self._blur(x)
        x = self._shadow(x, H, W)
        x = self._random_erase(x, H, W)
        x = self._random_crop(x, H, W)

        return x.reshape(T, B, C, H, W)

    # ------------------------------------------------------------------

    def _brightness_contrast(self, x):
        N = x.shape[0]
        b = (torch.rand(N, 1, 1, 1, device=x.device) * 2 - 1) * self.brightness
        c = 1.0 + (torch.rand(N, 1, 1, 1, device=x.device) * 2 - 1) * self.contrast
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        return ((x - mean) * c + mean + b).clamp(-0.5, 0.5)

    def _gamma(self, x):
        N = x.shape[0]
        lo, hi = self.gamma_range
        gamma = lo + torch.rand(N, 1, 1, 1, device=x.device) * (hi - lo)
        return (x + 0.5).clamp(0.0, 1.0).pow(gamma) - 0.5

    def _gaussian_noise(self, x):
        if self.noise_std <= 0:
            return x
        return (x + torch.randn_like(x) * self.noise_std).clamp(-0.5, 0.5)

    def _blur(self, x):
        if self.blur_prob <= 0:
            return x
        N, C, H, W = x.shape
        apply = torch.rand(N, device=x.device) < self.blur_prob
        if not apply.any():
            return x
        k = self.blur_kernel
        sigma = k / 6.0
        coords = torch.arange(k, dtype=x.dtype, device=x.device) - k // 2
        g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        g = g / g.sum()
        kernel = (g.unsqueeze(0) * g.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
        kernel = kernel.expand(C, 1, k, k)
        blurred = F.conv2d(x, kernel, padding=k // 2, groups=C)
        mask = apply.view(N, 1, 1, 1).expand_as(x)
        return torch.where(mask, blurred, x)

    def _shadow(self, x, H, W):
        """Vectorised horizontal-band shadow — no Python loop."""
        if self.shadow_prob <= 0:
            return x
        N = x.shape[0]
        apply = torch.rand(N, device=x.device) < self.shadow_prob
        if not apply.any():
            return x
        # Random band bounds, generated on CPU to avoid MPS randint quirks
        y0 = torch.randint(0, max(1, H // 2), (N,)).to(x.device)
        y1 = torch.randint(H // 2, H,         (N,)).to(x.device)
        rows = torch.arange(H, device=x.device).unsqueeze(0)   # [1, H]
        in_band = (rows >= y0.unsqueeze(1)) & (rows < y1.unsqueeze(1))  # [N, H]
        in_band = in_band & apply.unsqueeze(1)                  # only selected samples
        scale = torch.where(
            in_band.unsqueeze(1).unsqueeze(-1),                 # [N,1,H,1]
            torch.full_like(x[:, :, :, :1], self.shadow_intensity),
            torch.ones_like(x[:, :, :, :1]),
        )
        return (x * scale).clamp(-0.5, 0.5)

    def _random_erase(self, x, H, W):
        """Vectorised erase via additive mask — no Python loop."""
        if self.erase_prob <= 0:
            return x
        N, C, _, _ = x.shape
        apply = (torch.rand(N, device=x.device) < self.erase_prob)
        if not apply.any():
            return x
        max_h = max(1, int(H * (self.erase_max_frac ** 0.5)))
        max_w = max(1, int(W * (self.erase_max_frac ** 0.5)))
        eh = torch.randint(1, max_h + 1, (N,)).to(x.device)
        ew = torch.randint(1, max_w + 1, (N,)).to(x.device)
        y0 = (torch.rand(N, device=x.device) * (H - eh.float())).long().clamp(0, H - 1)
        x0 = (torch.rand(N, device=x.device) * (W - ew.float())).long().clamp(0, W - 1)
        # Build erase mask [N, H, W] via coordinate comparison
        rows = torch.arange(H, device=x.device).view(1, H, 1)
        cols = torch.arange(W, device=x.device).view(1, 1, W)
        erase = (
            (rows >= y0.view(N, 1, 1)) & (rows < (y0 + eh).view(N, 1, 1)) &
            (cols >= x0.view(N, 1, 1)) & (cols < (x0 + ew).view(N, 1, 1)) &
            apply.view(N, 1, 1)
        )                                                         # [N, H, W]
        erase = erase.unsqueeze(1).expand_as(x)
        return x.masked_fill(erase, 0.0)

    def _random_crop(self, x, H, W):
        """Vectorised crop+resize via affine_grid — no Python loop."""
        if self.crop_frac >= 1.0:
            return x
        N = x.shape[0]
        ch = int(H * self.crop_frac)
        cw = int(W * self.crop_frac)
        y0 = torch.randint(0, H - ch + 1, (N,)).to(dtype=x.dtype, device=x.device)
        x0 = torch.randint(0, W - cw + 1, (N,)).to(dtype=x.dtype, device=x.device)
        # Affine theta maps output [-1,1] grid → input crop region
        # scale: cw/W, ch/H — translate: 2*x0/W + cw/W - 1, 2*y0/H + ch/H - 1
        sx = cw / W;  tx = 2.0 * x0 / W + sx - 1.0
        sy = ch / H;  ty = 2.0 * y0 / H + sy - 1.0
        theta = torch.zeros(N, 2, 3, device=x.device, dtype=x.dtype)
        theta[:, 0, 0] = sx
        theta[:, 1, 1] = sy
        theta[:, 0, 2] = tx
        theta[:, 1, 2] = ty
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        return F.grid_sample(x, grid, mode='bilinear', align_corners=False,
                             padding_mode='border')
