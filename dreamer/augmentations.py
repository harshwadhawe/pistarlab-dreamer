"""
Observation augmentations for sim-to-real transfer.

Applied to replay buffer samples during world-model training (DrQ-style):
encoder sees augmented views → learns invariant representations.
Live rollout observations are NOT augmented.

Input/output: float32 tensor [T, B, C, H, W] in range [-0.5, 0.5].

Augmentations (all randomised per batch):
  - Random brightness & contrast  (lighting fluctuations)
  - Synthetic shadow               (random dark polygon)
  - Gaussian blur                  (motion blur / cheap optics)
  - Gaussian noise                 (camera sensor noise)
  - Random gamma                   (exposure variation / auto-exposure)
  - Random erasing / cutout        (occlusions, insects on lens)
  - Random crop + resize           (camera vibration, minor FOV shift)

No horizontal flip (would require flipping steering labels).
No rotation (distorts road geometry the policy relies on).
"""

import torch
import torch.nn.functional as F


class Augmenter:
    def __init__(
        self,
        brightness=0.2,       # ± brightness shift
        contrast=0.2,         # ± contrast scale around 1.0
        shadow_prob=0.5,      # probability of adding a shadow each sample
        shadow_intensity=0.4, # how dark the shadow is (0=black, 1=no effect)
        blur_prob=0.3,        # probability of Gaussian blur
        blur_kernel=5,        # kernel size (must be odd)
        noise_std=0.02,       # Gaussian noise std (in [-0.5,0.5] space)
        gamma_range=(0.7, 1.4),  # random gamma exponent range
        erase_prob=0.2,       # probability of random erasing
        erase_max_frac=0.2,   # max fraction of image area to erase
        crop_frac=0.9,        # random crop to this fraction then resize back
        device='cpu',
    ):
        self.brightness     = brightness
        self.contrast       = contrast
        self.shadow_prob    = shadow_prob
        self.shadow_intensity = shadow_intensity
        self.blur_prob      = blur_prob
        self.blur_kernel    = blur_kernel
        self.noise_std      = noise_std
        self.gamma_range    = gamma_range
        self.erase_prob     = erase_prob
        self.erase_max_frac = erase_max_frac
        self.crop_frac      = crop_frac
        self.device         = device

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: [T, B, C, H, W] float32 in [-0.5, 0.5]
        Returns augmented tensor, same shape and range.
        """
        T, B, C, H, W = obs.shape
        # Flatten time+batch for per-sample ops, then restore
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
        # Brightness: uniform shift per sample
        b = (torch.rand(N, 1, 1, 1, device=x.device) * 2 - 1) * self.brightness
        # Contrast: scale around mean per sample
        c = 1.0 + (torch.rand(N, 1, 1, 1, device=x.device) * 2 - 1) * self.contrast
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) * c + mean + b
        return x.clamp(-0.5, 0.5)

    def _gamma(self, x):
        # Shift to [0,1], apply gamma, shift back
        N = x.shape[0]
        lo, hi = self.gamma_range
        gamma = lo + torch.rand(N, 1, 1, 1, device=x.device) * (hi - lo)
        x01 = (x + 0.5).clamp(0.0, 1.0)
        x01 = x01.pow(gamma)
        return x01 - 0.5

    def _gaussian_noise(self, x):
        if self.noise_std <= 0:
            return x
        noise = torch.randn_like(x) * self.noise_std
        return (x + noise).clamp(-0.5, 0.5)

    def _blur(self, x):
        if self.blur_prob <= 0:
            return x
        N, C, H, W = x.shape
        mask = torch.rand(N, device=x.device) < self.blur_prob
        if not mask.any():
            return x
        k = self.blur_kernel
        # Build separable Gaussian kernel
        sigma = k / 6.0
        coords = torch.arange(k, dtype=x.dtype, device=x.device) - k // 2
        g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        g = g / g.sum()
        kernel = (g.unsqueeze(0) * g.unsqueeze(1))          # [k, k]
        kernel = kernel.unsqueeze(0).unsqueeze(0)            # [1, 1, k, k]
        kernel = kernel.expand(C, 1, k, k)                  # [C, 1, k, k]
        pad = k // 2
        blurred = F.conv2d(x, kernel, padding=pad, groups=C)
        idx = mask.nonzero(as_tuple=True)[0]
        x = x.clone()
        x[idx] = blurred[idx]
        return x

    def _shadow(self, x, H, W):
        if self.shadow_prob <= 0:
            return x
        N, C, _, _ = x.shape
        mask = torch.rand(N, device=x.device) < self.shadow_prob
        if not mask.any():
            return x
        x = x.clone()
        idx = mask.nonzero(as_tuple=True)[0]
        for i in idx:
            # Random horizontal band (simulate overhead shadow)
            y0 = torch.randint(0, H // 2, (1,)).item()
            y1 = torch.randint(H // 2, H, (1,)).item()
            x[i, :, y0:y1, :] *= self.shadow_intensity
        return x.clamp(-0.5, 0.5)

    def _random_erase(self, x, H, W):
        if self.erase_prob <= 0:
            return x
        N, C, _, _ = x.shape
        mask = torch.rand(N, device=x.device) < self.erase_prob
        if not mask.any():
            return x
        x = x.clone()
        max_area = int(H * W * self.erase_max_frac)
        idx = mask.nonzero(as_tuple=True)[0]
        for i in idx:
            area = torch.randint(1, max_area + 1, (1,)).item()
            eh = torch.randint(1, H, (1,)).item()
            ew = max(1, area // eh)
            ew = min(ew, W)
            y0 = torch.randint(0, max(1, H - eh), (1,)).item()
            x0 = torch.randint(0, max(1, W - ew), (1,)).item()
            x[i, :, y0:y0 + eh, x0:x0 + ew] = 0.0  # fill with mean (0 = 0.5 grey)
        return x

    def _random_crop(self, x, H, W):
        if self.crop_frac >= 1.0:
            return x
        N, C, _, _ = x.shape
        ch = int(H * self.crop_frac)
        cw = int(W * self.crop_frac)
        # Independent crop offset per sample
        y0s = torch.randint(0, H - ch + 1, (N,))
        x0s = torch.randint(0, W - cw + 1, (N,))
        out = torch.empty(N, C, ch, cw, dtype=x.dtype, device=x.device)
        for i in range(N):
            out[i] = x[i, :, y0s[i]:y0s[i] + ch, x0s[i]:x0s[i] + cw]
        # Resize back to original H×W
        return F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
