import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    """
    β-VAE for DonkeyCar images.

    Encoder spatial trace matches VisualEncoder: (C,64,64) → 31→14→6→2 → flat 1024 → 2*z_dim
    Decoder spatial trace matches VisualObservationModel: z → 512 → 1×1 → 5→13→30→64
    """

    def __init__(self, channels=3, z_dim=128):
        super().__init__()
        self.z_dim = z_dim

        self.enc_conv1 = nn.Conv2d(channels, 32,  4, stride=2)
        self.enc_conv2 = nn.Conv2d(32,        64,  4, stride=2)
        self.enc_conv3 = nn.Conv2d(64,       128,  4, stride=2)
        self.enc_conv4 = nn.Conv2d(128,      256,  4, stride=2)
        self.enc_fc    = nn.Linear(1024, 2 * z_dim)

        self.dec_fc    = nn.Linear(z_dim, 512)
        self.dec_conv1 = nn.ConvTranspose2d(512, 128, 5, stride=2)
        self.dec_conv2 = nn.ConvTranspose2d(128,  64, 5, stride=2)
        self.dec_conv3 = nn.ConvTranspose2d( 64,  32, 6, stride=2)
        self.dec_conv4 = nn.ConvTranspose2d( 32, channels, 6, stride=2)

    def encode(self, obs):
        """obs: (B,C,64,64) float32 [-0.5,0.5] → mean, log_var: (B,z_dim)"""
        h = F.relu(self.enc_conv1(obs))
        h = F.relu(self.enc_conv2(h))
        h = F.relu(self.enc_conv3(h))
        h = F.relu(self.enc_conv4(h))
        mean, log_var = self.enc_fc(h.view(h.size(0), -1)).chunk(2, dim=-1)
        return mean, log_var

    def decode(self, z):
        """z: (B,z_dim) → (B,C,64,64)"""
        h = self.dec_fc(z).view(z.size(0), 512, 1, 1)
        h = F.relu(self.dec_conv1(h))
        h = F.relu(self.dec_conv2(h))
        h = F.relu(self.dec_conv3(h))
        return self.dec_conv4(h)

    def forward(self, obs):
        mean, log_var = self.encode(obs)
        z = mean + (0.5 * log_var).exp() * torch.randn_like(mean)
        return self.decode(z), mean, log_var

    @staticmethod
    def loss(recon, obs, mean, log_var, beta=1.0):
        recon_loss = F.mse_loss(recon, obs, reduction='sum') / obs.size(0)
        kl         = -0.5 * (1 + log_var - mean.pow(2) - log_var.exp()).sum(dim=1).mean()
        return recon_loss + beta * kl, recon_loss.item(), kl.item()
