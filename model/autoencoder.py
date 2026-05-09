import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.SELU(),
            nn.AvgPool3d(kernel_size=2),
            nn.Conv3d(32, 16, kernel_size=3, padding=1),
            nn.SELU(),
            nn.AvgPool3d(kernel_size=2),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv3d(16, 16, kernel_size=3, padding=1),
            nn.SELU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.SELU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv3d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.decode(self.encode(x))
