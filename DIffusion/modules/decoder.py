import torch
import torch.nn as nn
from torch.nn import functional as F
from attention import SelfAttention

class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        n, c, h , w = x.shape
        x = x.view(n, c, h * w)

        x = x.transpose(-1, -2)

        x = self.attention(x)

        x = x.transpose(-1, -2)

        x = x.view((n, c, h, w))

        x += residual

        return x

class VAE_ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.gropnorm1 = nn.GroupNorm(32, in_channel)
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)

        self.gropnorm2 = nn.GroupNorm(32, out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)

        if in_channel == out_channel:
            self.res_layer = nn.Identity()
        else:
            self.res_layer = nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        residual = x

        x = self.gropnorm1(x)
        x = F.silu(x)
        x = self.conv1(x)

        x = self.gropnorm2(x)
        x = F.silu(x)
        x = self.conv2(x)

        return x + self.res_layer(residual)

class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),

            nn.Conv2d(4, 512, kernel_size=3, padding=1),

            VAE_ResBlock(512, 512),

            VAE_AttentionBlock(512),

            VAE_ResBlock(512, 512),
            VAE_ResBlock(512, 512),
            VAE_ResBlock(512, 512),
            VAE_ResBlock(512, 512),

            nn.Upsample(scale_factor=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            VAE_ResBlock(512, 512),
            VAE_ResBlock(512, 512),
            VAE_ResBlock(512, 512),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAE_ResBlock(512, 256),
            VAE_ResBlock(256, 256),
            VAE_ResBlock(256, 256),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            VAE_ResBlock(512, 128),
            VAE_ResBlock(128, 128),
            VAE_ResBlock(128, 128),

            nn.GroupNorm(32, 128),

            nn.SiLU(),

            nn.Conv2d(128, 3, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x /= 0.18215

        for module in self:
            x  = module(x)

        return x
