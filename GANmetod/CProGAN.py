"""
Implementation of Conditional ProGAN (C-ProGAN) with key attributes from the ProGAN paper.
This implementation incorporates conditional embeddings to support class-conditional generation,
while maintaining progressive growing, minibatch standard deviation, pixel normalization,
and equalized learning rate for convolutional layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log2

# Channel scaling factors for progressive growing
factors = [1, 1, 1, 1, 1/2, 1/4, 1/8, 1/16, 1/32]


class WSConv2d(nn.Module):
    """
    Weight-scaled Conv2d with Equalized Learning Rate.
    Input scaling is used instead of weight scaling for equivalent effect.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gain=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (gain / (in_channels * (kernel_size ** 2))) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None

        # Initialize weights and biases
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)


class WSLinear(nn.Module):
    """
    Weight-scaled Linear layer with Equalized Learning Rate.
    """
    def __init__(self, in_features, out_features, gain=2):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.scale = (gain / in_features) ** 0.5
        self.bias = self.linear.bias
        self.linear.bias = None

        # Initialize weights and biases
        nn.init.normal_(self.linear.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.linear(x * self.scale) + self.bias


class PixelNorm(nn.Module):
    """
    Pixel-wise normalization layer.
    """
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-8

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)


class ConvBlock(nn.Module):
    """
    Convolutional block with two WSConv2d layers, LeakyReLU, and optional PixelNorm.
    """
    def __init__(self, in_channels, out_channels, use_pixelnorm=True):
        super().__init__()
        self.use_pn = use_pixelnorm
        self.conv1 = WSConv2d(in_channels, out_channels)
        self.conv2 = WSConv2d(out_channels, out_channels)
        self.leaky = nn.LeakyReLU(0.2)
        self.pn = PixelNorm()

    def forward(self, x):
        x = self.leaky(self.conv1(x))
        x = self.pn(x) if self.use_pn else x
        x = self.leaky(self.conv2(x))
        x = self.pn(x) if self.use_pn else x
        return x


class Generator(nn.Module):
    """
    Conditional ProGAN Generator with progressive growing and embedding support.
    """
    def __init__(self, z_dim, in_channels, img_channels=3, emb_size=768):
        super().__init__()
        # Initial block: 1x1 -> 4x4 with embedding
        self.initial = nn.Sequential(
            PixelNorm(),
            nn.ConvTranspose2d(z_dim + emb_size, in_channels, 4, 1, 0),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            PixelNorm(),
        )
        self.initial_rgb = WSConv2d(in_channels, img_channels, kernel_size=1, stride=1, padding=0)
        self.prog_blocks = nn.ModuleList([])
        self.rgb_layers = nn.ModuleList([self.initial_rgb])

        # Progressive blocks
        for i in range(len(factors) - 1):
            conv_in_c = int(in_channels * factors[i])
            conv_out_c = int(in_channels * factors[i + 1])
            self.prog_blocks.append(ConvBlock(conv_in_c, conv_out_c))
            self.rgb_layers.append(
                WSConv2d(conv_out_c, img_channels, kernel_size=1, stride=1, padding=0)
            )

    def fade_in(self, alpha, upscaled, generated):
        """
        Fade-in mechanism for smooth transition between resolutions.
        """
        return torch.tanh(alpha * generated + (1 - alpha) * upscaled)

    def forward(self, x, labels, alpha, steps):
        # Concatenate noise and embedding
        embedding = labels.unsqueeze(2).unsqueeze(3)
        x = torch.cat([x, embedding], dim=1)
        out = self.initial(x)

        if steps == 0:
            return self.initial_rgb(out)

        for step in range(steps):
            upscaled = F.interpolate(out, scale_factor=2, mode="nearest")
            out = self.prog_blocks[step](upscaled)

        final_upscaled = self.rgb_layers[steps - 1](upscaled)
        final_out = self.rgb_layers[steps](out)
        return self.fade_in(alpha, final_upscaled, final_out)


class Discriminator(nn.Module):
    """
    Conditional ProGAN Discriminator with progressive growing and embedding support.
    """
    def __init__(self, z_dim, in_channels, img_channels=3, emb_size=768):
        super().__init__()
        self.prog_blocks = nn.ModuleList([])
        self.rgb_layers = nn.ModuleList([])
        self.emb_layers = nn.ModuleList([])
        self.leaky = nn.LeakyReLU(0.2)

        # Progressive blocks in reverse order
        for i in range(len(factors) - 1, 0, -1):
            conv_in = int(in_channels * factors[i])
            conv_out = int(in_channels * factors[i - 1])
            self.prog_blocks.append(ConvBlock(conv_in, conv_out, use_pixelnorm=False))
            self.rgb_layers.append(
                WSConv2d(img_channels + 1, conv_in, kernel_size=1, stride=1, padding=0)
            )
            res = 2 ** (i + 2)
            self.emb_layers.append(WSLinear(emb_size, res * res))

        # Initial RGB layer for 4x4 input
        self.initial_rgb = WSConv2d(img_channels + 1, in_channels, kernel_size=1, stride=1, padding=0)
        self.rgb_layers.append(self.initial_rgb)
        self.emb_layers.append(WSLinear(emb_size, 4 * 4))
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # Final block for 4x4 input
        self.final_block = nn.Sequential(
            WSConv2d(in_channels + 1, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, kernel_size=4, padding=0, stride=1),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, 1, kernel_size=1, padding=0, stride=1),
        )

    def fade_in(self, alpha, downscaled, out):
        """
        Fade-in mechanism for smooth transition between resolutions.
        """
        return alpha * out + (1 - alpha) * downscaled

    def minibatch_std(self, x):
        """
        Minibatch standard deviation layer for batch diversity.
        """
        batch_statistics = torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        return torch.cat([x, batch_statistics], dim=1)

    def forward(self, x, labels, alpha, steps):
        cur_step = len(self.prog_blocks) - steps
        res = 4 * (2 ** steps)
        # Transform embedding to spatial dimensions
        embedding = self.emb_layers[cur_step](labels).view(labels.shape[0], 1, res, res)
        x_cat = torch.cat([x, embedding], dim=1)
        out = self.leaky(self.rgb_layers[cur_step](x_cat))

        if steps == 0:
            out = self.minibatch_std(out)
            return self.final_block(out).view(out.shape[0], -1)

        # Downscaled path with embedding
        embedding_low = self.avg_pool(embedding)
        x_low = self.avg_pool(x)
        x_low_cat = torch.cat([x_low, embedding_low], dim=1)
        downscaled = self.leaky(self.rgb_layers[cur_step + 1](x_low_cat))
        out = self.avg_pool(self.prog_blocks[cur_step](out))

        out = self.fade_in(alpha, downscaled, out)

        for step in range(cur_step + 1, len(self.prog_blocks)):
            out = self.prog_blocks[step](out)
            out = self.avg_pool(out)

        out = self.minibatch_std(out)
        return self.final_block(out).view(out.shape[0], -1)


def initialize_weights(model):
    """
    Initialize weights according to the ProGAN methodology.
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)