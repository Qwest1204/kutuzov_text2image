import torch
import torch.nn as nn

def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, feat_g, img_size, emb_size=768):
        super().__init__()
        self.img_size = img_size
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(z_dim+emb_size, feat_g*16, kernel_size=4, stride=1, padding=0),
            self._block(feat_g*16, feat_g*8, kernel_size=4, stride=2, padding=1),
            self._block(feat_g*8, feat_g*4, kernel_size=4, stride=2, padding=1),
            self._block(feat_g*4, feat_g*2, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(feat_g*2, channels_img, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def _block(self, in_channel, out_channel, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channel,
                out_channel,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )
    def forward(self, x, labels):
        embedding = labels.unsqueeze(2).unsqueeze(3)
        x = torch.cat([x, embedding], dim=1)
        return self.gen(x)