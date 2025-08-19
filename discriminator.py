import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, ch_img, feat_d, img_size, emb_size=768):
        super().__init__()
        self.img_size = img_size
        self.disc = nn.Sequential(
            nn.Conv2d(ch_img+1, feat_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            self._block(feat_d, feat_d*2, kernel_size=4, stride=2, padding=1),
            self._block(feat_d*2, feat_d*4, kernel_size=4, stride=2, padding=1),
            self._block(feat_d*4, feat_d*8, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(feat_d*8, 1, kernel_size=4, stride=2, padding=0),
        )
        self.emb = nn.Linear(emb_size, img_size*img_size)

    def _block(self, in_channel, out_channel, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channel,
                out_channel,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.InstanceNorm2d(out_channel, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x, labels):
        embedding = self.emb(labels).view(labels.shape[0], 1, self.img_size, self.img_size)
        x = torch.cat([x, embedding], dim=1) #NxCxHxW
        return self.disc(x)