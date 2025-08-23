import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.transforms.v2.functional import resize

from attention import SelfAttention

class CLIPEmbedding(nn.Module):

    def __init__(self, vocab: int, n_embed: int, n_tokens: int):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab, n_embed)
        self.position_embedding = nn.Parameter(torch.zeros(n_tokens, n_embed))

    def forward(self, tokens):
        x = self.token_embedding(tokens)

        x += self.position_embedding

        return x

class CLIPLayer(nn.Module):

    def __init__(self, n_head: int, n_embed: int):
        super().__init__()

        self.layermorm_1 = nn.LayerNorm(n_embed)
        self.attention = SelfAttention(n_head, n_embed)
        self.layermorm_2 = nn.LayerNorm(n_embed)
        self.linear_1 = nn.Linear(n_embed, 4 * n_embed)
        self.linear_2 = nn.Linear(4 * n_embed, n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        residual = x

        x = self.layermorm_1(x)

        x = self.attention(x, True)

        x += residual

        residual = x

        x = self.layermorm_2(x)

        x = self.linear_1(x)

        x = x * torch.sigmoid(1.702 * x)

        x = self.linear_2(x)

        x += residual

        return x

class CLIP(nn.Module):
    def __init__(self):
        self.embedding = CLIPEmbedding(49408, 768, 77)
        self.laters = nn.Module([
            CLIPLayer(12, 768) for i in range(12)
        ])
        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens.type(torch.long)

        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)

        out = self.layernorm(state)
        return out
