import torch
import torch.nn as nn
from networkx.algorithms.shortest_paths import weighted
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_model: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        self.in_proj = nn.Linear(d_model, 3*d_model, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

    def forward(self, x: torch.Tensor, casual_mask) -> torch.Tensor:

        input_shape = x.shape
        batch_size, seq_len, d_model = input_shape

        interwim_shape = (batch_size, seq_len, self.n_heads, self.d_head)

        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        q = q.view(interwim_shape).transpose(1, 2)
        k = k.view(interwim_shape).transpose(1, 2)
        v = v.view(interwim_shape).transpose(1, 2)

        weight = q @ k.transpose(-1, -2)

        if casual_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)

        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)

        out = weight @ v

        out = out.transpose(1, 2).reshape(input_shape)

        out = self.out_proj(out)

        return out