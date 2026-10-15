"""Five-downsample U-Net with timestep residual blocks and bottleneck attention."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def norm(channels):
    return nn.GroupNorm(math.gcd(channels, 32), channels)


class TimeBlock(nn.Module):
    def __init__(self, incoming, outgoing, time_dim):
        super().__init__()
        self.first = nn.Sequential(norm(incoming), nn.SiLU(), nn.Conv2d(incoming, outgoing, 3, padding=1))
        self.time = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, outgoing * 2))
        self.norm = norm(outgoing)
        self.last = nn.Sequential(nn.SiLU(), nn.Conv2d(outgoing, outgoing, 3, padding=1))
        self.skip = nn.Conv2d(incoming, outgoing, 1) if incoming != outgoing else nn.Identity()

    def forward(self, x, time):
        h = self.first(x)
        scale, shift = self.time(time)[:, :, None, None].chunk(2, dim=1)
        h = self.norm(h) * (1 + scale) + shift
        return self.skip(x) + self.last(h)


class Attention(nn.Module):
    def __init__(self, channels, heads):
        super().__init__()
        if channels % heads:
            raise ValueError('Bottleneck channels must be divisible by attention heads')
        self.heads = heads
        self.norm = norm(channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.ff = nn.Sequential(norm(channels), nn.Conv2d(channels, 4 * channels, 1),
                                nn.GELU(), nn.Conv2d(4 * channels, channels, 1))

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w).unbind(1)
        y = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
        x = x + self.proj(y.transpose(-1, -2).reshape(b, c, h, w))
        return x + self.ff(x)


class ConditionalUNet(nn.Module):
    """Inputs: noisy q plus [u,w,mask,masked standardized observations]."""
    def __init__(self, base_channels=32, attention_depth=2, heads=8,
                 target_channels=1, condition_channels=4):
        super().__init__()
        if base_channels < 4 or base_channels % 2 or attention_depth < 1:
            raise ValueError('Use even base_channels >= 4 and attention_depth >= 1')
        self.base_channels = base_channels
        if target_channels < 1 or condition_channels < 1:
            raise ValueError('Channel counts must be positive')
        self.target_channels, self.condition_channels = target_channels, condition_channels
        time_dim = base_channels * 4
        self.time_mlp = nn.Sequential(nn.Linear(base_channels, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        channels = [base_channels * m for m in (1, 2, 4, 8, 8, 8)]
        self.stem = nn.Conv2d(target_channels + condition_channels, channels[0], 3, padding=1)
        self.down = nn.ModuleList([nn.Conv2d(channels[i], channels[i+1], 3, stride=2, padding=1) for i in range(5)])
        self.encoder = nn.ModuleList([TimeBlock(c, c, time_dim) for c in channels[1:]])
        self.attention = nn.Sequential(*[Attention(channels[-1], heads) for _ in range(attention_depth)])
        self.decoder = nn.ModuleList([TimeBlock(channels[i+1] + channels[i], channels[i], time_dim) for i in reversed(range(5))])
        self.output = nn.Sequential(norm(channels[0]), nn.SiLU(), nn.Conv2d(channels[0], target_channels, 3, padding=1))

    def forward(self, noisy, timestep, condition):
        if noisy.ndim != 4 or noisy.shape[1] != self.target_channels or condition.shape != (noisy.shape[0], self.condition_channels, *noisy.shape[-2:]):
            raise ValueError('Noisy target or condition channel/shape mismatch')
        half = self.base_channels // 2
        freq = torch.exp(-math.log(10000) * torch.arange(half, device=noisy.device) / max(half-1, 1))
        angle = timestep.float()[:, None] * freq[None]
        time = self.time_mlp(torch.cat([angle.sin(), angle.cos()], dim=1))
        height, width = noisy.shape[-2:]
        x = F.pad(torch.cat([noisy, condition], dim=1), (0, (-width) % 32, 0, (-height) % 32))
        x = self.stem(x)
        skips = [x]
        for down, block in zip(self.down, self.encoder):
            x = block(down(x), time)
            skips.append(x)
        x = self.attention(x)
        for block, skip in zip(self.decoder, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = block(torch.cat([x, skip], dim=1), time)
        return self.output(x)[..., :height, :width]
