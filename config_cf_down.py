"""Legacy fixed normalization constants only; importing creates no directories."""
import torch

_INPUTS = {
    'yplus_15': (0.377375, -0.000370, 0.118401, 0.008552),
    'yplus_30': (0.576315, -0.000798, 0.149083, 0.019150),
    'yplus_50': (0.712458, -0.000973, 0.149702, 0.031049),
    'yplus_100': (0.842697, -0.000849, 0.124568, 0.047473),
}

def get_normalization(yplus):
    u, v, su, sv = _INPUTS[yplus]
    mean, std = 0.006121, 0.002552
    return (
        torch.tensor([u, v, mean]).view(3, 1, 1),
        torch.tensor([su, sv, std]).view(3, 1, 1),
        torch.tensor([mean]).view(1, 1, 1),
        torch.tensor([std]).view(1, 1, 1),
    )
