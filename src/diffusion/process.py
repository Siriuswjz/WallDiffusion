"""Conditional diffusion with epsilon/v objectives and deterministic DDIM sampling."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class GaussianDiffusion(nn.Module):
    # 通过一个余弦数学公式，提前算好了在 1000 步的扩散过程中，每一步应该往图片里加多少噪声、保留多少信号，
    # 并把它们存进 self.alpha_bar 里，供后面加噪和去噪时随时查表调用。
    def __init__(self, timesteps=1000, prediction_type='epsilon'):
        super().__init__()
        if timesteps < 2:
            raise ValueError('timesteps must be >= 2')
        if prediction_type not in ('epsilon', 'v'):
            raise ValueError('prediction_type must be epsilon or v')
        self.prediction_type = prediction_type
        steps = torch.arange(timesteps + 1, dtype=torch.float64)
        cumulative = torch.cos((steps / timesteps + .008) / 1.008 * math.pi / 2).square()
        cumulative = cumulative / cumulative[0]
        betas = (1 - cumulative[1:] / cumulative[:-1]).clamp(max=.999)
        self.register_buffer('alpha_bar', (1 - betas).cumprod(0).float())
        self.timesteps = timesteps

    def add_noise(self, clean, timestep, noise):
        alpha = self.alpha_bar[timestep][:, None, None, None]
        return alpha.sqrt() * clean + (1 - alpha).sqrt() * noise

    def loss(self, model, clean, condition):
        timestep = torch.randint(self.timesteps, (len(clean),), device=clean.device)
        noise = torch.randn_like(clean)
        predicted = model(self.add_noise(clean, timestep, noise), timestep, condition)
        alpha = self.alpha_bar[timestep][:, None, None, None]
        target = noise if self.prediction_type == 'epsilon' else alpha.sqrt()*noise-(1-alpha).sqrt()*clean
        return F.mse_loss(predicted, target)

    @torch.no_grad()
    def sample(self, model, condition, steps=50, seed=42, enforce_observations=False):
        if not 1 <= steps <= self.timesteps:
            raise ValueError('sampling steps must be in [1, timesteps]')
        generator = torch.Generator(device=condition.device).manual_seed(seed)
        channels = getattr(model, 'target_channels', 1)
        if enforce_observations and (channels != 1 or condition.shape[1] != 4):
            raise ValueError('Hard observation projection requires a single-target mask/observation condition')
        noisy = torch.randn((len(condition), channels, *condition.shape[-2:]), device=condition.device, generator=generator)
        schedule = torch.linspace(self.timesteps-1, 0, steps, device=condition.device).long().tolist()
        for index, step in enumerate(schedule):
            timestep = torch.full((len(condition),), step, device=condition.device, dtype=torch.long)
            predicted = model(noisy, timestep, condition)
            alpha = self.alpha_bar[step]
            if self.prediction_type == 'v':
                clean = alpha.sqrt()*noisy-(1-alpha).sqrt()*predicted
                epsilon = (1-alpha).sqrt()*noisy+alpha.sqrt()*predicted
            else:
                epsilon = predicted
                clean = (noisy - (1-alpha).sqrt() * epsilon) / alpha.sqrt()
            if enforce_observations:
                mask, observations = condition[:, 2:3], condition[:, 3:4]
                clean = clean * (1-mask) + observations
            if index == len(schedule)-1:
                noisy = clean
            else:
                next_alpha = self.alpha_bar[schedule[index+1]]
                noisy = next_alpha.sqrt() * clean + (1-next_alpha).sqrt() * epsilon
        return noisy
