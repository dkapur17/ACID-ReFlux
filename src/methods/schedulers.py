"""
Noise schedules for diffusion models.
"""

import math
from abc import ABC, abstractmethod

import torch


class NoiseSchedule(ABC):
    """Base class for noise schedules that produce a beta sequence."""

    @abstractmethod
    def __call__(self, num_timesteps: int) -> torch.Tensor:
        """Return a 1-D tensor of betas with length ``num_timesteps``."""


class LinearSchedule(NoiseSchedule):
    def __init__(self, beta_start: float, beta_end: float):
        self.beta_start = beta_start
        self.beta_end = beta_end

    def __call__(self, num_timesteps: int) -> torch.Tensor:
        return torch.linspace(self.beta_start, self.beta_end, num_timesteps, dtype=torch.float32)


class CosineSchedule(NoiseSchedule):
    """Cosine schedule from Nichol & Dhariwal 2021."""

    def __init__(self, s: float = 0.008):
        self.s = s

    def __call__(self, num_timesteps: int) -> torch.Tensor:
        steps = num_timesteps + 1
        t = torch.linspace(0, num_timesteps, steps, dtype=torch.float32)
        alphas_cumprod = torch.cos(((t / num_timesteps) + self.s) / (1 + self.s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, min=0, max=0.999)


SCHEDULES = {
    "linear": LinearSchedule,
    "cosine": CosineSchedule,
}


def get_schedule(name: str, **kwargs) -> NoiseSchedule:
    """Look up a noise schedule by name.

    For ``"linear"`` pass ``beta_start`` and ``beta_end``.
    For ``"cosine"`` pass ``s`` (optional).
    """
    if name not in SCHEDULES:
        raise ValueError(f"Unknown noise schedule: {name}. Choose from {list(SCHEDULES.keys())}.")
    return SCHEDULES[name](**kwargs)
