"""
Denoising Diffusion Probabilistic Models (DDPM)
"""

from typing import Dict, Tuple, Optional, Literal, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseMethod
from .schedulers import get_schedule


class DDPM(BaseMethod):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_timesteps: int,
        beta_start: float,
        beta_end: float,
        noise_schedule: str = "linear",
        prediction_target: str = "epsilon",
    ):
        super().__init__(model, device)

        self.num_timesteps = int(num_timesteps)
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.noise_schedule = noise_schedule
        self.prediction_target = prediction_target

        # Validate prediction target
        if prediction_target not in ["epsilon", "x0"]:
            raise ValueError(f"Unknown prediction target: {prediction_target}. Use 'epsilon' or 'x0'.")

        # Create beta schedule based on the specified type
        schedule_kwargs = {"beta_start": beta_start, "beta_end": beta_end} if noise_schedule == "linear" else {}
        betas = get_schedule(noise_schedule, device=device, **schedule_kwargs)(num_timesteps)

        # Precompute useful values for diffusion process
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Register as buffers (will be moved to device with model)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # Calculations for forward diffusion q(x_t | x_0)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        # Calculations for reverse diffusion (denoising)
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0))

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)

        # Log calculation for clipped posterior variance (for numerical stability)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20))
        )

        # Coefficients for posterior mean
        self.register_buffer(
            "posterior_mean_coef1",
            (betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)).float()
        )
        self.register_buffer(
            "posterior_mean_coef2",
            ((1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)).float()
        )

        # Move all buffers to the correct device
        self.to(device)

    # =========================================================================
    # Helper functions
    # =========================================================================

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Extract values from a at indices t and reshape for broadcasting with x.

        Args:
            a: Tensor to extract from (e.g., alphas_cumprod)
            t: Timestep indices
            x_shape: Shape of x for proper broadcasting

        Returns:
            Extracted values with shape compatible for broadcasting
        """

        batch_size = t.shape[0]
        out = a.gather(-1, t)
        # Reshape to (batch_size, 1, 1, 1) for broadcasting with images
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    # =========================================================================
    # Forward process
    # =========================================================================

    def forward_process(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Implement the forward (noise adding) process of DDPM: q(x_t | x_0)

        Args:
            x_0: Clean images of shape (batch_size, channels, height, width)
            t: Timesteps of shape (batch_size,)
            noise: Optional pre-sampled noise. If None, sample from N(0, I)

        Returns:
            x_t: Noisy images at timestep t
            noise: The noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        # Extract coefficients for the given timesteps
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)

        # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

        return x_t, noise

    # =========================================================================
    # Training loss
    # =========================================================================

    def compute_loss(self, x_0: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Implement DDPM loss function: simplified objective from Ho et al. 2020

        Args:
            x_0: Clean data samples of shape (batch_size, channels, height, width)
            **kwargs: Additional method-specific arguments

        Returns:
            loss: Scalar loss tensor for backpropagation
            metrics: Dictionary of metrics for logging (e.g., {'mse': 0.1})
        """
        batch_size = x_0.shape[0]
        device = x_0.device


        # Sample random timesteps for each image in the batch
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device, dtype=torch.long)

        # Sample noise
        noise = torch.randn_like(x_0)

        # Get noisy images using forward process
        x_t, _ = self.forward_process(x_0, t, noise)

        # Get model prediction
        model_output = self.model(x_t, t)

        # Compute loss based on prediction target
        if self.prediction_target == "epsilon":
            # Predict noise (standard DDPM)
            target = noise
            prediction_name = "epsilon"
        elif self.prediction_target == "x0":
            # Predict clean image directly
            target = x_0
            prediction_name = "x0"
        else:
            raise ValueError(f"Unknown prediction target: {self.prediction_target}")

        # Compute MSE loss between prediction and target
        loss = F.mse_loss(model_output, target)

        # Track metrics
        metrics = {
            'loss': loss.item(),
            'mse': loss.item(),  # For logging purposes
            f'mse_{prediction_name}': loss.item(),
        }

        return loss, metrics

    # =========================================================================
    # Reverse process (sampling)
    # =========================================================================
    
    def _predict_x0(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:

        sqrt_alpha_bar = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alpha_bar = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return (x_t - sqrt_one_minus_alpha_bar * noise) / sqrt_alpha_bar

    def _posterior_mean(self, x_0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:

        c1 = self._extract(self.posterior_mean_coef1, t, x_t.shape)
        c2 = self._extract(self.posterior_mean_coef2, t, x_t.shape)
        return c1 * x_0 + c2 * x_t

    @torch.no_grad()
    def reverse_process(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Implement one step of the DDPM reverse process (Algorithm 2 from Ho et al. 2020)

        Args:
            x_t: Noisy samples at time t (batch_size, channels, height, width)
            t: the timestep (batch_size,)

        Returns:
            x_prev: Noisy samples at time t-1 (batch_size, channels, height, width)
        """

        # Get model prediction
        model_output = self.model(x_t, t)

        # Derive x_0 prediction based on what the model predicts
        if self.prediction_target == "epsilon":
            # Model predicts noise, derive x_0 from it
            x_0_pred = self._predict_x0(x_t, t, model_output)
        elif self.prediction_target == "x0":
            # Model directly predicts x_0
            x_0_pred = model_output
        else:
            raise ValueError(f"Unknown prediction target: {self.prediction_target}")

        # Clamp x_0 prediction to valid range
        x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)

        # Compute posterior mean using predicted x_0
        posterior_mean = self._posterior_mean(x_0_pred, x_t, t)
        posterior_var = self._extract(self.posterior_variance, t, x_t.shape)

        # Sample noise for stochastic sampling
        noise = torch.randn_like(x_t)

        # Don't add noise at t=0 (final step)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))

        # Compute x_{t-1}
        x_prev = posterior_mean + nonzero_mask * torch.sqrt(posterior_var) * noise
        return x_prev

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        num_steps: Optional[int] = None,
        return_trajectory: bool = False,
        show_progress: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """
        Implement DDPM sampling loop (Algorithm 2 from Ho et al. 2020)
        Start from pure noise, iterate through all timesteps using reverse_process()

        Args:
            batch_size: Number of samples to generate
            image_shape: Shape of each image (channels, height, width)
            num_steps: Number of sampling steps (default: self.num_timesteps)
            **kwargs: Additional method-specific arguments

        Returns:
            samples: Generated samples of shape (batch_size, *image_shape)
        """
        self.eval_mode()

        # Use all timesteps by default
        if num_steps is None:
            num_steps = self.num_timesteps

        # Start from pure noise (x_T ~ N(0, I))
        x = torch.randn(batch_size, *image_shape, device=self.device)

        # If using fewer steps than training, create a subset of timesteps
        if num_steps < self.num_timesteps:
            # Use evenly spaced timesteps
            timesteps = torch.linspace(self.num_timesteps - 1, 0, num_steps, dtype=torch.long, device=self.device)
        else:
            # Use all timesteps in reverse order
            timesteps = torch.arange(self.num_timesteps - 1, -1, -1, dtype=torch.long, device=self.device)

        trajectory = []
        # Iteratively denoise
        pbar = enumerate(timesteps)
        if show_progress:
            pbar = __import__('tqdm').tqdm(pbar)
        
        for i, t in pbar:
            # Create batch of timesteps
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

            # One step of reverse diffusion
            x = self.reverse_process(x, t_batch)
            
            if return_trajectory:
                trajectory.append(torch.clamp(-1,))

        if return_trajectory:
            return x, trajectory
        return x

    # =========================================================================
    # Device / state
    # =========================================================================

    def state_dict(self) -> Dict:
        state = super().state_dict()
        state["num_timesteps"] = self.num_timesteps
        state["beta_start"] = self.beta_start
        state["beta_end"] = self.beta_end
        state["noise_schedule"] = self.noise_schedule
        state["prediction_target"] = self.prediction_target
        return state

    @classmethod
    def from_config(cls, model: nn.Module, config: dict, device: torch.device) -> "DDPM":
        ddpm_config = config.get("ddpm", config)
        return cls(
            model=model,
            device=device,
            num_timesteps=ddpm_config["num_timesteps"],
            beta_start=ddpm_config["beta_start"],
            beta_end=ddpm_config["beta_end"],
            noise_schedule=ddpm_config.get("noise_schedule", "linear"),
            prediction_target=ddpm_config.get("prediction_target", "epsilon"),
        )