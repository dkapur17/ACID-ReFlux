"""
Converts a checkpoint with extra history/stats into the clean format
expected by sample.py (matching train.py's save structure).
"""

import sys
import torch
from dataclasses import dataclass, field

@dataclass
class ReFluxConfig:
    """All hyperparameters for ACID ReFlux fine-tuning."""

    # --- Paths ---
    checkpoint_path: str = "checkpoints/cfm_final.pt"
    data_path: str = "data/celeba-subset/train"
    output_dir: str = "checkpoints/reflux"

    image_shape: tuple = (3, 64, 64)

    # --- Coupling ---
    batch_size: int = 128               # Images per training step
    noise_oversample: int = 8           # Total candidates = batch_size * noise_oversample
    cost_num_steps: int = 2             # ODE steps for cost computation
    cache_patience: int = 50           # Rolling window size; cache-only mode activates when avg usage over this window >= threshold
    crystallization_threshold: float = 0.95

    # --- Endpoint Consistency ---
    lambda_epc: float = 1.0             # Weight for combined EPC anchor losses
    anchor_x_1: bool = False            # Whether to include EPC anchor at t=1 (x_1)

    # --- Training ---
    lr: float = 1e-5  # Reduced from 2e-4 for refinement phase
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    num_workers: int = 4

    # --- NFE Budget (coupling + training) ---
    nfe_budget: int = 40_000_000

    # --- Scheduling ---
    eval_interval: int = 500
    save_interval: int = 2500
    vis_interval: int = 250
    plot_interval: int = 500

    # --- EMA ---
    ema_decay: float = 0.9999
    ema_start_step: int = 200

    # --- Evaluation ---
    eval_num_steps: list = field(default_factory=lambda: [1, 2, 4, 10, 20, 50, 100])
    eval_batch_size: int = 16
    eval_straightness_K: int = 20


@dataclass
class MAMBOTEPCCacheConfig:
    """All hyperparameters for MAMBOT-EPC-Cache fine-tuning."""

    # --- Paths ---
    checkpoint_path: str = "checkpoints/cfm_final.pt"
    data_path: str = "data/celeba-subset/train"
    output_dir: str = "checkpoints/mambot_epc_cache"

    image_shape: tuple = (3, 64, 64)

    # --- Coupling ---
    batch_size: int = 128               # Images per training step
    noise_oversample: int = 8           # Total candidates = batch_size * noise_oversample
    cost_num_steps: int = 2             # ODE steps for cost computation
    cache_patience: int = 50           # After this many consecutive steps with 100% cache use, stop sampling fresh noise
    crystallization_threshold: float = 0.97

    # --- Endpoint Consistency ---
    lambda_epc: float = 1.0             # Weight for combined EPC anchor losses

    # --- Training ---
    lr: float = 5e-5  # Reduced from 2e-4 for refinement phase
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    num_workers: int = 4

    # --- NFE Budget (coupling only) ---
    nfe_budget: int = 60_000_000  # Increased for extended training

    # --- Scheduling ---
    eval_interval: int = 500
    save_interval: int = 2500
    vis_interval: int = 250
    plot_interval: int = 500

    # --- EMA ---
    ema_decay: float = 0.9999
    ema_start_step: int = 200

    # --- Evaluation ---
    eval_num_steps: list = field(default_factory=lambda: [1, 2, 4, 10, 20, 50, 100])
    eval_batch_size: int = 16
    eval_straightness_K: int = 20

keep_keys = ['model', 'optimizer', 'scaler', 'step', 'config', 'ema']

def convert_checkpoint(input_path, output_path=None):
    if output_path is None:
        output_path = input_path.replace('.pt', '_clean.pt')

    print(f"Loading checkpoint from {input_path}...")
    checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)

    # Reconstruct the model to get a clean state dict
    clean_checkpoint = {k:v for k,v in checkpoint.items() if k in keep_keys}

    torch.save(clean_checkpoint, output_path)
    print("Done!")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python convert_checkpoint.py <input_checkpoint> [output_checkpoint]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    convert_checkpoint(input_path, output_path)