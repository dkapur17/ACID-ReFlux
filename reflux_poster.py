# %%
"""
ACID ReFlux: Amortized Noise Coupling with Endpoint Consistency
The version used for the Poster - has a critical bug in the NFE budget calculation
=====================================================================

Replaces O(N^3) Hungarian matching with noise oversampling + a per-sample
cache.  At each training step:

    1. Load a batch of N images from the dataloader.
    2. Generate (oversample_factor - 1) * N fresh noise vectors.
    3. Retrieve N cached noise vectors for the current batch.
    4. Run the forward ODE on all oversample_factor * N noise vectors.
    5. For each image, pick whichever noise (fresh or cached) has lowest cost.
    6. Update the cache with the chosen noise and its cost.
    7. Train directly on the N pairs.

The cache is warm-started with random noise for all dataset images, so every
lookup is guaranteed to hit. This simplifies the logic and ensures the cached
noise always participates in the competition from step 1.

Benefits:
    - O(N*M) coupling via oversampled argmin instead of O(N^3) Hungarian.
    - Coupling quality improves monotonically as the cache fills.
    - Cached costs are always fresh (computed with current model).
    - No miss handling — cache is pre-populated.
    - Cache lives in CPU RAM (~3 GB for 60K CelebA samples).

Losses:
    L_cfm:   ||v_θ(x_t, t) - (x_1 - x_0)||^2       random t, linear interp x_t
    L_epc0:  ||v_θ(x_0, 0) - (x_1 - x_0)||^2       endpoint anchor at t=0
    L_epc1:  ||v_θ(x_1, 1) - (x_1 - x_0)||^2       endpoint anchor at t=1

    L_total = L_cfm + λ * (L_epc0 + L_epc1)

EPC = Endpoint Consistency: enforce constant velocity at both endpoints.

Usage:
    python reflux_poster.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np
import time
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.models import create_model_from_config
from src.utils import EMA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.set_float32_matmul_precision('high')


# %%
# ============================================================================
# Configuration
# ============================================================================
@dataclass
class MAMBOTEPCCacheConfig:
    """All hyperparameters for ACID ReFlux fine-tuning."""

    # --- Paths ---
    checkpoint_path: str = "checkpoints/cfm_final.pt"
    data_path: str = "data/celeba-subset/train"
    output_dir: str = "checkpoints/reflux_poster"

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


config = MAMBOTEPCCacheConfig()
Path(config.output_dir).mkdir(parents=True, exist_ok=True)


def compute_nfe_budget_info(cfg: MAMBOTEPCCacheConfig) -> dict:
    """Compute NFE costs per step and max steps under the budget.

    Each step (full mode):
        Forward ODE on oversample * N noise vectors
        = (oversample - 1) * N fresh + N cached
    
    Note: EPC losses at t=0 and t=1 use x_0 and x_1 directly, no ODE needed.
    """
    N = cfg.batch_size
    M = N * cfg.noise_oversample  # total candidates per step
    K = cfg.cost_num_steps

    # Coupling NFEs: M total candidates
    coupling_nfe_per_step = M * K

    # Training: 3 forward passes per sample (random t, t=0, t=1) - single model calls
    training_nfe_per_step = N * 3

    max_steps = cfg.nfe_budget // coupling_nfe_per_step

    return {
        'coupling_nfe_per_step': coupling_nfe_per_step,
        'training_nfe_per_step': training_nfe_per_step,
        'max_steps': max_steps,
        'total_samples': max_steps * N,
        'noise_candidates_per_step': M,
        'fresh_noise_per_step': (cfg.noise_oversample - 1) * N,
    }


nfe_info = compute_nfe_budget_info(config)
# Cache-only steps only cost N*K NFEs (just the cached noise)
cache_only_nfe_per_step = config.batch_size * config.cost_num_steps

print(f"\n{'='*60}")
print(f"NFE Budget Analysis (ACID ReFlux)")
print(f"{'='*60}")
print(f"  Coupling budget:     {config.nfe_budget:,} NFEs")
print(f"  ODE steps:           {config.cost_num_steps}")
print(f"  Batch size:          {config.batch_size}")
print(f"  Noise oversample:    {config.noise_oversample}x ({nfe_info['noise_candidates_per_step']} total candidates)")
print(f"  Fresh noise/step:    {nfe_info['fresh_noise_per_step']} (oversample-1 × batch)")
print(f"  Coupling NFE/step:   {nfe_info['coupling_nfe_per_step']:,}")
print(f"  Cache-only NFE/step: {cache_only_nfe_per_step:,} (N cached only)")
print(f"  Training NFE/step:   {nfe_info['training_nfe_per_step']:,} (3 fwd passes, free)")
print(f"  Max opt steps:       {nfe_info['max_steps']:,} (before cache-only savings)")
print(f"  Cache patience:      {config.cache_patience} steps")
print(f"  Total samples:       {nfe_info['total_samples']:,}")
print(f"  Lambda EPC:          {config.lambda_epc}")
print(f"{'='*60}\n")


# %%
# ============================================================================
# Noise Cache (Warm-Started)
# ============================================================================
class NoiseCache:
    """Per-sample cache of the best (lowest-cost) noise vector found so far.

    The cache is warm-started with random noise for all dataset samples,
    so every lookup is guaranteed to hit. Entries are stored in CPU RAM.

    Each entry stores:
        x_0:  noise vector  (CPU tensor, float32)
        cost: last computed cost (for stats only; selection uses fresh costs)
    """

    def __init__(self, num_samples: int, image_shape: tuple):
        self.image_shape = image_shape
        self.num_samples = num_samples
        
        # Pre-allocate storage for all samples
        print(f"  Initializing noise cache with {num_samples:,} random entries...")
        self.x_0 = torch.randn(num_samples, *image_shape)  # CPU tensor
        self.costs = torch.full((num_samples,), float('inf'))  # Will be updated on first use
        print(f"  Cache size: {self.x_0.numel() * 4 / 1e9:.2f} GB")

    def lookup(self, indices: list[int], device: torch.device) -> torch.Tensor:
        """Return cached noise for a batch of sample indices.

        Returns:
            cached_x0: (B, C, H, W) on *device*
        """
        indices_t = torch.tensor(indices, dtype=torch.long)
        return self.x_0[indices_t].to(device)

    def update(self, indices: list[int], x_0: torch.Tensor, costs: torch.Tensor):
        """Store the chosen noise and its cost."""
        indices_t = torch.tensor(indices, dtype=torch.long)
        self.x_0[indices_t] = x_0.detach().cpu()
        self.costs[indices_t] = costs.detach().cpu()

    def __len__(self):
        return self.num_samples

    def stats(self) -> dict:
        valid_costs = self.costs[self.costs < float('inf')]
        if len(valid_costs) == 0:
            return {'size': self.num_samples, 'mean_cost': 0.0, 'visited': 0}
        return {
            'size': self.num_samples,
            'mean_cost': float(valid_costs.mean()),
            'visited': len(valid_costs),
        }

    def save(self, path: str):
        """Save cache to disk."""
        torch.save({
            'x_0': self.x_0,
            'costs': self.costs,
            'num_samples': self.num_samples,
            'image_shape': self.image_shape,
        }, path)
        print(f"  Cache saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'NoiseCache':
        """Load cache from disk."""
        data = torch.load(path, weights_only=False)
        cache = cls.__new__(cls)
        cache.x_0 = data['x_0']
        cache.costs = data['costs']
        cache.num_samples = data['num_samples']
        cache.image_shape = data['image_shape']
        print(f"  Cache loaded from {path} ({cache.num_samples:,} entries)")
        return cache


# %%
# ============================================================================
# Index-Aware Dataset
# ============================================================================
class IndexedDataset(Dataset):
    """Wraps a dataset so __getitem__ also returns the integer index."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img, label, idx


# %%
# ============================================================================
# Data & Model Setup
# ============================================================================
def get_celeba_dataset_and_loader(data_path, batch_size=128, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize(64), transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    indexed_dataset = IndexedDataset(dataset)
    loader = DataLoader(
        indexed_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    return indexed_dataset, loader


dataset, dataloader = get_celeba_dataset_and_loader(
    config.data_path, batch_size=config.batch_size, num_workers=config.num_workers,
)
num_dataset_samples = len(dataset)
print(f"Dataset size: {num_dataset_samples:,} samples")


def load_source_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_config = checkpoint['config']

    model = create_model_from_config(ckpt_config).to(device)

    state_dict = checkpoint['model']
    if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
        state_dict = {
            key.replace('_orig_mod.', ''): value
            for key, value in state_dict.items()
        }
    model.load_state_dict(state_dict)

    ema = EMA(model, decay=ckpt_config['training']['ema_decay'])
    if 'ema' in checkpoint:
        ema_state_dict = checkpoint['ema']
        if 'shadow' in ema_state_dict and any(
            key.startswith('_orig_mod.') for key in ema_state_dict['shadow'].keys()
        ):
            ema_state_dict['shadow'] = {
                key.replace('_orig_mod.', ''): value
                for key, value in ema_state_dict['shadow'].items()
            }
        ema.load_state_dict(ema_state_dict)

    return model, ckpt_config, ema


model, source_config, ema = load_source_checkpoint(config.checkpoint_path, device)

ema.apply_shadow()
print("Applied EMA weights as starting point for fine-tuning")
model.train()
model = torch.compile(model, mode="default")
print("Model compiled with torch.compile (default mode)")


# %%
# ============================================================================
# ODE Helpers
# ============================================================================
@torch.no_grad()
def ode_forward(model, x, num_steps):
    """Forward ODE t=0->1.  Returns x_final only."""
    dt = 1.0 / num_steps
    B = x.shape[0]
    x_t = x.clone()

    for k in range(num_steps):
        t = torch.full((B,), k * dt, device=x.device)
        v = model(x_t, t)
        x_t = x_t + v * dt

    return x_t


# %%
# ============================================================================
# Evaluation & Diagnostics
# ============================================================================
@torch.no_grad()
def sample_euler(model, batch_size, image_shape, num_steps, device):
    x = torch.randn(batch_size, *image_shape, device=device)
    dt = 1.0 / num_steps
    for step in range(num_steps):
        t = torch.full((batch_size,), step * dt, device=device)
        x = x + model(x, t) * dt
    return x


@torch.no_grad()
def compute_trajectory_straightness(model, batch_size, image_shape, num_steps, device):
    x_0 = torch.randn(batch_size, *image_shape, device=device)
    x_t = x_0.clone()
    dt = 1.0 / num_steps
    velocities = []
    for step in range(num_steps):
        t = torch.full((batch_size,), step * dt, device=device)
        v = model(x_t, t)
        velocities.append(v.clone())
        x_t = x_t + v * dt

    displacement = (x_t - x_0).view(batch_size, -1)
    cos_sims = [
        F.cosine_similarity(v.view(batch_size, -1), displacement, dim=1).mean().item()
        for v in velocities
    ]
    straight_dist = displacement.norm(dim=1)
    path_length = sum(v.view(batch_size, -1).norm(dim=1) * dt for v in velocities)
    path_ratio = (straight_dist / (path_length + 1e-8)).mean().item()
    return {'mean_cos_sim': np.mean(cos_sims), 'path_length_ratio': path_ratio}


@torch.no_grad()
def evaluate_model(model, config):
    model.eval()
    straight = compute_trajectory_straightness(
        model, config.eval_batch_size, config.image_shape,
        config.eval_straightness_K, device,
    )
    results = {f'straightness/{k}': v for k, v in straight.items()}
    model.train()
    return results


def visualize_samples_grid(model, config, step, save_dir):
    model.eval()
    ncols = len(config.eval_num_steps)
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
    if ncols == 1:
        axes = [axes]

    for ax, K in zip(axes, config.eval_num_steps):
        torch.manual_seed(42)
        s = sample_euler(model, config.eval_batch_size, config.image_shape, K, device)
        s = s.clamp(-1, 1)
        grid = make_grid(s.cpu(), nrow=4, normalize=True, value_range=(-1, 1))
        ax.imshow(grid.permute(1, 2, 0).numpy())
        ax.set_title(f'K={K}')
        ax.axis('off')

    plt.suptitle(f'ACID ReFlux Samples — step {step}')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/samples_step{step:06d}.png', dpi=150, bbox_inches='tight')
    plt.close()
    model.train()


def _plot_smoothed(ax, data, color, label, title, ylabel, ylim=None):
    if len(data) == 0:
        return
    ax.plot(data, color=color, alpha=0.15)
    window = min(50, max(1, len(data) // 5))
    if window > 1 and len(data) > window:
        smoothed = np.convolve(data, np.ones(window) / window, mode='valid')
        ax.plot(np.arange(window - 1, window - 1 + len(smoothed)), smoothed,
                color=color, alpha=0.8, label=label)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xlabel('Step', fontsize=8)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)


def plot_training_dashboard(histories, save_path):
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))

    # Row 0, Col 0 — Loss breakdown
    _plot_smoothed(axes[0, 0], histories['total_loss'], 'black', 'total', 'Loss Breakdown', 'Loss')
    _plot_smoothed(axes[0, 0], histories['cfm_loss'], 'tab:blue', 'cfm', '', '')
    _plot_smoothed(axes[0, 0], histories['epc_loss'], 'tab:red', 'epc', '', '')
    axes[0, 0].legend(fontsize=7)

    # Row 0, Col 1 — Coupling cost
    _plot_smoothed(axes[0, 1], histories['coupling_cost'], 'tab:green',
                   'cost', 'Mean Coupling Cost', 'Cost')

    # Row 0, Col 2 — Gradient norm
    _plot_smoothed(axes[0, 2], histories['grad_norm'], 'tab:orange',
                   'grad_norm', 'Gradient Norm', 'Norm')

    # Row 0, Col 3 — NFE usage
    nfe_data = histories.get('cumulative_nfe', [])
    if nfe_data:
        budget = histories.get('nfe_budget', 10_000_000)
        axes[0, 3].plot(nfe_data, color='tab:red', alpha=0.8)
        axes[0, 3].axhline(y=budget, color='black', linestyle='--', alpha=0.5,
                           label=f'Budget: {budget / 1e6:.0f}M')
        axes[0, 3].set_title('Cumulative Coupling NFE', fontsize=9)
        axes[0, 3].set_ylabel('NFEs', fontsize=8)
        axes[0, 3].set_xlabel('Step', fontsize=8)
        axes[0, 3].legend(fontsize=7)
        axes[0, 3].grid(True, alpha=0.3)
        axes[0, 3].tick_params(labelsize=7)

    # Row 1, Col 0 — EPC loss components
    _plot_smoothed(axes[1, 0], histories['epc0_loss'], 'tab:purple', 'epc0', 'EPC Loss Components', 'Loss')
    _plot_smoothed(axes[1, 0], histories['epc1_loss'], 'tab:pink', 'epc1', '', '')
    axes[1, 0].legend(fontsize=7)

    # Row 1, Col 1 — Straightness (cosine similarity)
    st = histories['straightness']
    if st:
        st_steps = [s for s, _ in st]
        cos_sims = [m['straightness/mean_cos_sim'] for _, m in st]
        path_ratios = [m['straightness/path_length_ratio'] for _, m in st]
        axes[1, 1].plot(st_steps, cos_sims, 'tab:blue', marker='o', markersize=3)
    axes[1, 1].set_title('Cosine Similarity (straightness)', fontsize=9)
    axes[1, 1].set_ylabel('Cos Sim', fontsize=8)
    axes[1, 1].set_xlabel('Step', fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].tick_params(labelsize=7)

    # Row 1, Col 2 — Straightness (path-length ratio)
    if st:
        axes[1, 2].plot(st_steps, path_ratios, 'tab:red', marker='o', markersize=3)
    axes[1, 2].set_title('Path-Length Ratio (straightness)', fontsize=9)
    axes[1, 2].set_ylabel('Ratio', fontsize=8)
    axes[1, 2].set_xlabel('Step', fontsize=8)
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].tick_params(labelsize=7)

    # Row 1, Col 3 — Cache stats
    cache_use = histories.get('cache_use_rate', [])
    if cache_use:
        _plot_smoothed(axes[1, 3], cache_use, 'tab:brown', 'use%', 'Cache Use Rate', 'Rate')
        axes[1, 3].legend(fontsize=7)
    else:
        axes[1, 3].set_visible(False)

    plt.suptitle('ACID ReFlux Fine-Tuning Dashboard', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# %%
# ============================================================================
# Training Loop
# ============================================================================
def train_reflux(model, ema, source_config, dataloader, num_dataset_samples, config):
    # Create fresh EMA for fine-tuning (starting from the loaded weights)
    ema = EMA(model, decay=config.ema_decay)

    nfe_info = compute_nfe_budget_info(config)
    coupling_nfe_per_step = nfe_info['coupling_nfe_per_step']
    max_steps = nfe_info['max_steps']
    
    N = config.batch_size
    M_total = N * config.noise_oversample      # total candidates per step
    M_fresh = (config.noise_oversample - 1) * N  # fresh noise per step

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
    )

    # Histories
    total_loss_history = []
    cfm_loss_history = []
    epc_loss_history = []
    epc0_loss_history = []
    epc1_loss_history = []
    coupling_cost_history = []
    grad_norm_history = []
    lr_history = []
    straightness_history = []
    solve_time_history = []
    cumulative_nfe_history = []
    cache_use_rate_history = []

    samples_dir = Path(config.output_dir) / 'samples'
    plots_dir = Path(config.output_dir) / 'plots'
    samples_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Cache-only mode tracking
    cache_only_nfe_per_step = config.batch_size * config.cost_num_steps
    consecutive_full_cache_steps = 0
    cache_only_mode = False
    cache_only_since_step = None

    best_cos_sim = -1.0
    cumulative_nfe = 0
    start_step = 1

    # ======================================================================
    # Check for existing checkpoint and cache to resume training
    # ======================================================================
    checkpoint_path = Path(config.output_dir) / 'reflux_latest.pt'
    cache_path = Path(config.output_dir) / 'noise_cache_latest.pt'

    if checkpoint_path.exists() and cache_path.exists():
        print(f"\n  Found existing checkpoint and cache, resuming training...")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        ema.load_state_dict(checkpoint['ema'])
        
        # Optimizer state is optional (for backward compatibility)
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print("  Warning: No optimizer state in checkpoint, using fresh optimizer")
        
        start_step = checkpoint['step'] + 1
        cumulative_nfe = checkpoint['cumulative_nfe']
        best_cos_sim = checkpoint.get('best_cos_sim', -1.0)
        
        # Load histories
        total_loss_history = checkpoint.get('total_loss_history', [])
        cfm_loss_history = checkpoint.get('cfm_loss_history', [])
        epc_loss_history = checkpoint.get('epc_loss_history', [])
        epc0_loss_history = checkpoint.get('epc0_loss_history', [])
        epc1_loss_history = checkpoint.get('epc1_loss_history', [])
        coupling_cost_history = checkpoint.get('coupling_cost_history', [])
        grad_norm_history = checkpoint.get('grad_norm_history', [])
        lr_history = checkpoint.get('lr_history', [])
        straightness_history = checkpoint.get('straightness_history', [])
        solve_time_history = checkpoint.get('solve_time_history', [])
        cumulative_nfe_history = checkpoint.get('cumulative_nfe_history', [])
        cache_use_rate_history = checkpoint.get('cache_use_rate_history', [])
        
        # Load cache-only mode state
        cache_only_mode = checkpoint.get('cache_only_mode', False)
        cache_only_since_step = checkpoint.get('cache_only_since_step', None)
        consecutive_full_cache_steps = checkpoint.get('consecutive_full_cache_steps', 0)
        
        # Recalculate max_steps if in cache-only mode
        if cache_only_mode:
            remaining_nfe = config.nfe_budget - cumulative_nfe
            extra_steps = remaining_nfe // cache_only_nfe_per_step
            max_steps = start_step + extra_steps - 1
        
        print(f"  Resumed from step {start_step - 1}, cumulative NFE: {cumulative_nfe:,}")
        print(f"  Cache-only mode: {cache_only_mode}")
        
        # Load cache
        noise_cache = NoiseCache.load(str(cache_path))
    else:
        # Warm-start cache with random noise for all samples
        noise_cache = NoiseCache(num_dataset_samples, config.image_shape)

    data_iter = iter(dataloader)

    print("\n" + "=" * 70)
    print("ACID ReFlux Fine-Tuning (Endpoint Consistency)")
    print("=" * 70)
    print(f"  ODE steps:         {config.cost_num_steps}")
    print(f"  Batch size:        {config.batch_size}")
    print(f"  Noise oversample:  {config.noise_oversample}x ({M_total} total candidates)")
    print(f"  Fresh noise/step:  {M_fresh} ({config.noise_oversample - 1}x batch)")
    print(f"  Cached noise/step: {N} (1x batch)")
    print(f"  Cache patience:    {config.cache_patience} steps")
    print(f"  Learning rate:     {config.lr}")
    print(f"  Lambda EPC:        {config.lambda_epc}")
    print(f"  NFE budget:        {config.nfe_budget:,} (coupling only)")
    print(f"  Coupling NFE/step: {coupling_nfe_per_step:,} (full) / {cache_only_nfe_per_step:,} (cache-only)")
    print(f"  Max opt steps:     {max_steps:,} (before cache-only savings)")
    print(f"  Noise cache:       {noise_cache.num_samples:,} entries")
    print(f"  EPC anchors:       t=0 (x_0) and t=1 (x_1)")
    print(f"  Starting step:     {start_step}")
    print(f"  Output:            {config.output_dir}")
    print("=" * 70 + "\n")

    t0 = time.time()

    pbar = tqdm(range(start_step, max_steps + 1), desc="EPC-Cache", dynamic_ncols=True, initial=start_step-1, total=max_steps)
    for step in pbar:
        # ==================================================================
        # 1. Get a batch of images + indices
        # ==================================================================
        try:
            x_1, _, indices = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x_1, _, indices = next(data_iter)

        x_1 = x_1.to(device)
        sample_indices_list = indices.tolist()

        # ==================================================================
        # 2. Noise selection: fresh + cached candidates
        # ==================================================================
        solve_start = time.time()
        model.eval()

        # Get cached noise (always hits due to warm-start)
        cached_x0 = noise_cache.lookup(sample_indices_list, device)  # (N, C, H, W)

        if not cache_only_mode:
            # --- Full mode: generate fresh noise + include cached ---

            # 2a. Generate (oversample - 1) * N fresh noise candidates
            x_0_fresh = torch.randn(M_fresh, *config.image_shape, device=device)

            # 2b. Concatenate fresh + cached for single ODE pass
            all_x0 = torch.cat([x_0_fresh, cached_x0], dim=0)  # (M_total, C, H, W)
            all_x1_pred = ode_forward(model, all_x0, config.cost_num_steps)

            x1_pred_fresh = all_x1_pred[:M_fresh]   # (M_fresh, C, H, W)
            x1_pred_cached = all_x1_pred[M_fresh:]  # (N, C, H, W)

            # 2c. Build N x M_fresh cost matrix for fresh candidates
            x1_flat = x_1.reshape(N, -1)                        # (N, D)
            x1_pred_fresh_flat = x1_pred_fresh.reshape(M_fresh, -1)  # (M_fresh, D)

            x1_sq = (x1_flat ** 2).sum(dim=1, keepdim=True)                    # (N, 1)
            x1_pred_fresh_sq = (x1_pred_fresh_flat ** 2).sum(dim=1, keepdim=True)  # (M_fresh, 1)
            cost_matrix = x1_sq + x1_pred_fresh_sq.T - 2.0 * x1_flat @ x1_pred_fresh_flat.T  # (N, M_fresh)
            cost_matrix = cost_matrix.clamp(min=0.0)

            # 2d. Best fresh noise per image
            best_fresh_idx = cost_matrix.argmin(dim=1)  # (N,)
            fresh_costs = cost_matrix[torch.arange(N, device=device), best_fresh_idx]  # (N,)
            x_0_best_fresh = x_0_fresh[best_fresh_idx]  # (N, C, H, W)

            # 2e. Compute cached costs (fresh, from current model)
            cached_costs = ((x1_pred_cached - x_1).reshape(N, -1) ** 2).sum(dim=1)  # (N,)

            # 2f. For each image, pick whichever has lower cost
            use_cached = cached_costs < fresh_costs

            x_0 = torch.where(use_cached.view(-1, 1, 1, 1), cached_x0, x_0_best_fresh)
            chosen_costs = torch.where(use_cached, cached_costs, fresh_costs)

            step_nfe = coupling_nfe_per_step
        else:
            # --- Cache-only mode: only use cached noise ---
            x1_pred_cached = ode_forward(model, cached_x0, config.cost_num_steps)
            cached_costs = ((x1_pred_cached - x_1).reshape(N, -1) ** 2).sum(dim=1)

            x_0 = cached_x0
            chosen_costs = cached_costs
            use_cached = torch.ones(N, dtype=torch.bool, device=device)

            step_nfe = cache_only_nfe_per_step

        model.train()

        # 2g. Update cache with chosen noise and fresh cost
        noise_cache.update(sample_indices_list, x_0, chosen_costs)

        cumulative_nfe += step_nfe
        solve_elapsed = time.time() - solve_start

        # Stats
        num_cache_used = use_cached.sum().item()
        cache_use_rate = num_cache_used / N
        mean_coupling_cost = chosen_costs.mean().item()

        # --- Cache patience: track consecutive >=99% cache-use steps ---
        if cache_use_rate >= config.crystallization_threshold and not cache_only_mode:
            consecutive_full_cache_steps += 1
        elif not cache_only_mode:
            consecutive_full_cache_steps = 0

        if not cache_only_mode and consecutive_full_cache_steps >= config.cache_patience:
            cache_only_mode = True
            cache_only_since_step = step
            # Recalculate remaining NFE budget → more steps available
            remaining_nfe = config.nfe_budget - cumulative_nfe
            extra_steps = remaining_nfe // cache_only_nfe_per_step
            old_max = max_steps
            max_steps = step + extra_steps
            pbar.total = max_steps
            pbar.refresh()
            tqdm.write(
                f"\n  >>> CACHE-ONLY MODE activated at step {step} "
                f"(patience={config.cache_patience} reached)\n"
                f"      Coupling cost drops from {coupling_nfe_per_step:,} → "
                f"{cache_only_nfe_per_step:,} NFE/step\n"
                f"      Max steps extended: {old_max:,} → {max_steps:,} "
                f"(+{max_steps - old_max:,} steps)\n"
            )

        # ==================================================================
        # 3. Training — direct forward/backward on the N pairs
        # ==================================================================
        optimizer.zero_grad()

        v_target = x_1 - x_0

        # --- A. CFM loss (random t, linear interpolation) ---
        t = torch.rand(N, device=device)
        t_expand = t.view(N, 1, 1, 1)
        x_t = (1.0 - t_expand) * x_0 + t_expand * x_1
        v_pred_cfm = model(x_t, t)
        loss_cfm = F.mse_loss(v_pred_cfm, v_target)
        loss_cfm.backward()

        # --- B. EPC anchor at t=0 ---
        t_zero = torch.zeros(N, device=device)
        v_pred_0 = model(x_0, t_zero)
        loss_epc0 = F.mse_loss(v_pred_0, v_target)
        (config.lambda_epc * loss_epc0).backward()

        # --- C. EPC anchor at t=1 ---
        t_one = torch.ones(N, device=device)
        v_pred_1 = model(x_1, t_one)
        loss_epc1 = F.mse_loss(v_pred_1, v_target)
        (config.lambda_epc * loss_epc1).backward()

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()

        if step >= config.ema_start_step:
            ema.update()

        # Metrics
        loss_epc = config.lambda_epc * (loss_epc0.item() + loss_epc1.item())
        loss_total = loss_cfm.item() + loss_epc

        # ==================================================================
        # Logging
        # ==================================================================
        total_loss_history.append(loss_total)
        cfm_loss_history.append(loss_cfm.item())
        epc_loss_history.append(loss_epc)
        epc0_loss_history.append(loss_epc0.item())
        epc1_loss_history.append(loss_epc1.item())
        coupling_cost_history.append(mean_coupling_cost)
        grad_norm_history.append(grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm))
        lr_history.append(config.lr)
        solve_time_history.append(solve_elapsed)
        cumulative_nfe_history.append(cumulative_nfe)
        cache_use_rate_history.append(cache_use_rate)

        nfe_pct = 100.0 * cumulative_nfe / config.nfe_budget
        mode_tag = 'C' if cache_only_mode else 'O'
        pbar.set_postfix({
            'tot': f'{loss_total:.4f}',
            'cfm': f'{loss_cfm.item():.4f}',
            'epc': f'{loss_epc:.4f}',
            'cost': f'{mean_coupling_cost:.1f}',
            'grad': f'{(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm):.3f}',
            'NFE%': f'{nfe_pct:.0f}',
            'use%': f'{cache_use_rate:.0%}',
            'mode': mode_tag,
        })

        # Stop if NFE budget is exhausted
        if cumulative_nfe >= config.nfe_budget:
            tqdm.write(f"  NFE budget exhausted at step {step}")
            break

        # ==================================================================
        # Eval / Vis / Save
        # ==================================================================
        if step % config.eval_interval == 0:
            if step >= config.ema_start_step:
                ema.apply_shadow()

            ev = evaluate_model(model, config)
            straightness_history.append((step, ev))

            cos_sim = ev['straightness/mean_cos_sim']
            path_ratio = ev['straightness/path_length_ratio']
            cs = noise_cache.stats()
            tqdm.write(
                f"  [EVAL @ step {step}]  cos_sim={cos_sim:.4f}  path_ratio={path_ratio:.4f}  "
                f"visited={cs['visited']}/{cs['size']}  cache_cost={cs['mean_cost']:.2f}"
            )

            if cos_sim > best_cos_sim:
                best_cos_sim = cos_sim
                torch.save(model.state_dict(), f"{config.output_dir}/reflux_poster_best.pt")

            if step >= config.ema_start_step:
                ema.restore()

        if step % config.vis_interval == 0:
            if step >= config.ema_start_step:
                ema.apply_shadow()
            visualize_samples_grid(model, config, step, str(samples_dir))
            if step >= config.ema_start_step:
                ema.restore()

        if step % config.plot_interval == 0:
            histories = {
                'total_loss': total_loss_history,
                'cfm_loss': cfm_loss_history,
                'epc_loss': epc_loss_history,
                'epc0_loss': epc0_loss_history,
                'epc1_loss': epc1_loss_history,
                'coupling_cost': coupling_cost_history,
                'grad_norm': grad_norm_history,
                'lr': lr_history,
                'straightness': straightness_history,
                'solve_time': solve_time_history,
                'cumulative_nfe': cumulative_nfe_history,
                'nfe_budget': config.nfe_budget,
                'cache_use_rate': cache_use_rate_history,
            }
            plot_training_dashboard(histories, str(plots_dir / 'dashboard.png'))

        if step % config.save_interval == 0:
            # Save checkpoint with all state needed for resumption
            checkpoint_data = {
                'model': model.state_dict(),
                'ema': ema.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step,
                'cumulative_nfe': cumulative_nfe,
                'best_cos_sim': best_cos_sim,
                'config': source_config,
                'epc_cache_config': config,
                # Cache-only mode state
                'cache_only_mode': cache_only_mode,
                'cache_only_since_step': cache_only_since_step,
                'consecutive_full_cache_steps': consecutive_full_cache_steps,
                # Histories
                'total_loss_history': total_loss_history,
                'cfm_loss_history': cfm_loss_history,
                'epc_loss_history': epc_loss_history,
                'epc0_loss_history': epc0_loss_history,
                'epc1_loss_history': epc1_loss_history,
                'coupling_cost_history': coupling_cost_history,
                'grad_norm_history': grad_norm_history,
                'lr_history': lr_history,
                'straightness_history': straightness_history,
                'solve_time_history': solve_time_history,
                'cumulative_nfe_history': cumulative_nfe_history,
                'cache_use_rate_history': cache_use_rate_history,
            }
            # Save timestamped checkpoint
            torch.save(checkpoint_data, f"{config.output_dir}/reflux_poster_step{step:06d}.pt")
            # Save latest checkpoint for resumption
            torch.save(checkpoint_data, f"{config.output_dir}/reflux_poster_latest.pt")
            # Save cache
            noise_cache.save(f"{config.output_dir}/noise_cache_latest.pt")

    # ======================================================================
    # Final saves
    # ======================================================================
    final_checkpoint = {
        'model': model.state_dict(),
        'ema': ema.state_dict(),
        'step': step,  # Use actual last step
        'cumulative_nfe': cumulative_nfe,
        'best_cos_sim': best_cos_sim,
        'config': source_config,
        'epc_cache_config': config,
        'cache_only_mode': cache_only_mode,
        'cache_only_since_step': cache_only_since_step,
        'consecutive_full_cache_steps': consecutive_full_cache_steps,
        'total_loss_history': total_loss_history,
        'cfm_loss_history': cfm_loss_history,
        'epc_loss_history': epc_loss_history,
        'epc0_loss_history': epc0_loss_history,
        'epc1_loss_history': epc1_loss_history,
        'coupling_cost_history': coupling_cost_history,
        'grad_norm_history': grad_norm_history,
        'straightness_history': straightness_history,
        'cumulative_nfe_history': cumulative_nfe_history,
        'cache_use_rate_history': cache_use_rate_history,
    }
    torch.save(final_checkpoint, f"{config.output_dir}/reflux_poster_final.pt")
    torch.save(final_checkpoint, f"{config.output_dir}/reflux_poster_latest.pt")
    noise_cache.save(f"{config.output_dir}/noise_cache_latest.pt")
    noise_cache.save(f"{config.output_dir}/noise_cache_final.pt")

    if max_steps >= config.ema_start_step:
        ema.apply_shadow()
    visualize_samples_grid(model, config, max_steps, str(samples_dir))
    if max_steps >= config.ema_start_step:
        ema.restore()

    cs = noise_cache.stats()
    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed / 3600:.2f} hours.")
    print(f"  Total coupling NFE: {cumulative_nfe:,} / {config.nfe_budget:,} "
          f"({100 * cumulative_nfe / config.nfe_budget:.1f}%)")
    print(f"  Best cos_sim:       {best_cos_sim:.4f}")
    print(f"  Cache visited:      {cs['visited']:,} / {cs['size']:,}")
    print(f"  Cache mean cost:    {cs['mean_cost']:.4f}")

    return total_loss_history, cfm_loss_history, epc_loss_history, straightness_history


# %%
print("Starting ACID ReFlux fine-tuning ...")
results = train_reflux(model, ema, source_config, dataloader, num_dataset_samples, config)