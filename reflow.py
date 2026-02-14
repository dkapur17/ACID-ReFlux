"""
ReFlow: Rectified Flow Iterations

Takes a pretrained CFM model (1-RF), generates coupled (noise, data) pairs
using the model's ODE trajectory, and finetunes on those pairs to straighten
the flow paths. Repeats for multiple iterations to get k-RF models.

Optionally performs a final distillation step for few-step generation.

Usage:
    python reflow.py --config configs/reflow/reflow_16m.yaml --checkpoint path/to/1-RF.pt
"""

import os
import sys
import argparse
import math
import time
from datetime import datetime

import yaml
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.models import create_model_from_config
from src.data import save_image, unnormalize
from src.methods import FlowMatching, get_solver
from src.utils import EMA

import wandb
from PIL import Image as PILImage


def _pairs_cache_dir(checkpoint_path: str, datagen_config: dict, seed: int, iteration: int) -> str:
    """Build a human-readable cache directory path next to the checkpoint."""
    ckpt_dir = os.path.dirname(checkpoint_path)
    ckpt_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    num_pairs = datagen_config.get('num_pairs', 0)
    num_steps = datagen_config.get('num_steps', 0)
    solver = datagen_config.get('solver', 'euler')
    folder = f"{ckpt_name}_pairs{num_pairs}_steps{num_steps}_{solver}_seed{seed}"
    return os.path.join(ckpt_dir, folder, f"iter{iteration}")


def save_pairs_cache(
    cache_dir: str,
    x_0: torch.Tensor,
    x_1: torch.Tensor,
) -> None:
    """Save generated pairs to disk for reuse across runs."""
    os.makedirs(cache_dir, exist_ok=True)
    torch.save(x_0, os.path.join(cache_dir, "x_0.pt"))
    torch.save(x_1, os.path.join(cache_dir, "x_1.pt"))
    print(f"Cached {x_0.shape[0]} pairs to {cache_dir}")


def load_pairs_cache(cache_dir: str) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Load cached pairs from disk. Returns None if cache is missing."""
    x0_path = os.path.join(cache_dir, "x_0.pt")
    x1_path = os.path.join(cache_dir, "x_1.pt")
    if os.path.isfile(x0_path) and os.path.isfile(x1_path):
        x_0 = torch.load(x0_path, map_location="cpu", weights_only=True)
        x_1 = torch.load(x1_path, map_location="cpu", weights_only=True)
        print(f"Loaded {x_0.shape[0]} cached pairs from {cache_dir}")
        return x_0, x_1
    return None


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_source_checkpoint(checkpoint_path: str, device: torch.device):
    """Load a CFM checkpoint and return model, config, EMA."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    model = create_model_from_config(config).to(device)

    state_dict = checkpoint['model']
    if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
        state_dict = {key.replace('_orig_mod.', ''): value
                     for key, value in state_dict.items()}
    model.load_state_dict(state_dict)

    ema = EMA(model, decay=config['training']['ema_decay'])
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

    return model, config, ema


@torch.no_grad()
def generate_pairs(
    method: FlowMatching,
    num_pairs: int,
    image_shape: tuple[int, int, int],
    device: torch.device,
    batch_size: int = 256,
    num_steps: int = 100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate coupled (x_0, x_1) pairs using the current model's ODE.

    Samples noise x_0 ~ N(0,I), integrates through the learned ODE to get x_1.
    Returns paired tensors on CPU to save GPU memory.

    Args:
        method: FlowMatching method with the current model.
        num_pairs: Total number of pairs to generate.
        image_shape: (C, H, W) shape of each image.
        device: Device for generation.
        batch_size: Batch size for generation (no grads, can be large).
        num_steps: Number of ODE steps for integration.

    Returns:
        (x_0_all, x_1_all): Paired tensors of shape (num_pairs, C, H, W) on CPU.
    """
    method.eval_mode()

    all_x0 = []
    all_x1 = []
    remaining = num_pairs

    pbar = tqdm(total=num_pairs, desc="Generating coupled pairs")
    while remaining > 0:
        bs = min(batch_size, remaining)

        # Sample noise
        x_0 = torch.randn(bs, *image_shape, device=device)

        # Integrate ODE: x_0 -> x_1
        timesteps = torch.linspace(0, 1, num_steps, device=device)
        x = x_0.clone()

        for t, t_next in zip(timesteps[:-1], timesteps[1:]):
            t_batch = torch.full((bs,), t, device=device)
            t_next_batch = torch.full((bs,), t_next, device=device)
            x = method.reverse_process(x, t_batch, t_next_batch)

        x_1 = x

        all_x0.append(x_0.cpu())
        all_x1.append(x_1.cpu())

        remaining -= bs
        pbar.update(bs)

    pbar.close()

    return torch.cat(all_x0, dim=0), torch.cat(all_x1, dim=0)


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: EMA,
    scaler: GradScaler,
    step: int,
    config: dict,
):
    """Save training checkpoint."""
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
        'step': step,
        'config': config,
    }
    if ema is not None:
        checkpoint['ema'] = ema.state_dict()
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def save_samples(
    samples: torch.Tensor,
    save_path: str,
    num_samples: int,
) -> None:
    """Save generated samples as an image grid."""
    samples = unnormalize(samples)
    nrow = int(math.ceil(math.sqrt(num_samples)))
    save_image(samples, save_path, nrow=nrow, normalize=False)


@torch.no_grad()
def generate_sample_images(
    method: FlowMatching,
    num_samples: int,
    image_shape: tuple[int, int, int],
    device: torch.device,
    ema: EMA,
    num_steps: int,
    ema_start: int = 0,
    current_step: int = 0,
) -> torch.Tensor:
    """Generate sample images for visualization during training."""
    method.eval_mode()

    use_ema = ema is not None and current_step >= ema_start
    if use_ema:
        ema.apply_shadow()

    samples = method.sample(
        batch_size=num_samples,
        image_shape=image_shape,
        num_steps=num_steps,
    )

    if use_ema:
        ema.restore()

    method.train_mode()
    return samples


def train_reflow_iteration(
    model: nn.Module,
    source_config: dict,
    reflow_config: dict,
    x_0_all: torch.Tensor,
    x_1_all: torch.Tensor,
    iteration: int,
    log_dir: str,
    device: torch.device,
    wandb_run=None,
) -> tuple[nn.Module, EMA]:
    """
    Run one ReFlow training iteration on coupled pairs.

    Args:
        model: The model to finetune.
        source_config: Original model config (for model/data settings).
        reflow_config: ReFlow config (for training hyperparams).
        x_0_all: Noise tensor (N, C, H, W) on CPU.
        x_1_all: Data tensor (N, C, H, W) on CPU.
        iteration: Which RF iteration (2, 3, ...).
        log_dir: Directory for logs/samples/checkpoints.
        device: Device for training.
        wandb_run: Optional wandb run for logging.

    Returns:
        (model, ema): Trained model and EMA.
    """
    training_config = reflow_config['training']
    data_config = source_config['data']

    # Create dataloader from in-memory pairs
    dataset = TensorDataset(x_0_all, x_1_all)
    dataloader = DataLoader(
        dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
        num_workers=0,  # in-memory data, no need for workers
        pin_memory=True,
        drop_last=True,
    )

    # Create method
    cfm_config = source_config.get('cfm', {})
    solver_name = reflow_config.get('sampling', {}).get('solver', cfm_config.get('solver', 'euler'))
    method = FlowMatching(
        model=model,
        device=device,
        num_timesteps=cfm_config.get('num_timesteps', 100),
        solver=get_solver(solver_name),
    )

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config['learning_rate'],
        betas=tuple(training_config.get('betas', [0.9, 0.999])),
        weight_decay=training_config.get('weight_decay', 0.0),
    )

    # EMA
    ema = EMA(model, decay=training_config.get('ema_decay', 0.9999))

    # Mixed precision
    use_amp = reflow_config.get('infrastructure', {}).get('mixed_precision', True)
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    scaler = GradScaler(device_type, enabled=use_amp)

    # Training params
    num_iterations = training_config['num_iterations']
    log_every = training_config.get('log_every', 100)
    sample_every = training_config.get('sample_every', 2000)
    save_every = training_config.get('save_every', 10000)
    num_samples = training_config.get('num_samples', 16)
    gradient_clip_norm = training_config.get('gradient_clip_norm', 1.0)
    ema_start = training_config.get('ema_start', 2000)

    image_shape = (data_config['channels'], data_config['image_size'], data_config['image_size'])
    sampling_steps = reflow_config.get('sampling', {}).get('num_steps', 100)

    # Directories
    iter_dir = os.path.join(log_dir, f"{iteration}-RF")
    samples_dir = os.path.join(iter_dir, 'samples')
    checkpoints_dir = os.path.join(iter_dir, 'checkpoints')
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"ReFlow Iteration {iteration}: Training {iteration}-RF")
    print(f"{'=' * 60}")
    print(f"  Pairs: {len(dataset)}")
    print(f"  Iterations: {num_iterations}")
    print(f"  Batch size: {training_config['batch_size']}")
    print(f"  Learning rate: {training_config['learning_rate']}")
    print(f"  Output: {iter_dir}")
    print(f"{'=' * 60}")

    method.train_mode()
    data_iter = iter(dataloader)

    metrics_sum = {}
    metrics_count = 0
    start_time = time.time()

    pbar = tqdm(range(num_iterations), desc=f"{iteration}-RF Training")
    for step in pbar:
        # Get batch (cycle through dataset)
        try:
            x_0_batch, x_1_batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x_0_batch, x_1_batch = next(data_iter)

        x_0_batch = x_0_batch.to(device)
        x_1_batch = x_1_batch.to(device)

        # Forward pass
        optimizer.zero_grad()

        with autocast(device_type, enabled=use_amp):
            loss, metrics = method.compute_loss(x_1_batch, x_0=x_0_batch)

        # Backward pass
        scaler.scale(loss).backward()

        if gradient_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)

        scaler.step(optimizer)
        scaler.update()

        # EMA
        ema.update()

        # Accumulate metrics
        for k, v in metrics.items():
            if k not in metrics_sum:
                metrics_sum[k] = []
            metrics_sum[k].append(v if isinstance(v, float) else v)
        metrics_count += 1

        # Logging
        if (step + 1) % log_every == 0:
            elapsed = time.time() - start_time
            steps_per_sec = metrics_count / elapsed

            avg_metrics = {k: sum(v) / len(v) for k, v in metrics_sum.items()}

            pbar.set_postfix({
                'loss': f"{avg_metrics['loss']:.4f}",
                'steps/s': f"{steps_per_sec:.2f}",
            })

            if wandb_run is not None:
                log_dict = {
                    f'{iteration}-RF/step': step + 1,
                    f'{iteration}-RF/loss': avg_metrics['loss'],
                    f'{iteration}-RF/steps_per_sec': steps_per_sec,
                    f'{iteration}-RF/learning_rate': optimizer.param_groups[0]['lr'],
                }
                try:
                    wandb.log(log_dict)
                except Exception as e:
                    print(f"Warning: Failed to log to wandb: {e}")

            metrics_sum = {}
            metrics_count = 0
            start_time = time.time()

        # Generate samples
        if (step + 1) % sample_every == 0:
            print(f"\nGenerating samples at step {step + 1}...")
            samples = generate_sample_images(
                method, num_samples, image_shape, device, ema,
                num_steps=sampling_steps, ema_start=ema_start, current_step=step + 1,
            )
            sample_path = os.path.join(samples_dir, f'samples_{step + 1:07d}.png')
            save_samples(samples, sample_path, num_samples)

            if wandb_run is not None:
                try:
                    img = PILImage.open(sample_path)
                    wandb.log({
                        f'{iteration}-RF/samples': wandb.Image(img, caption=f'{iteration}-RF Step {step + 1}')
                    })
                except Exception as e:
                    print(f"Warning: Failed to log samples to wandb: {e}")

        # Save checkpoint
        if (step + 1) % save_every == 0:
            checkpoint_path = os.path.join(checkpoints_dir, f'{iteration}-RF_{step + 1:07d}.pt')
            save_checkpoint(checkpoint_path, model, optimizer, ema, scaler, step + 1, source_config)

    # Save final checkpoint for this iteration
    final_path = os.path.join(checkpoints_dir, f'{iteration}-RF_final.pt')
    save_checkpoint(final_path, model, optimizer, ema, scaler, num_iterations, source_config)

    print(f"\n{iteration}-RF training complete!")
    return model, ema


def run_reflow(config: dict, checkpoint: str):
    """
    Run the full ReFlow pipeline.

    Loads a pretrained 1-RF model, then iterates:
      1. Generate coupled pairs using k-RF
      2. Finetune on pairs to get (k+1)-RF
    Optionally performs a final distillation step.

    Args:
        config: ReFlow configuration dictionary.
        checkpoint: Path to the pretrained 1-RF checkpoint.
    """
    # Device setup
    infra_config = config.get('infrastructure', {})
    device_name = infra_config.get('device', 'cuda')
    use_cuda = torch.cuda.is_available() and device_name != 'cpu'
    use_mps = torch.mps.is_available() and device_name != 'cpu'
    device = torch.device('cuda' if use_cuda else 'mps' if use_mps else 'cpu')

    print(f"Using device: {device}")

    # Seed
    seed = infra_config.get('seed', 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Load source model (1-RF)
    source_checkpoint = checkpoint
    print(f"Loading source model from {source_checkpoint}...")
    model, source_config, ema = load_source_checkpoint(source_checkpoint, device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,} ({num_params / 1e6:.2f}M)")

    # Apply EMA weights as starting point
    ema.apply_shadow()
    print("Applied EMA weights as starting point for ReFlow")

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_base = config.get('logging', {}).get('dir', './logs')
    run_name = config.get('logging', {}).get('run_name', f'reflow_{timestamp}')
    if run_name is None:
        run_name = f'reflow_{timestamp}'
    log_dir = os.path.join(log_base, run_name)
    os.makedirs(log_dir, exist_ok=True)

    # Save config
    with open(os.path.join(log_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    # Wandb
    wandb_run = None
    wandb_config = config.get('logging', {}).get('wandb', {})
    if wandb_config.get('enabled', False):
        try:
            wandb_run = wandb.init(
                project=wandb_config.get('project', 'cmu-10799'),
                entity=wandb_config.get('entity', None),
                name=run_name,
                config=config,
                dir=log_dir,
                tags=['reflow'],
            )
            print(f"Weights & Biases: {wandb_run.url}")
        except Exception as e:
            print(f"Warning: Failed to initialize wandb: {e}")

    # Image shape
    data_config = source_config['data']
    image_shape = (data_config['channels'], data_config['image_size'], data_config['image_size'])

    # Datagen config
    datagen_config = config['datagen']
    num_pairs = datagen_config['num_pairs']
    gen_batch_size = datagen_config['batch_size']
    gen_num_steps = datagen_config['num_steps']
    gen_solver_name = datagen_config.get('solver', 'euler')
    cache_enabled = datagen_config.get('cache', False)

    # ReFlow iterations
    num_rf_iterations = config['reflow']['num_iterations']

    for rf_iter in range(num_rf_iterations):
        iteration = rf_iter + 2  # 1-RF is the source, so we start at 2-RF

        # Step 1: Try loading cached pairs, otherwise generate them
        print(f"\n{'=' * 60}")
        print(f"ReFlow Iteration {rf_iter + 1}/{num_rf_iterations}: Generating pairs for {iteration}-RF")
        print(f"{'=' * 60}")

        # Build per-iteration cache directory
        cache_dir = None
        if cache_enabled:
            cache_dir = _pairs_cache_dir(checkpoint, datagen_config, seed, rf_iter)

        cached = load_pairs_cache(cache_dir) if cache_dir else None

        if cached is not None:
            x_0_all, x_1_all = cached
        else:
            # Create method with datagen solver for pair generation
            cfm_config = source_config.get('cfm', {})
            gen_method = FlowMatching(
                model=model,
                device=device,
                num_timesteps=cfm_config.get('num_timesteps', 100),
                solver=get_solver(gen_solver_name),
            )

            x_0_all, x_1_all = generate_pairs(
                method=gen_method,
                num_pairs=num_pairs,
                image_shape=image_shape,
                device=device,
                batch_size=gen_batch_size,
                num_steps=gen_num_steps,
            )

            if cache_dir is not None:
                save_pairs_cache(cache_dir, x_0_all, x_1_all)

        print(f"Pairs ready: {x_0_all.shape[0]} coupled pairs")
        print(f"  x_0 shape: {x_0_all.shape}, x_1 shape: {x_1_all.shape}")
        print(f"  Memory: {(x_0_all.nbytes + x_1_all.nbytes) / 1e9:.2f} GB")

        # Step 2: Finetune on coupled pairs
        model, ema = train_reflow_iteration(
            model=model,
            source_config=source_config,
            reflow_config=config,
            x_0_all=x_0_all,
            x_1_all=x_1_all,
            iteration=iteration,
            log_dir=log_dir,
            device=device,
            wandb_run=wandb_run,
        )

        # Apply EMA for next iteration's pair generation
        ema.apply_shadow()
        print(f"Applied EMA weights from {iteration}-RF for next iteration")

        # Free pair memory
        del x_0_all, x_1_all
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Optional distillation
    distill_config = config.get('distillation', {})
    if distill_config.get('enabled', False):
        print(f"\n{'=' * 60}")
        print(f"Distillation: Generating pairs from final {iteration}-RF")
        print(f"{'=' * 60}")

        distill_datagen = distill_config.get('datagen', datagen_config)
        distill_num_pairs = distill_datagen.get('num_pairs', num_pairs)
        distill_batch_size = distill_datagen.get('batch_size', gen_batch_size)
        distill_num_steps = distill_datagen.get('num_steps', gen_num_steps)
        distill_solver_name = distill_datagen.get('solver', gen_solver_name)

        cfm_config = source_config.get('cfm', {})
        distill_gen_method = FlowMatching(
            model=model,
            device=device,
            num_timesteps=cfm_config.get('num_timesteps', 100),
            solver=get_solver(distill_solver_name),
        )

        x_0_all, x_1_all = generate_pairs(
            method=distill_gen_method,
            num_pairs=distill_num_pairs,
            image_shape=image_shape,
            device=device,
            batch_size=distill_batch_size,
            num_steps=distill_num_steps,
        )

        print(f"Generated {distill_num_pairs} distillation pairs")

        # Build a training config for distillation, falling back to reflow training config
        distill_training = distill_config.get('training', config['training'])

        distill_full_config = {
            'training': distill_training,
            'sampling': config.get('sampling', {}),
            'infrastructure': config.get('infrastructure', {}),
        }

        model, ema = train_reflow_iteration(
            model=model,
            source_config=source_config,
            reflow_config=distill_full_config,
            x_0_all=x_0_all,
            x_1_all=x_1_all,
            iteration='distilled',
            log_dir=log_dir,
            device=device,
            wandb_run=wandb_run,
        )

        del x_0_all, x_1_all

    # Finish
    if wandb_run is not None:
        try:
            wandb.finish()
        except Exception as e:
            print(f"Warning: Failed to finish wandb run: {e}")

    print(f"\nReFlow complete! All outputs saved to {log_dir}")


def main():
    parser = argparse.ArgumentParser(description='ReFlow: Rectified Flow Iterations')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to ReFlow config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to pretrained 1-RF (CFM) checkpoint')

    args = parser.parse_args()
    config = load_config(args.config)
    run_reflow(config, checkpoint=args.checkpoint)


if __name__ == '__main__':
    main()
