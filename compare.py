"""
Compare sampling quality across multiple checkpoints at different step counts.

For each checkpoint, generates samples at K=1, 2, 4, 10, 20, 50, 100
Euler steps, then stacks all checkpoints vertically in a single figure.

Usage:
    python compare.py --checkpoints checkpoints/cfm_final.pt checkpoints/reflux.pt

    # 6 samples in a row
    python compare.py --checkpoints ckpt1.pt --samples 6

    # 4x4 grid of samples
    python compare.py --checkpoints ckpt1.pt --samples 4x4

    # Custom labels for each checkpoint
    python compare.py --checkpoints ckpt1.pt ckpt2.pt --labels "CFM" "ReFlux"

    # Use training weights instead of EMA
    python compare.py --checkpoints ckpt1.pt --no_ema
"""

import os
import argparse
from datetime import datetime

import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torchvision.utils import make_grid

from src.models import create_model_from_config
from src.data import unnormalize
from src.methods import FlowMatching
from src.methods.solvers import EulerSolver
from src.utils import EMA


STEP_COUNTS = [1, 2, 4, 10, 20, 50, 100]


def load_checkpoint(checkpoint_path: str, device: torch.device):
    """Load checkpoint and return model, config, and EMA."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    model = create_model_from_config(config).to(device)

    # Handle torch.compile() wrapper prefix
    state_dict = checkpoint['model']
    if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
        state_dict = {key.replace('_orig_mod.', ''): value
                      for key, value in state_dict.items()}
    model.load_state_dict(state_dict)

    # Create EMA and load
    ema = EMA(model, decay=config['training']['ema_decay'])
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


def parse_samples_arg(value: str) -> tuple[int, int]:
    """Parse --samples arg. Returns (nrow, ncol) e.g. '4' -> (4,1), '4x4' -> (4,4)."""
    if 'x' in value:
        parts = value.split('x')
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(f"Invalid samples format '{value}'. Use N or NxM.")
        return int(parts[0]), int(parts[1])
    else:
        return int(value), 1


def generate_grids_for_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    nrow: int,
    ncol: int,
    use_ema: bool = True,
    seed: int = 42,
) -> list[torch.Tensor]:
    """
    Generate samples for each step count in STEP_COUNTS.

    Returns a list of grid tensors (one per step count), each as a (3, H, W) image.
    """
    print(f"\nLoading {checkpoint_path}...")
    model, config, ema = load_checkpoint(checkpoint_path, device)

    method = FlowMatching(
        model=model,
        device=device,
        num_timesteps=config.get('cfm', config).get('num_timesteps', 1000),
        solver=EulerSolver(),
    )

    if use_ema:
        ema.apply_shadow()

    method.eval_mode()

    data_config = config['data']
    image_shape = (data_config['channels'], data_config['image_size'], data_config['image_size'])

    # Use the same noise for all step counts so we can compare trajectories
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    num_samples = nrow * ncol
    fixed_noise = torch.randn(num_samples, *image_shape, device=device)

    grids = []
    for K in STEP_COUNTS:
        print(f"  Sampling K={K}...")
        with torch.no_grad():
            x = fixed_noise.clone()
            timesteps = torch.linspace(0, 1, K + 1, device=device)

            for t, t_next in zip(timesteps[:-1], timesteps[1:]):
                batch_size = x.shape[0]
                t_batch = torch.full((batch_size,), t, device=device)
                t_next_batch = torch.full((batch_size,), t_next, device=device)
                x = method.reverse_process(x, t_batch, t_next_batch)

        # Unnormalize and make grid
        samples = unnormalize(x.clamp(-1, 1))
        grid = make_grid(samples, nrow=nrow, padding=1, pad_value=1.0)
        grids.append(grid.cpu())

    if use_ema:
        ema.restore()

    return grids


def main():
    parser = argparse.ArgumentParser(
        description='Compare sampling quality across checkpoints at different step counts'
    )
    parser.add_argument('--checkpoints', type=str, nargs='+', required=True,
                        help='Paths to model checkpoints')
    parser.add_argument('--labels', type=str, nargs='+', default=None,
                        help='Labels for each checkpoint (default: derived from filename)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path (default: comparison_<timestamp>.png)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--samples', type=str, default='4',
                        help='Number of samples: N for a row of N, or NxM for an NxM grid (default: 4)')
    parser.add_argument('--no_ema', action='store_true',
                        help='Use training weights instead of EMA weights')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    args = parser.parse_args()
    nrow, ncol = parse_samples_arg(args.samples)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    num_checkpoints = len(args.checkpoints)

    # Derive labels
    if args.labels is not None:
        if len(args.labels) != num_checkpoints:
            raise ValueError(
                f"Number of labels ({len(args.labels)}) must match "
                f"number of checkpoints ({num_checkpoints})"
            )
        labels = args.labels
    else:
        labels = [
            os.path.splitext(os.path.basename(p))[0] for p in args.checkpoints
        ]

    # Generate grids for each checkpoint
    all_grids = []  # list of (list of grid tensors per step count)
    for ckpt_path in args.checkpoints:
        grids = generate_grids_for_checkpoint(
            ckpt_path, device, nrow=nrow, ncol=ncol,
            use_ema=not args.no_ema, seed=args.seed,
        )
        all_grids.append(grids)

    # Compose all grids into a single image, then annotate with matplotlib
    num_steps = len(STEP_COUNTS)

    # All grids have same size; get dimensions from first
    grid_h, grid_w = all_grids[0][0].shape[1], all_grids[0][0].shape[2]
    vgap = 2   # pixels between rows
    hgap = 10  # pixels between columns (timesteps)

    # Total canvas size
    canvas_w = num_steps * grid_w + (num_steps - 1) * hgap
    canvas_h = num_checkpoints * grid_h + (num_checkpoints - 1) * vgap
    canvas = torch.ones(3, canvas_h, canvas_w)  # white background

    for row_idx, grids in enumerate(all_grids):
        for col_idx, grid in enumerate(grids):
            y = row_idx * (grid_h + vgap)
            x = col_idx * (grid_w + hgap)
            canvas[:, y:y + grid_h, x:x + grid_w] = grid

    # Plot the composited canvas with labels
    label_margin = 0.06  # fraction of width for row labels
    header_margin = 0.04  # fraction of height for column headers
    img_w_in = canvas_w / 80  # scale pixels to inches
    img_h_in = canvas_h / 80
    fig_w = img_w_in / (1 - label_margin)
    fig_h = img_h_in / (1 - header_margin)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.subplots_adjust(left=label_margin, right=1.0, top=1.0 - header_margin, bottom=0.0)

    ax.imshow(canvas.permute(1, 2, 0).numpy())
    ax.axis('off')

    # Column headers
    for col_idx, K in enumerate(STEP_COUNTS):
        x_center = col_idx * (grid_w + hgap) + grid_w / 2
        ax.text(x_center, -vgap * 2, f'K={K}', fontsize=10,
                ha='center', va='bottom')

    # Row labels
    for row_idx, label in enumerate(labels):
        y_center = row_idx * (grid_h + vgap) + grid_h / 2
        ax.text(-hgap, y_center, label, fontsize=10,
                ha='right', va='center', rotation=90)

    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"comparison_{timestamp}.png"

    fig.savefig(args.output, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved comparison to {args.output}")


if __name__ == '__main__':
    main()
