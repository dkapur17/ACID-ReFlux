"""
Sampling Script for DDPM (Denoising Diffusion Probabilistic Models)

Generate samples from a trained model. By default, saves individual images to avoid
memory issues with large sample counts. Use --grid to generate a single grid image.

Usage:
    # Sample from DDPM (saves individual images to ./samples/)
    python sample.py --checkpoint checkpoints/ddpm_final.pt --method ddpm --num_samples 64

    # With custom number of sampling steps
    python sample.py --checkpoint checkpoints/ddpm_final.pt --method ddpm --num_steps 500

    # Generate a grid image instead of individual images
    python sample.py --checkpoint checkpoints/ddpm_final.pt --method ddpm --num_samples 64 --grid

    # Save individual images to custom directory
    python sample.py --checkpoint checkpoints/ddpm_final.pt --method ddpm --output_dir my_samples

What you need to implement:
- Incorporate your sampling scheme to this pipeline
- Save generated samples as images for logging
"""

import os
import sys
import argparse
from datetime import datetime

import yaml
import torch
from tqdm import tqdm

from src.models import create_model_from_config
from src.data import save_image, unnormalize
import math
from src.methods import DDPM, FlowMatching
from src.utils import EMA


def load_checkpoint(checkpoint_path: str, device: torch.device):
    """Load checkpoint and return model, config, and EMA."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    # Create model
    model = create_model_from_config(config).to(device)

    # Handle torch.compile() wrapper prefix
    # If model was compiled during training, state_dict keys have '_orig_mod.' prefix
    state_dict = checkpoint['model']
    if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
        state_dict = {key.replace('_orig_mod.', ''): value
                     for key, value in state_dict.items()}

    model.load_state_dict(state_dict)

    # Create EMA and load
    ema = EMA(model, decay=config['training']['ema_decay'])

    # Handle torch.compile() prefix in EMA state_dict too
    ema_state_dict = checkpoint['ema']
    if 'shadow' in ema_state_dict and any(key.startswith('_orig_mod.') for key in ema_state_dict['shadow'].keys()):
        ema_state_dict['shadow'] = {key.replace('_orig_mod.', ''): value
                                     for key, value in ema_state_dict['shadow'].items()}

    ema.load_state_dict(ema_state_dict)

    return model, config, ema


def save_samples(
    samples: torch.Tensor,
    save_path: str,
    sample_idx: int | None = None,
    nrow: int | None = None,
) -> None:
    """
    Save generated samples as images.

    Args:
        samples: Generated samples tensor with shape (num_samples, C, H, W).
        save_path: File path to save the image grid.
        num_samples: Number of samples, used to calculate grid layout.
        nrow: Number of images per row in the grid (default: 8).
    """
    # Unnormalize from [-1, 1] to [0, 1]
    samples = unnormalize(samples)

    # Save image grid
    if sample_idx is not None:
        save_image(samples[sample_idx], save_path, nrow=1)
    elif nrow:
        save_image(samples, save_path, nrow=nrow)
    else:
        save_image(samples, save_path)

def main():
    parser = argparse.ArgumentParser(description='Generate samples from trained model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--method', type=str, required=True,
                       choices=['ddpm', 'cfm'],
                       help='Method used for training (currently only ddpm is supported)')
    parser.add_argument('--num_samples', type=int, default=64,
                       help='Number of samples to generate')
    parser.add_argument('--output_dir', type=str, default='samples',
                       help='Directory to save individual images (default: samples)')
    parser.add_argument('--grid', action='store_true',
                       help='Save as grid image instead of individual images')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for grid (only used with --grid, default: samples_<timestamp>.png)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for generation')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    
    # Sampling arguments
    parser.add_argument('--num_steps', type=int, default=None,
                       help='Number of sampling steps (default: from config)')
    parser.add_argument('--sampler', type=str, default='ddpm',
                       choices=['ddpm', 'ddim'],
                       help='Sampler to use for DDPM (default: ddpm). Ignored for cfm.')
    parser.add_argument('--solver', type=str, default=None,
                       choices=['euler', 'heun', 'rk2', 'rk4'],
                       help='ODE solver for CFM (default: from config). Ignored for ddpm.')
    
    # Other options
    parser.add_argument('--no_ema', action='store_true',
                       help='Use training weights instead of EMA weights')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    model, config, ema = load_checkpoint(args.checkpoint, device)
    
    # Create method
    if args.method == 'ddpm':
        method = DDPM.from_config(model, config, device)
    elif args.method == 'cfm':
        method = FlowMatching.from_config(model, config, device, solver_override=args.solver)
    else:
        raise ValueError(f"Unknown method: {args.method}. Supported: 'ddpm', 'cfm'.")
    
    # Apply EMA weights
    if not args.no_ema:
        print("Using EMA weights")
        ema.apply_shadow()
    else:
        print("Using training weights (no EMA)")
    
    method.eval_mode()
    
    # Image shape
    data_config = config['data']
    image_shape = (data_config['channels'], data_config['image_size'], data_config['image_size'])
    
    # Generate samples
    print(f"Generating {args.num_samples} samples...")

    all_samples = []
    remaining = args.num_samples
    sample_idx = 0

    # Create output directory if saving individual images
    if not args.grid:
        os.makedirs(args.output_dir, exist_ok=True)

    with torch.no_grad():
        pbar = tqdm(total=args.num_samples, desc="Generating samples")
        while remaining > 0:
            batch_size = min(args.batch_size, remaining)

            num_steps = args.num_steps or config['sampling']['num_steps']

            sampling_kwargs = {}
            if args.method == 'ddpm':
                sampling_kwargs['sampler'] = args.sampler

            samples = method.sample(
                batch_size=batch_size,
                image_shape=image_shape,
                num_steps=num_steps,
                show_progress=True,
                **sampling_kwargs,
            )

            # Save individual images immediately or collect for grid
            if args.grid:
                all_samples.append(samples)
            else:
                for i in range(samples.shape[0]):
                    img_path = os.path.join(args.output_dir, f"{sample_idx:06d}.png")
                    save_samples(samples, img_path, sample_idx=i)
                    sample_idx += 1

            remaining -= batch_size
            pbar.update(batch_size)

        pbar.close()

    # Save samples
    if args.grid:
        # Concatenate all samples for grid
        all_samples = torch.cat(all_samples, dim=0)[:args.num_samples]

        if args.output is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output = f"samples_{timestamp}.png"

        save_samples(all_samples, args.output, nrow=8)
        print(f"Saved grid to {args.output}")
    else:
        print(f"Saved {args.num_samples} individual images to {args.output_dir}")

    # Restore EMA if applied
    if not args.no_ema:
        ema.restore()


if __name__ == '__main__':
    main()
