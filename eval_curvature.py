"""
Evaluate trajectory curvature of a flow matching / reflow model.

For a perfectly rectified flow, the ODE trajectory from noise x_0 to data x_1
should be a straight line: x_t = (1-t)*x_0 + t*x_1. This script measures how
much the actual trajectories deviate from that ideal.

Metric (per-sample):
    curvature = mean_t[ ||x_t - lerp(x_0, x_1, t)||^2 ] / ||x_1 - x_0||^2

The normalization by ||x_1 - x_0||^2 makes the metric relative (0 = perfectly
straight, higher = more curved), independent of the magnitude of the displacement.

Usage:
    python eval_straightness.py --checkpoint logs/cfm/run/cfm_final.pt --num_samples 256
    python eval_straightness.py --checkpoint logs/reflow/2-RF/reflow_final.pt --num_samples 256
"""

import argparse
import torch
from tqdm import tqdm

from sample import load_checkpoint
from src.methods import FlowMatching


@torch.no_grad()
def compute_curvature(
    method: FlowMatching,
    image_shape: tuple[int, int, int],
    batch_size: int,
    num_steps: int,
) -> torch.Tensor:
    """
    Generate trajectories and compute per-sample curvature.

    Returns:
        curvature: Tensor of shape (batch_size,) with curvature values per sample.
        per_timestep: Tensor of shape (num_interior,) with deviation at each
            interior timestep, averaged over samples in this batch.
    """
    device = method.device

    # Start from noise
    x_0 = torch.randn(batch_size, *image_shape, device=device)
    x = x_0.clone()

    timesteps = torch.linspace(0, 1, num_steps, device=device)

    # Collect all intermediate states (unclamped) including x_0
    # states[i] corresponds to timesteps[i]
    states = [x_0]

    for t, t_next in zip(timesteps[:-1], timesteps[1:]):
        t_batch = torch.full((batch_size,), t, device=device)
        t_next_batch = torch.full((batch_size,), t_next, device=device)
        x = method.reverse_process(x, t_batch, t_next_batch)
        states.append(x)

    # states: list of num_steps tensors, each (batch_size, C, H, W)
    x_1 = states[-1]  # final sample

    # Compute deviation from straight-line interpolation at each intermediate t
    # Skip t=0 and t=1 (endpoints) since deviation is zero by definition
    total_deviation = torch.zeros(batch_size, device=device)
    per_timestep_deviation = []

    for i in range(1, num_steps - 1):
        t = timesteps[i].item()
        x_t = states[i]
        x_t_ideal = (1 - t) * x_0 + t * x_1  # straight-line interpolation

        # Per-sample squared L2 deviation, flattened over C, H, W
        diff = (x_t - x_t_ideal).flatten(1)
        step_deviation = (diff ** 2).sum(dim=1)  # (batch_size,)
        total_deviation += step_deviation
        per_timestep_deviation.append(step_deviation.mean())  # average over batch

    num_interior = len(per_timestep_deviation)

    # Average over interior timesteps
    mean_deviation = total_deviation / num_interior

    # Normalize by ||x_1 - x_0||^2 to get relative curvature
    displacement = (x_1 - x_0).flatten(1)
    displacement_sq = (displacement ** 2).sum(dim=1)
    curvature = mean_deviation / displacement_sq

    # Per-timestep deviation averaged over batch (not normalized — absolute scale)
    per_timestep = torch.stack(per_timestep_deviation)

    return curvature, per_timestep


def main():
    parser = argparse.ArgumentParser(description="Evaluate trajectory curvature")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--num_samples", type=int, default=256,
                        help="Total number of trajectories to evaluate")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for generation")
    parser.add_argument("--num_steps", type=int, default=100,
                        help="Number of ODE steps for trajectory (more = finer-grained)")
    parser.add_argument("--solver", type=str, default="rk4",
                        choices=["euler", "heun", "rk2", "rk4"],
                        help="ODE solver for trajectory integration (default: rk4)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_ema", action="store_true",
                        help="Use training weights instead of EMA weights")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Use a high-order solver (RK4 by default) so intermediate states closely
    # track the true ODE path, giving an accurate curvature measurement
    print(f"Loading checkpoint from {args.checkpoint}...")
    model, config, ema = load_checkpoint(args.checkpoint, device)

    print(f"Using solver: {args.solver}")
    method = FlowMatching.from_config(model, config, device, solver_override=args.solver)

    if not args.no_ema:
        print("Using EMA weights")
        ema.apply_shadow()

    method.eval_mode()

    image_shape = (
        config["data"]["channels"],
        config["data"]["image_size"],
        config["data"]["image_size"],
    )

    # Interior timesteps (exclude t=0 and t=1)
    timesteps = torch.linspace(0, 1, args.num_steps)
    interior_timesteps = timesteps[1:-1]

    # Generate and evaluate in batches
    all_curvature = []
    all_per_timestep = []
    remaining = args.num_samples
    num_batches = 0

    pbar = tqdm(total=args.num_samples, desc="Evaluating curvature")
    while remaining > 0:
        bs = min(args.batch_size, remaining)
        c, pt = compute_curvature(method, image_shape, bs, args.num_steps)
        all_curvature.append(c.cpu())
        all_per_timestep.append(pt.cpu())
        remaining -= bs
        num_batches += 1
        pbar.update(bs)
    pbar.close()

    all_curvature = torch.cat(all_curvature)
    # Average per-timestep deviation across batches
    avg_per_timestep = torch.stack(all_per_timestep).mean(dim=0)

    min_idx = avg_per_timestep.argmin().item()
    max_idx = avg_per_timestep.argmax().item()

    print(f"\n{'='*50}")
    print(f"Trajectory Curvature Evaluation")
    print(f"{'='*50}")
    print(f"Checkpoint:  {args.checkpoint}")
    print(f"Solver:      {args.solver}")
    print(f"Num samples: {args.num_samples}")
    print(f"Num steps:   {args.num_steps}")
    print(f"{'='*50}")
    print(f"Mean curvature: {all_curvature.mean().item():.6f}")
    print(f"Std  curvature: {all_curvature.std().item():.6f}")
    print(f"Min  curvature: {all_curvature.min().item():.6f}")
    print(f"Max  curvature: {all_curvature.max().item():.6f}")
    print(f"{'='*50}")
    print(f"Min deviation at t={interior_timesteps[min_idx].item():.4f}")
    print(f"Max deviation at t={interior_timesteps[max_idx].item():.4f}")
    print(f"{'='*50}")
    print(f"(0 = perfectly straight, higher = more curved)")

    if not args.no_ema:
        ema.restore()


if __name__ == "__main__":
    main()
