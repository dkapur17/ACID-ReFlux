"""
Visualize ODE trajectories of a flow matching / reflow model.

Generates trajectories from noise to data and plots them with x-axis = time (t)
and y-axis = 1D projection (PCA or UMAP). Noise points (t=0) are on the left,
data points (t=1) on the right. Ideal straight-line paths are shown as dashed
lines for comparison — deviation from the dashed line shows trajectory curvature.

Usage:
    python visualize_trajectories.py --checkpoint logs/cfm/run/cfm_final.pt
    python visualize_trajectories.py --checkpoint logs/reflow/2-RF/reflow_final.pt --projection umap
"""

import argparse
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm

from sample import load_checkpoint
from src.methods import FlowMatching


@torch.no_grad()
def generate_trajectories(
    method: FlowMatching,
    image_shape: tuple[int, int, int],
    num_samples: int,
    num_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate ODE trajectories from noise to data.

    Returns:
        states: Tensor of shape (num_steps, num_samples, C*H*W) — flattened
            trajectory states at each timestep.
        timesteps: Tensor of shape (num_steps,).
    """
    device = method.device

    x_0 = torch.randn(num_samples, *image_shape, device=device)
    x = x_0.clone()

    timesteps = torch.linspace(0, 1, num_steps, device=device)
    states = [x_0.flatten(1).cpu()]

    for t, t_next in tqdm(zip(timesteps[:-1], timesteps[1:]), total=num_steps - 1, desc="Integrating"):
        t_batch = torch.full((num_samples,), t, device=device)
        t_next_batch = torch.full((num_samples,), t_next, device=device)
        x = method.reverse_process(x, t_batch, t_next_batch)
        states.append(x.flatten(1).cpu())

    # (num_steps, num_samples, dim)
    states = torch.stack(states)
    return states, timesteps.cpu()


def project_1d(states: torch.Tensor, method: str = "pca") -> np.ndarray:
    """
    Project high-dimensional trajectory states to 1D.

    Args:
        states: Tensor of shape (num_steps, num_samples, dim).
        method: "pca" or "umap".

    Returns:
        projected: ndarray of shape (num_steps, num_samples).
    """
    num_steps, num_samples, dim = states.shape
    flat = states.reshape(-1, dim).numpy()

    if method == "pca":
        from sklearn.decomposition import PCA
        proj = PCA(n_components=1).fit_transform(flat).squeeze(-1)
    elif method == "umap":
        try:
            import umap
        except ImportError:
            raise ImportError("UMAP requires the umap-learn package: pip install umap-learn")
        proj = umap.UMAP(n_components=1, random_state=42).fit_transform(flat).squeeze(-1)
    else:
        raise ValueError(f"Unknown projection method: {method}")

    projected = proj.reshape(num_steps, num_samples)
    # Spread noise points to match data spread.
    # Sort trajectories by data endpoint value, then evenly space noise starts
    # across the same range. Apply a linear-in-t offset so curvature is preserved
    # (linear offsets don't add or hide curvature).

    # data_vals = projected[-1]
    # sort_order = np.argsort(data_vals)
    # projected = projected[:, sort_order]  # sort all timesteps consistently

    # data_min, data_max = projected[-1].min(), projected[-1].max()
    # even_starts = np.linspace(data_min, data_max, num_samples)
    # original_starts = projected[0].copy()

    # offsets = even_starts - original_starts  # per-sample offset at t=0
    # t_grid = np.linspace(0, 1, num_steps)[:, None]  # (num_steps, 1)
    # projected = projected + offsets[None, :] * (1 - t_grid)  # fade offset to 0 at t=1

    noise_vals = projected[0]
    sort_order = np.argsort(noise_vals)
    projected = projected[:, sort_order]  # sort all timesteps consistently

    noise_min, noise_max = projected[0].min(), projected[0].max()
    even_ends = np.linspace(noise_min, noise_max, num_samples)
    original_ends = projected[-1].copy()

    offsets = even_ends - original_ends  # per-sample offset at t=0
    t_grid = np.linspace(0, 1, num_steps)[:, None]  # (num_steps, 1)
    projected = projected + offsets[None, :] * (t_grid)  # fade offset to 0 at t=1

    return projected


def plot_trajectories(
    projected: np.ndarray,
    timesteps: np.ndarray,
    output_path: str,
    title: str = "ODE Trajectories",
):
    """
    Plot trajectories with x=time and y=1D projection.

    Args:
        projected: ndarray of shape (num_steps, num_samples) — 1D projected values.
        timesteps: ndarray of shape (num_steps,).
        output_path: Path to save the figure.
        title: Plot title.
    """
    num_steps, num_samples = projected.shape
    cmap = plt.cm.tab20 if num_samples <= 20 else plt.cm.viridis
    t = timesteps

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    for j in range(num_samples):
        color = cmap(j / max(num_samples - 1, 1))

        # Actual trajectory
        ax.plot(t, projected[:, j], color=color, alpha=0.7, linewidth=1.2)

        # Ideal straight line (dashed) from (0, y_noise) to (1, y_data)
        # ax.plot([t[0], t[-1]], [projected[0, j], projected[-1, j]],
        #         color=color, alpha=0.3, linewidth=0.8, linestyle="--")

        # Start (noise) marker
        ax.scatter(t[0], projected[0, j],
                   color=color, marker="x", s=40, zorder=5)

        # End (data) marker
        ax.scatter(t[-1], projected[-1, j],
                   color=color, marker="o", s=40, zorder=5, edgecolors="black", linewidths=0.5)

    legend_elements = [
        Line2D([0], [0], color="gray", linewidth=1.2, label="Trajectory"),
        Line2D([0], [0], color="gray", linewidth=0.8, linestyle="--", alpha=0.5, label="Ideal (straight)"),
        Line2D([0], [0], marker="x", color="gray", linestyle="None", markersize=6, label="Noise (t=0)"),
        Line2D([0], [0], marker="o", color="gray", linestyle="None", markersize=6,
               markeredgecolor="black", markeredgewidth=0.5, label="Data (t=1)"),
    ]
    # ax.legend(handles=legend_elements, loc="best")
    ax.set_title(title)
    ax.set_xlabel("t")
    ax.set_ylabel("PC 1" if "PCA" in title else "Projection")
    ax.set_xlim(-0.02, 1.02)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize ODE trajectories")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--num_samples", type=int, default=16,
                        help="Number of trajectories to plot")
    parser.add_argument("--num_steps", type=int, default=100,
                        help="Number of ODE integration steps")
    parser.add_argument("--solver", type=str, default="rk4",
                        choices=["euler", "heun", "rk2", "rk4"],
                        help="ODE solver (default: rk4)")
    parser.add_argument("--projection", type=str, default="pca",
                        choices=["pca", "umap"],
                        help="Projection method for 2D (default: pca)")
    parser.add_argument("--output", type=str, default="trajectories.png",
                        help="Output image path (default: trajectories.png)")
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

    # Generate trajectories
    states, timesteps = generate_trajectories(method, image_shape, args.num_samples, args.num_steps)

    # Project to 1D
    print(f"Projecting to 1D via {args.projection.upper()}...")
    projected = project_1d(states, method=args.projection)

    # Plot
    checkpoint_name = os.path.basename(os.path.dirname(args.checkpoint)) or os.path.basename(args.checkpoint)
    title = f"ODE Trajectories — {checkpoint_name} ({args.projection.upper()})"
    plot_trajectories(projected, timesteps.numpy(), args.output, title=title)

    if not args.no_ema:
        ema.restore()


if __name__ == "__main__":
    main()
