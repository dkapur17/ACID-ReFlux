# ACID ReFlux

This repository implements several generative modeling methods on CelebA, culminating in **ACID ReFlux** — a novel fine-tuning approach that straightens flow trajectories using amortized noise coupling and endpoint consistency regularization.

## Methods

### DDPM (`src/methods/ddpm.py`)

Standard Denoising Diffusion Probabilistic Models (Ho et al. 2020). Implements:
- Forward process: `q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)`
- Reverse process with both **DDPM** (stochastic) and **DDIM** (deterministic, Song et al. 2020) samplers
- Supports `epsilon` and `x0` prediction targets
- DDIM enables accelerated sampling with `num_steps << num_timesteps`

### CFM — Continuous Flow Matching (`src/methods/cfm.py`)

Implements continuous flow matching (1-Rectified Flow). The forward process interpolates linearly between noise and data:

```
x_t = (1 - t) * x_0 + t * x_1
```

The model learns to predict the velocity `v = x_1 - x_0` at each interpolated point, and Euler integration integrates the ODE from `t=0` to `t=1` at inference. Supports pluggable solvers (Euler, etc.).

### ReFlow (`reflow.py`)

Rectified Flow iterations (Liu et al. 2022). Starting from a pretrained CFM model (1-RF):

1. **Generate coupled pairs**: run the current model's ODE from noise `x_0 ~ N(0, I)` to get endpoint `x_1`, creating (noise, image) pairs.
2. **Fine-tune on pairs**: train a new model on these coupled pairs to straighten the flow trajectories.
3. Repeat for `k` iterations to produce a k-RF model with increasingly straight trajectories.

Also supports optional **1-step distillation**: after the final ReFlow iteration, train with `t` fixed to `0` so `v_theta(x_0, 0)` directly predicts `x_1 - x_0`, enabling single-step generation.

---

## ACID ReFlux (`reflux.py`)

**ACID** = **A**mortized noise **C**oupling with endpoint cons**I**stency **D**irection

ACID ReFlux is a fine-tuning algorithm that builds on a pretrained CFM model to produce straighter ODE trajectories, making the model more accurate with fewer function evaluations (NFEs). It replaces the O(N^3) Hungarian matching used in naive optimal transport coupling with a scalable, cache-based approach.

### Core Idea

Rather than solving an expensive matching problem, ACID ReFlux maintains a **per-sample noise cache**: for each training image, it stores the best noise vector found so far. At each step, it generates fresh noise candidates and competes them against the cached noise using the current model's ODE, keeping whichever achieves lower transport cost.

### Algorithm

Each training step proceeds as follows:

**1. Load a batch of N images**

Sample a batch of images `x_1` along with their dataset indices.

**2. Oversample noise candidates**

Generate `(oversample_factor - 1) * N` fresh noise vectors. Retrieve `N` cached noise vectors for the current batch indices. This gives `oversample_factor * N` total candidates.

**3. Evaluate cost via forward ODE**

Run the current model's ODE forward (`t=0 -> t=1`) on all candidates using `cost_num_steps` Euler steps. Compute the squared L2 distance between each transported noise and its candidate image:

```
cost(x_0, x_1) = ||ODE(x_0) - x_1||^2
```

**4. Select best noise per image**

For each image, pick whichever candidate (fresh or cached) achieves the lowest cost. Update the cache with the chosen noise vector and cost.

**5. Train on selected pairs**

Use the selected (noise, image) pairs to compute:

- **CFM loss** (random `t`, linear interpolation):
  ```
  L_cfm = ||v_theta(x_t, t) - (x_1 - x_0)||^2
  ```

- **EPC loss** (Endpoint Consistency, active after cache crystallization):
  ```
  L_epc0 = ||v_theta(x_0, 0) - (x_1 - x_0)||^2    # anchor at t=0
  L_epc1 = ||v_theta(x_1, 1) - (x_1 - x_0)||^2    # anchor at t=1 (optional)
  ```

  ```
  L_total = L_cfm + lambda_epc * (L_epc0 + L_epc1)
  ```

EPC enforces that the velocity field is consistent with the true displacement at the trajectory endpoints, which is critical for accurate few-step generation.

### Cache Crystallization & Two-Phase Training

Training proceeds in two phases, triggered automatically:

**Coupling phase (full mode):**
- Each step: run ODE on all `oversample_factor * N` candidates, select best, train with CFM loss only.
- NFE/step = `oversample_factor * N * cost_num_steps` (coupling) + `N` (training)

**Cache-only phase (after crystallization):**
- Activated when the rolling average cache reuse rate exceeds `crystallization_threshold` (default 0.95) over `cache_patience` steps. This means the cache has stabilized — fresh noise rarely beats cached noise.
- Skip ODE evaluation entirely; use cached noise directly.
- Enable EPC losses (both `L_epc0` and optionally `L_epc1`).
- NFE/step = `N * 2` or `N * 3` (training only, no coupling cost)
- The saved NFE budget is redistributed into more training steps.

### Noise Cache

The `NoiseCache` class manages per-sample noise storage:
- **Warm-started** with random noise for all dataset samples — every lookup is guaranteed to hit from step 1.
- Stored in CPU RAM (~3 GB for 60K CelebA samples at 64x64).
- Updated after each step with the newly selected noise and its cost.
- Saved to disk periodically and loaded on resume.

### NFE Budget

ACID ReFlux operates under a fixed **NFE (Neural Function Evaluation) budget** rather than a fixed step count. This ensures fair comparison across different configurations (oversample factor, ODE steps, batch size). The budget is computed as:

```
budget = nfe_budget (default: 40M)
max_steps = budget // nfe_per_step
```

When cache-only mode activates, the reduced per-step NFE cost is used to compute additional training steps from the remaining budget.

### Training Metrics

The training dashboard (`plots/dashboard.png`) tracks:
- Loss breakdown: total, CFM, EPC (t=0 and t=1)
- Mean coupling cost over time
- Gradient norm
- Cumulative NFE vs. budget
- Trajectory straightness: cosine similarity and path-length ratio
- Cache use rate (fraction of steps where cached noise beats fresh)

### Usage

```bash
python reflux.py
```

Configuration is set directly in the `ReFluxConfig` dataclass at the top of `reflux.py`. Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `checkpoint_path` | `checkpoints/cfm_final.pt` | Source CFM checkpoint |
| `data_path` | `data/celeba-subset/train` | Training data |
| `batch_size` | 128 | Images per step |
| `noise_oversample` | 8 | Total candidates = `batch_size * noise_oversample` |
| `cost_num_steps` | 2 | ODE steps for cost evaluation |
| `cache_patience` | 50 | Rolling window for crystallization check |
| `crystallization_threshold` | 0.95 | Cache reuse rate threshold for cache-only mode |
| `lambda_epc` | 1.0 | Weight on EPC anchor losses |
| `anchor_x_1` | False | Whether to include EPC at `t=1` |
| `lr` | 1e-5 | Learning rate (lower than CFM training; this is fine-tuning) |
| `nfe_budget` | 40M | Total NFE budget |

Checkpoints are saved to `checkpoints/reflux/` every `save_interval` steps, and training resumes automatically if a checkpoint and cache are found.

---

## Project Structure

```
acid-reflux/
├── reflux.py                 # ACID ReFlux fine-tuning (main contribution)
├── reflow.py                 # ReFlow iterations + distillation
├── train.py                  # CFM/DDPM training script
├── sample.py                 # Sampling script
├── eval_curvature.py         # Evaluate trajectory curvature
├── visualize_trajectories.py # Visualize ODE trajectories from checkpoints
├── compare.py                # Compare methods side-by-side
├── download_dataset.py       # Download CelebA dataset
├── modal_app.py              # Modal cloud setup
│
├── src/
│   ├── models/
│   │   ├── blocks.py         # U-Net components (ResBlock, Attention, etc.)
│   │   └── unet.py           # U-Net architecture
│   ├── methods/
│   │   ├── base.py           # Base method class
│   │   ├── ddpm.py           # DDPM + DDIM
│   │   ├── cfm.py            # Continuous Flow Matching
│   │   ├── schedulers.py     # Noise schedules
│   │   └── solvers.py        # ODE solvers (Euler, etc.)
│   ├── data/
│   │   └── celeba.py         # CelebA dataset loading
│   └── utils/
│       ├── ema.py            # EMA helper
│       └── logging_utils.py  # Logging utilities
│
└── configs/                  # YAML configs for training runs
```

## Quick Start

### 1. Setup

```bash
./setup-uv.sh
source .venv-cuda121/bin/activate
```

### 2. Download Dataset

```bash
python download_dataset.py
```

### 3. Train CFM (prerequisite for ReFlux)

```bash
python train.py --method cfm --config configs/cfm.yaml
```

### 4. Run ACID ReFlux

Edit `ReFluxConfig` in `reflux.py` to set your checkpoint path, then:

```bash
python reflux.py
```

### 5. Run ReFlow (alternative trajectory straightening)

```bash
python reflow.py --config configs/reflow/reflow.yaml --checkpoint checkpoints/cfm_final.pt
```

### 6. Evaluate

```bash
# Trajectory straightness / curvature
python eval_curvature.py --checkpoint checkpoints/reflux/reflux_final.pt

# Visual comparison
python visualize_trajectories.py --checkpoint checkpoints/reflux/reflux_final.pt
```

---

## References

- Ho et al. (2020). *Denoising Diffusion Probabilistic Models.* NeurIPS 2020.
- Song et al. (2020). *Denoising Diffusion Implicit Models.* ICLR 2021.
- Liu et al. (2022). *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow.* ICLR 2023.
- Lipman et al. (2022). *Flow Matching for Generative Modeling.* ICLR 2023.
