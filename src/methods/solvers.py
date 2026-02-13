from abc import ABC, abstractmethod
from typing import Callable

import torch


class Solver(ABC):
    """Base class for ODE solvers used in flow matching."""

    @abstractmethod
    def step(
        self,
        model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform a single solver step.

        Args:
            model: Velocity field v(x, t) -> dx/dt
            x_t: Current state
            t: Current time (batch,)
            t_next: Next time (batch,)

        Returns:
            x at t_next
        """
        pass


class EulerSolver(Solver):
    """Forward Euler method. 1 NFE per step."""

    def step(self, model, x_t, t, t_next):
        dt = (t_next - t).view(-1, *([1] * (x_t.dim() - 1)))
        return x_t + model(x_t, t) * dt


class HeunSolver(Solver):
    """Heun's method (improved Euler / explicit trapezoidal). 2 NFEs per step."""

    def step(self, model, x_t, t, t_next):
        dt = (t_next - t).view(-1, *([1] * (x_t.dim() - 1)))
        v1 = model(x_t, t)
        x_pred = x_t + v1 * dt
        v2 = model(x_pred, t_next)
        return x_t + 0.5 * (v1 + v2) * dt


class RK2Solver(Solver):
    """Explicit midpoint method (RK2). 2 NFEs per step."""

    def step(self, model, x_t, t, t_next):
        dt = (t_next - t).view(-1, *([1] * (x_t.dim() - 1)))
        t_mid = 0.5 * (t + t_next)
        v1 = model(x_t, t)
        x_mid = x_t + v1 * (0.5 * dt)
        v2 = model(x_mid, t_mid)
        return x_t + v2 * dt


class RK4Solver(Solver):
    """Classic 4th-order Runge-Kutta. 4 NFEs per step."""

    def step(self, model, x_t, t, t_next):
        dt = (t_next - t).view(-1, *([1] * (x_t.dim() - 1)))
        t_mid = 0.5 * (t + t_next)

        k1 = model(x_t, t)
        k2 = model(x_t + k1 * (0.5 * dt), t_mid)
        k3 = model(x_t + k2 * (0.5 * dt), t_mid)
        k4 = model(x_t + k3 * dt, t_next)

        return x_t + (k1 + 2 * k2 + 2 * k3 + k4) * (dt / 6)


SOLVERS: dict[str, type[Solver]] = {
    "euler": EulerSolver,
    "heun": HeunSolver,
    "rk2": RK2Solver,
    "rk4": RK4Solver,
}


def get_solver(name: str) -> Solver:
    """Get a solver instance by name."""
    if name not in SOLVERS:
        raise ValueError(f"Unknown solver '{name}'. Choose from: {list(SOLVERS.keys())}")
    return SOLVERS[name]()
