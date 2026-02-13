import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseMethod
from .solvers import Solver, EulerSolver, get_solver

class FlowMatching(BaseMethod):

    def __init__(
            self,
            model: nn.Module,
            device: torch.device,
            num_timesteps: int,
            solver: Solver | None = None,
    ):
        super().__init__(model, device)

        self.num_timesteps = num_timesteps
        self.solver = solver or EulerSolver()
        self.register_buffer(
            'timesteps',
            torch.linspace(0., 1., self.num_timesteps, device=device)
        )
    
    def forward_process(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = t.view(x_0.shape[0], *([1] * (x_0.dim() - 1)))
        return x_1 * t + (1 - t) * x_0

    def compute_loss(self, x_1: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:

        batch_size = x_1.shape[0]
        device = x_1.device

        t_idx = torch.randint(0, self.num_timesteps, (batch_size, ), device=device)
        t = self.timesteps[t_idx]

        x_0 = torch.randn_like(x_1)

        x_t = self.forward_process(x_0, x_1, t)

        v_pred = self.model(x_t, t)

        loss = F.mse_loss(v_pred, x_1 - x_0)

        metrics = {
            'loss': loss.item(),
            'mse': loss.item()
        }

        return loss, metrics
    
    @torch.no_grad()
    def reverse_process(self, x_t: torch.Tensor, t: torch.Tensor, t_next: torch.Tensor) -> torch.Tensor:
        return self.solver.step(self.model, x_t, t, t_next)
        
    @torch.no_grad()
    def sample(
        self, 
        batch_size: int, 
        image_shape: tuple[int, int, int], 
        num_steps: int | None = None, 
        return_trajectory: bool = False,
        show_progress: bool = False,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        
        self.eval_mode()
        device = self.device

        if num_steps is None:
            num_steps = self.num_timesteps

        x = torch.randn(batch_size, *image_shape, device=device)

        if num_steps < self.num_timesteps:
            timesteps = torch.linspace(0, 1, num_steps, device=device)
        else:
            timesteps = self.timesteps

        trajectory = []

        pbar = zip(timesteps[:-1], timesteps[1:])
        if show_progress:
            try:
                from tqdm.auto import tqdm
                pbar = tqdm(pbar, total=num_steps-1)
            except ImportError:
                pass

        for t, t_next in pbar:

            t_batch = torch.full((batch_size, ), t, device=device)
            t_next_batch = torch.full((batch_size, ), t_next, device=device)

            x = self.reverse_process(x, t_batch, t_next_batch)

            if return_trajectory:
                trajectory.append(torch.clamp(x, -1, 1))

        if return_trajectory:
            return x, trajectory
        return x
    
    def to(self, device: torch.device) -> "FlowMatching":
        super().to(device)
        self.device = device
        return self
    
    def state_dict(self) -> dict:
        state = super().state_dict()
        state["num_timesteps"] = self.num_timesteps
        return state
    
    @classmethod
    def from_config(cls, model: nn.Module, config: dict, device: torch.device, solver_override: str | None = None) -> "FlowMatching":

        flow_matching_config = config.get("cfm", config)
        solver_name = solver_override or flow_matching_config.get("solver", "euler")
        return FlowMatching(
            model=model,
            device=device,
            num_timesteps=flow_matching_config['num_timesteps'],
            solver=get_solver(solver_name),
        )