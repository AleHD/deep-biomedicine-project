import torch
from torch import nn
from denoising_diffusion_pytorch import Unet
from torch.nn import functional as F


class DiffusionModel(nn.Module):
    def __init__(self, model: Unet, steps: int = 5, objective="x0"):
        super().__init__()
        self.model = model
        self.steps = steps
        self.objective = objective

    def x0(self, x_t, x_T, t):
        if self.objective == "x0":  # x_0 = out(x_t)
            return self.model(x_t, t)
        if self.objective == "delta":  # x_0 = x_T + out(x_t) 
            return x_T + self.model(x_t, t)
        raise ValueError(f"Invalid objective {self.objective}")

    def forward(self, x_T: torch.Tensor) -> torch.Tensor:
        x_t = x_T = x_T.to(self.dtype)
        for t in reversed(range(1, self.steps + 1)):
            t = torch.full((x_T.size(0),), t, device=x_T.device, dtype=torch.long)
            alpha = t.view(-1, 1, 1, 1).to(self.dtype)/self.steps
            x_t = (1 - alpha)*self.x0(x_t, x_T, t) + alpha*x_T
            # alpha_tminus1 = (t - 1).to(self.dtype)/self.steps
            # out = self.x0(x_t, x_T, t)
            # xtilde_t = (1 - alpha_t)*out + alpha_t*x_T
            # xtilde_tminus1 = (1 - alpha_tminus1)*out + alpha_tminus1*x_T
            # x_t = x_t - xtilde_t + xtilde_tminus1
            # x_t = (1 - alpha_t)*self.x0(x_t, x_T, t) + alpha_t*x_T
            x_t = x_t.clip(0, 1)
        return x_t

    def predict(self, x_T: torch.Tensor) -> torch.Tensor:
        return self(x_T.to(self.device)).to(x_T.device)

    def closure(self, batch: dict) -> torch.Tensor:
        x_0 = batch["output_image"].to(self.dtype)
        x_T = batch["input_image"].to(self.dtype)
        t = torch.randint(0, self.steps + 1, (x_0.size(0),), device=self.device, dtype=torch.long)
        alpha = t.to(self.dtype)/self.steps
        x_t = (1 - alpha)*x_0 + alpha*x_T
        return F.l1_loss(x_0, self.x0(x_t, x_T, t))
        # return F.mse_loss(x_0, self.x0(x_t, x_T, t))

    @property
    def dtype(self):
        return next(p.dtype for p in self.parameters())

    @property
    def device(self):
        return next(p.device for p in self.parameters())
