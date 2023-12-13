import torch
from torch.nn import functional as F

from src.training import Trainer


class DiffusionTrainer(Trainer):
    def __init__(self, model, learning_rate=1e-3, weight_decay=0.0):
        self.model = model
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=learning_rate,
                                          weight_decay=weight_decay)
        self.device = self.model.device
        self.patience = 15

    def compute_loss(self, batch, do_step=True):
        if do_step:
            self.model = self.model.train()
        else:
            self.model = self.model.eval()

        self.optimizer.zero_grad()
        x_0 = batch["output_image"].to(self.model.dtype)
        x_T = batch["input_image"].to(self.model.dtype)
        t = torch.randint(0, self.model.steps + 1, (x_0.size(0),),
                          device=self.model.device, dtype=torch.long)
        alpha = t.to(self.model.dtype)/self.model.steps
        x_t = (1 - alpha)*x_0 + alpha*x_T
        loss = F.l1_loss(x_0, self.model.x0(x_t, x_T, t))

        if do_step:
            loss.backward()
            self.optimizer.step()
        return loss
