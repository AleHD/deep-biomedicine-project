import torch, gc
import torch.optim as optim
import numpy as np

from src.training import Trainer

class MWCNNTrainer(Trainer):
    def __init__(self, model, learning_rate, device, local_criterion, global_criterion=None, scale=None):
        """ 
        Initate trainer
        Parameters:
        1. model: Model to train.
        2. learning_rate
        3. device: Device to train on (e.g. cuda:0)
        4. local_criterion: local loss function such as L1 or L2 loss.
        5. global_criterion: global loss function, such as fft or perception loss.
        6. scale: scaling of global loss.
        """

        self.model = model
        self.device = device
        # initialize optimizer        
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        # like L1 loss
        self.local_criterion = local_criterion
        # like vgg loss or fft loss
        if global_criterion:
            self.global_criterion = global_criterion.to(device)
            self.scale = scale
        else:
            self.global_criterion = None
                        
    def compute_loss(self, data, do_step=True):
        
        if do_step:
            self.model.train()
        else:
            self.model.eval()
        # Calculation predicted output using forward pass.
        output = self.model(data["input_image"])

        # Calculating the loss value.
        loss_value = self.local_criterion(output, data["output_image"])
        if self.global_criterion:
            loss_value += self.scale * self.global_criterion(output, data["output_image"])

        if do_step:
            # Clearing gradients of optimizer.
            self.optimizer.zero_grad()
            # Computing the gradients.
            loss_value.backward()
            # Optimizing the network parameters.
            self.optimizer.step()

        return loss_value.item()
