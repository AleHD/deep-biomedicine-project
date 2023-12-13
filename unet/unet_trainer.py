import torch
import sys
sys.path.insert(0,".")
import torch.optim as optim
import os
import numpy as np
from src.training import Trainer

class Unet_trainer(Trainer):
    def __init__(self, model, learning_rate, device ):
        """ 
        The Trainer need to receive the model and the device.
        """

        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Loss function we use in this exercise.
        self.criterion = torch.nn.L1Loss()
        # for early stopping
        self.patience = 10

    def compute_loss(self, data, do_step=True):
        if do_step:
            
            self.model.train()
        else:
            self.model.eval()  

        
        output = self.model(data["input_image"])  
        
          

         
        loss = self.criterion(output, data["output_image"])
        
        

        if do_step:
            # take step down gradient
            self.optimizer.zero_grad()
            
            # Computing the gradients.
            loss.backward()
        
            # Optimizing the network parameters.
            self.optimizer.step()
            
            
        
        return loss.item()




    