import torch, gc
import torch.optim as optim

import numpy as np

class Trainer():
    def __init__(self, model, local_criterion, device, global_criterion=None, clear_cache=False):
        """ 
        The Trainer need to receive the model and the device.
        """

        self.model = model
        self.device = device        

        # To clear cache after each epoch
        self.clear_cache = clear_cache

        # like L1 loss
        self.local_criterion = local_criterion
        # like vgg loss or fft loss
        if global_criterion:
            self.global_criterion = global_criterion.to(device)
        else:
            self.global_criterion = None

    def train(self, epochs, trainloader, validationloader, mini_batch=None, learning_rate=0.001):

        """ 
        Train the model.

        Parameters:
        1. epochs: Number of epochs for the training session.
        2. trainloader: Training dataloader.
        3. mini_batch: The number of batches used during training.
        4. learning_rate: Learning rate for optimizer.
        
        Return:
        1. history(dict): 'train_loss': List of loss at every epoch.
        """

        # For recording the loss value.
        train_loss_record = []
        validation_loss_record = []


        # ToDo 1: We choose Adam to be the optimizer.
        # Link to all the optimizers in torch.optim: https://pytorch.org/docs/stable/optim.html
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Reducing LR on plateau feature to improve training.
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=0.5, patience=1, verbose=False)
        
        print('Starting Training Process')

        self.model.train()

        # Epoch Loop
        for epoch in range(epochs):

            # Training a single epoch
            epoch_loss = self._train_epoch(trainloader)
            # Collecting all epoch loss values for future visualization.
            train_loss_record.append(epoch_loss)

            validation_epoch_loss = self._validate_epoch(validationloader)
            validation_loss_record.append(validation_epoch_loss)

            # Reduce LR On Plateau
            self.scheduler.step(epoch_loss)
            if (epoch+1) % 10 == 0 or epoch == 0:
                # Training Logs printed.
                print(f'Epoch: {epoch+1:03d},  ', end='')
                print(f'Loss:{epoch_loss:.7f},  ', end='\n')
            
            gc.collect()
            torch.cuda.empty_cache()

        return train_loss_record, validation_loss_record
    
    def _train_epoch(self, trainloader):
        """ Training each epoch.
        Parameters:
        1. trainloader: Training dataloader for the optimizer.

        Returns:
            epoch_loss(float): Loss calculated for each epoch.
        """

        epoch_loss, batch_iteration = 0, 0

        # Write a training loop. You can check exercise 1 for a standard training loop.
        for _, data in enumerate(trainloader):

            # Keeping track how many iteration is happening.
            batch_iteration += 1

            # Loading data to device used.
            noisy = data['input_image'].to(self.device)
            sharp = data['output_image'].to(self.device)

            # Clearing gradients of optimizer.
            self.optimizer.zero_grad()

            # Calculation predicted output using forward pass.
            output = self.model(noisy)

            # Calculating the loss value.
            loss_value = self.local_criterion(output, sharp)
            if self.global_criterion:
                loss_value += 0.001 * self.global_criterion(output, sharp)

            # Computing the gradients.
            loss_value.backward()

            # Optimizing the network parameters.
            self.optimizer.step()

            # Updating the running training loss
            epoch_loss += loss_value.item()

        epoch_loss = epoch_loss/(batch_iteration*trainloader.batch_size)
        return epoch_loss


    def _validate_epoch(self, validationloader):
        """ Training each epoch.
        Parameters:
        1. validationloader: Validation dataloader.

        Returns:
            epoch_loss(float): Loss calculated for each epoch.
        """

        epoch_loss, batch_iteration = 0, 0

        # set models to eval mode
        self.model.eval()

        for _, data in enumerate(validationloader):
            
            # Keeping track how many iteration is happening.
            batch_iteration += 1
            
            # Loading data to device used.
            noisy = data['input_image'].to(self.device)
            sharp = data['output_image'].to(self.device)

            # Clearing gradients of optimizer.
            self.optimizer.zero_grad()

            # Calculation predicted output using forward pass.
            output = self.model(noisy)

            # Calculating the loss value.
            loss_value = self.local_criterion(output, sharp)
            if self.global_criterion:
                loss_value += 0.001 * self.global_criterion(output, sharp)

            epoch_loss += loss_value.item()
            
        # set model to training mode again
        self.model.train()

        epoch_loss = epoch_loss/(batch_iteration*validationloader.batch_size)
        return epoch_loss
