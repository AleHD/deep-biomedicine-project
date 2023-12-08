import torch
import torch.optim as optim

import numpy as np


class SchedulerWrapper:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.optimizer = self.scheduler.optimizer

    def step(self, loss=None):
        self.scheduler.step()


class Trainer():
    def __init__(self, model, learning_rate, device, closure=None):
        """ 
        The Trainer need to receive the model and the device.
        """

        self.model = model
        self.device = device
        self.learning_rate = learning_rate

    def train(self, epochs, trainloader, validation_loader,
              scheduler="plateau", warmup=0.0):

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
        train_loss_record, validation_loss_record = [], []
        
        # Reducing LR on plateau feature to improve training.
        warmup = int(warmup*epochs)
        if scheduler == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, factor=0.85, patience=2, verbose=True)
        elif scheduler == "cosine":
            self.scheduler = SchedulerWrapper(optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, epochs - warmup, 0.01*self.learning_rate
            ))
        else:
            raise KeyError(f"Unknown scheduler {scheduler}")

        if warmup > 0:
            self.scheduler = SchedulerWrapper(optim.lr_scheduler.ChainedScheduler([
                optim.lr_scheduler.LinearLR(self.optimizer, 0.1, total_iters=warmup),
                self.scheduler
            ]))
        
        print('Starting Training Process')

        self.model.train()

        # Epoch Loop
        try:
            for epoch in range(epochs):

                # Training a single epoch
                epoch_loss = self._train_epoch(trainloader)
                validation_loss = self._validate_epoch(validation_loader)
        
                # Collecting all epoch loss values for future visualization.
                train_loss_record.append(epoch_loss)
                validation_loss_record.append(validation_loss)
                # Reduce LR On Plateau
                self.scheduler.step(epoch_loss)

                # Training Logs printed.
                print(f'Epoch: {epoch+1:03d},  ', end='', flush=True)
                print(f'Train Loss:{epoch_loss:.7f},  ', end='', flush=True)
                print(f'Validation Loss:{validation_loss:.7f},  ', end='', flush=True)
        except KeyboardInterrupt:
            print("Training interrupted by user!")

        return train_loss_record, validation_loss_record
    
    def _train_epoch(self, trainloader):
        """ Training each epoch.
        Parameters:
        1. trainloader: Training dataloader for the optimizer.
        2. mini_batch: The number of batches used during training.

        Returns:
            epoch_loss(float): Loss calculated for each epoch.
        """

        epoch_loss, batch_iteration = 0, 0

        # Write a training loop. You can check exercise 1 for a standard training loop.
        for data in trainloader:

            # Keeping track how many iteration is happening.
            batch_iteration += 1

            # Loading data to device used.
            data = {"index": data["index"],
                    "input_image": data["input_image"].to(self.device),
                    "output_image": data["output_image"].to(self.device)}

            epoch_loss += self.compute_loss(data, do_step=True)

        epoch_loss = epoch_loss/(batch_iteration*trainloader.batch_size)
        return epoch_loss
    
    def _validate_epoch(self, validation_loader):
        """ Training each epoch.
        Parameters:
        1. validation_loader: Validation dataloader.

        Returns:
            validation loss(float): Loss calculated for each epoch.
        """

        validation_loss, batch_iteration = 0, 0

        # Write a training loop. You can check exercise 1 for a standard training loop.
        for data in validation_loader:

            # Keeping track how many iteration is happening.
            batch_iteration += 1

            # Loading data to device used.
            data = {"index": data["index"],
                    "input_image": data["input_image"].to(self.device),
                    "output_image": data["output_image"].to(self.device)}
            # Dont calculate gradient
            self.compute_loss(data, do_step=False)
            # Clearing gradients of optimizer.
            validation_loss += self.optimizer.zero_grad()

        validation_loss = validation_loss/(batch_iteration*validation_loader.batch_size)
        return validation_loss
    

    def test(self, testloader):
        """ 
        To test the performance of model on testing dataset.

        Parameters:
        1. testloader: The testloader we create in Section.1 Dataset and Dataloader.
        2. Threshold: We want to segment our image to foreground and background and the output of each pixel is a number between 0 and 1. 
                      Thus, we need a threshold to decide is the pixel belongs to the foreground or background.

        Returns:
        mean_val_score: The mean PSNR for the whole test dataset.

        You do not need to change this function.
        """

        self.model.eval()

        test_data_indexes = testloader.sampler.indices[:]
        data_len = len(test_data_indexes)
        mean_val_score = 0
        mean_mse_score = 0

        testloader = iter(testloader)

        while len(test_data_indexes) != 0:

            data = next(testloader)
            index = int(data['index'])
            
            if index in test_data_indexes:
                test_data_indexes.remove(index)
            else:
                continue

            input_image = data['input_image'].to(self.device)
            output_image = data['output_image'].numpy()

            pred = self.model(input_image).detach().cpu()
            pred = pred.numpy()
            
            mean_val_score += self._psnr(pred, output_image)
            mean_mse_score += ((pred - output_image)**2).mean(axis=None)
        
        mean_val_score = mean_val_score / data_len
        mean_mse_score = mean_mse_score / data_len
        return mean_val_score, mean_mse_score

    def predict(self, data):
        """ 
        Calculate the output mask on a single input data.
        """
        self.model.eval()
        input_image = data['input_image'].to(self.device)
        output_image = data['output_image'].squeeze().numpy()
        # add dimension if needed, expected shape for model is [batch, 1, x,y]
        if len(input_image.shape) < 4:
            input_image = input_image[None,:,:,:]
        pred = self.model(input_image).detach().cpu().squeeze().numpy()

        input_image = input_image.detach().cpu().squeeze().numpy()
        
        original_score = self._psnr(output_image, input_image)
        improved_score = self._psnr(output_image, pred)

        return input_image, pred, output_image, original_score, improved_score


    def create_optimizers(self, learning_rate=1e-3, weightdecay=0):
        # function to initialize optimizers
        pass
    
    def compute_loss(self, data, do_step=True):
        # function to calculate loss
        # If do_step is True, step down the gradient
        pass
    
