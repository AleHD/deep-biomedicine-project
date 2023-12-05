import torch
import torch.optim as optim

import numpy as np

class Trainer():
    def __init__(self, model, criterion, device):
        """ 
        The Trainer need to receive the model and the device.
        """

        self.model = model
        self.device = device

        # we use Dice-loss as our loss function in this exercise.
        # ToDo 0: Check the following links for more details of Dice loss:
        # 1. https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
        # 2. https://dev.to/_aadidev/3-common-loss-functions-for-image-segmentation-545o
        self.criterion = criterion

    def train(self, epochs, trainloader, mini_batch=None, learning_rate=0.001):

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
        loss_record = []

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
            epoch_loss = self._train_epoch(trainloader, mini_batch)

            # Collecting all epoch loss values for future visualization.
            loss_record.append(epoch_loss)

            # Reduce LR On Plateau
            self.scheduler.step(epoch_loss)
            if (epoch+1) % 10 == 0:
                # Training Logs printed.
                print(f'Epoch: {epoch+1:03d},  ', end='')
                print(f'Loss:{epoch_loss:.7f},  ', end='\n')

        return loss_record
    
    def _train_epoch(self, trainloader, mini_batch):
        """ Training each epoch.
        Parameters:
        1. trainloader: Training dataloader for the optimizer.
        2. mini_batch: The number of batches used during training.

        Returns:
            epoch_loss(float): Loss calculated for each epoch.
        """

        epoch_loss, batch_loss, batch_iteration = 0, 0, 0

        # Write a training loop. You can check exercise 1 for a standard training loop.
        for batch, data in enumerate(trainloader):

            # Keeping track how many iteration is happening.
            batch_iteration += 1

            # Loading data to device used.
            image = data['input_image'].to(self.device)
            mask = data['output_image'].to(self.device)

            # Clearing gradients of optimizer.
            self.optimizer.zero_grad()

            # Calculation predicted output using forward pass.
            output = self.model(image)

            # Calculating the loss value.
            # Hint: self.criterion
            loss_value = self.criterion(output, mask)
            # ToDo 6: Computing the gradients.
            loss_value.backward()

            # Optimizing the network parameters.
            self.optimizer.step()

            # Updating the running training loss
            epoch_loss += loss_value.item()
            batch_loss += loss_value.item()

            # Printing batch logs if any.
            if mini_batch:
                if (batch+1) % mini_batch == 0:
                    batch_loss = batch_loss / (mini_batch*trainloader.batch_size)
                    # print(f'Batch: {batch+1:02d},\tBatch Loss: {batch_loss:.7f}')
                    batch_loss = 0

        epoch_loss = epoch_loss/(batch_iteration*trainloader.batch_size)
        return epoch_loss

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

    def _psnr(self, predicted, target):
        """
        Predicted: the prediction from the model.
        Target: the groud truth.
        """
        mse = np.mean((predicted - target) ** 2) 
        if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                    # Therefore PSNR have no importance. 
            return 100
        max_pixel = 1   # minmaxed
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse)) 
        return psnr 