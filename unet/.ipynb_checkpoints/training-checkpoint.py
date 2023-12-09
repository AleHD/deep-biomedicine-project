import torch
import utils
import torch.optim as optim
import os
import numpy as np

class Trainer():
    def __init__(self, model, device):
        """ 
        The Trainer need to receive the model and the device.
        """

        self.model = model
        self.device = device

        # Loss function we use in this exercise.
        self.criterion = torch.nn.L1Loss()

    def train(self, epochs, trainloader, validationloader, mini_batch=None, learning_rate=0.002):

        """ 
        Train the model

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

        # ToDo: We choose Adam to be the optimizer.
        # Links
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Reducing LR on plateau feature to improve training.
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=0.85, patience=2, verbose=True)

        patience = 10
        best_val_loss = float('inf')
        best_model_state_dict = None
        counter_since_improvement = 0
        
        print('Starting Training Process')
        if os.path.exists('models/baseline.pth'):

            self.model.train()

            train_losses=[]

            # Epoch Loop
            for epoch in range(epochs):

                # Training a single epoch
                training_epoch_loss = self._train_epoch(trainloader, mini_batch)
                validation_epoch_loss = self._validate_epoch(validationloader, mini_batch)

                

                # Collecting all epoch loss values for future visualization.
                train_loss_record.append(training_epoch_loss)
                validation_loss_record.append(validation_epoch_loss)

                # Reduce LR On Plateau
                #self.scheduler.step(training_epoch_loss)

                # Training Logs printed.
                print(f'Epoch: {epoch+1:03d},  \n ', end='')
                print(f'Training Loss:{training_epoch_loss:.7f},  \n  ', end='')
                print(f'Validation Loss:{validation_epoch_loss:.7f},  \n  ', end='')

                if validation_epoch_loss < best_val_loss:
                    best_val_loss = validation_epoch_loss
                    best_model_state_dict = self.model.state_dict()
                    counter_since_improvement = 0
                else:
                    counter_since_improvement += 1


                # Early stopping check
                if counter_since_improvement >= patience:
                    print(f'Early stopping after {patience} epochs without improvement.')
                    break
                

            # Load the best model state dict and save the model
            self.model.load_state_dict(best_model_state_dict)
            torch.save(self.model, 'models/baseline_with_scheduler.pth')
            
                
            
            #save the trained model
            #torch.save(self.model, 'models/baseline.pth')

        else:
            self.model = torch.load('models/baseline.pth')
             

        return train_loss_record,validation_loss_record
    
    def _train_epoch(self, trainloader, mini_batch):
        """ Training each epoch.
        Parameters:
        1. trainloader: Training dataloader for the optimizer.
        2. mini_batch: The number of batches used during training.

        Returns:
            epoch_loss(float): Loss calculated for each epoch.
        """

        epoch_loss, batch_loss, batch_iteration = 0, 0, 0

        # ToDo: Write a training loop. You can check exercise 1 for a standard training loop.
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
            loss_value = self.criterion(output, mask)

            # Computing the gradients.
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
                    print(f'Batch: {batch+1:02d},\tBatch Loss: {batch_loss:.7f}')
                    batch_loss = 0



        epoch_loss = epoch_loss/(batch_iteration*trainloader.batch_size)
        return epoch_loss


        
       
    
    def _validate_epoch(self, validationloader, mini_batch):
        """ Training each epoch.
        Parameters:
        1. trainloader: Training dataloader for the optimizer.
        2. mini_batch: The number of batches used during training.

        Returns:
            epoch_loss(float): Loss calculated for each epoch.
        """

        epoch_loss, batch_loss, batch_iteration = 0, 0, 0

        # ToDo: Write a training loop. You can check exercise 1 for a standard training loop.
        for batch, data in enumerate(validationloader):

            # Keeping track how many iteration is happening.
            batch_iteration += 1

            # Loading data to device used.
            image = data['input_image'].to(self.device)
            mask = data['output_image'].to(self.device)

            

            # Clearing gradients of optimizer.

            

            # Calculation predicted output using forward pass.
            output = self.model(image)
            

            # Calculating the loss value.
            loss_value = self.criterion(output, mask)

            # Computing the gradients.

            # Optimizing the network parameters.

            # Updating the running training loss
            epoch_loss += loss_value.item()
            batch_loss += loss_value.item()

            # Printing batch logs if any.
            if mini_batch:
                if (batch+1) % mini_batch == 0:
                    batch_loss = batch_loss / (mini_batch*trainloader.batch_size)
                    print(f'Batch: {batch+1:02d},\tBatch Loss: {batch_loss:.7f}')
                    batch_loss = 0



        epoch_loss = epoch_loss/(batch_iteration*validationloader.batch_size)
        return epoch_loss

    def test(self, testloader, threshold=0.5):
        """ 
        To test the performance of model on testing dataset.

        Parameters:
        1. testloader: The testloader we create in Section.1 Dataset and Dataloader.
        2. Threshold: We want to segment our image to foreground and background and the output of each pixel is a number between 0 and 1. 
                      Thus, we need a threshold to decide is the pixel belongs to the foreground or background.

        Returns:
        mean_val_score: The mean Sørensen–Dice Coefficient for the whole test dataset.

        You do not need to change this function.
        """

        self.model.eval()

        test_data_indexes = testloader.sampler.indices[:]
        data_len = len(test_data_indexes)
        mean_val_score = 0

        testloader = iter(testloader)

        while len(test_data_indexes) != 0:

            data = next(testloader)
            index = int(data['index'])
            
            if index in test_data_indexes:
                test_data_indexes.remove(index)
            else:
                continue

            image = data['input_image'].view((1, 1, 1024, 1024)).to(self.device)
            mask = data['output_image']

            mask_pred = self.model(image).cpu()

            mask_pred = (mask_pred > threshold)
            mask_pred = mask_pred.numpy()
            
            mask = np.resize(mask, (1, 1024, 1024))
            mask_pred = np.resize(mask_pred, (1, 1024, 1024))
            
            mean_val_score += self._dice_coefficient(mask_pred, mask)

        mean_val_score = mean_val_score / data_len

        return mean_val_score

    def predict(self, data, threshold=0.5):
        """ 
        Calculate the output mask on a single input data.

        You do not need to change this function.
        """
        self.model.eval()
        image = data['input_image'].numpy()
        mask = data['output_image'].numpy()

        image_tensor = torch.Tensor(data['image'])
        image_tensor = image_tensor.view((-1, 1, 1024, 1024)).to(self.device)

        output = self.model(image_tensor).detach().cpu()
        output = (output > threshold)
        output = output.numpy()

        image = np.resize(image, (1024, 1024))
        mask = np.resize(mask, (1024, 1024))
        output = np.resize(output, (1024, 1024))
        score = self._dice_coefficient(output, mask)

        return image, mask, output, score

    def _dice_coefficient(self, predicted, target):
        """
        Calculates the Sørensen–Dice Coefficient for a single sample.

        Predicted: the prediction from the model.
        Target: the groud truth.

        You do not need to change this function.
        """
        smooth = 1
        product = np.multiply(predicted, target)
        intersection = np.sum(product)
        coefficient = (2*intersection + smooth) / (np.sum(predicted) + np.sum(target) + smooth)
        
        return coefficient