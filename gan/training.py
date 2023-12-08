import torch, gc
import torch.optim as optim

import numpy as np


class Trainer():
    def __init__(self, generator, discriminator, local_criterion, global_criterion, device):
        """ 
        The Trainer need to receive the model and the device.
        """

        self.generator = generator
        self.discriminator = discriminator
        self.device = device

        # loss functions
        # like L1 loss
        self.local_criterion = local_criterion.to(device)
        # like vgg loss or fft loss
        self.global_criterion = global_criterion.to(device)
        # adverserial loss function
        self.adverserial_loss = torch.nn.BCELoss()

    def train(self, epochs, trainloader, validationloader, mini_batch=None, learning_rate=1e-4):

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

        self.d_optim = optim.Adam(self.discriminator.parameters(), lr=learning_rate)
        self.g_optim = optim.Adam(self.generator.parameters(), lr=learning_rate)
        
        # Reducing LR, on plateau gives memory issues, so chose this simpler one
        self.scheduler = optim.lr_scheduler.StepLR(self.g_optim, step_size=10, gamma = 0.5)
        
        print('Starting Training Process')

        # set to training mode
        self.generator.train()
        self.discriminator.train()

        # Epoch Loop
        for epoch in range(epochs):

            # Training a single epoch
            epoch_g_loss, epoch_d_loss = self._train_epoch(trainloader)
            # Collecting all epoch loss values for future visualization.
            train_loss_record.append((epoch_g_loss, epoch_d_loss))
            # get validation results
            epoch_g_loss, epoch_d_loss = _validate_epoch(self, validationloader)
            validation_loss_record.append((epoch_g_loss, epoch_d_loss))

            # Reduce LR On Plateau
            self.scheduler.step(epoch_g_loss)

            if (epoch+1) % 10 == 0 or epoch == 0:
                # Training Logs printed.
                print(f'Epoch: {epoch+1:03d},  ', end='')
                print(f'G Loss:{epoch_g_loss:.7f}, D Loss:{epoch_d_loss:.7f}, ', end='\n')
            
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

        epoch_g_loss, epoch_d_loss, batch_iteration = 0, 0, 0

        # Write a training loop. You can check exercise 1 for a standard training loop.
        for _, data in enumerate(trainloader):
            
            # Keeping track how many iteration is happening.
            batch_iteration += 1
            # Loading data to device used.
            noisy = data['input_image'].to(self.device)
            sharp = data['output_image'].float().to(self.device)
                        
            ## Training Discriminator
            output = self.generator(noisy)
            real_prob = self.discriminator(sharp)
            fake_prob = self.discriminator(output)

            real_label = torch.ones_like(real_prob).to(self.device)
            fake_label = torch.zeros_like(fake_prob).to(self.device)
            # compute adverserial loss
            d_loss_real = self.adverserial_loss(real_prob, real_label)
            d_loss_fake = self.adverserial_loss(fake_prob, fake_label)
            
            d_loss = d_loss_real + d_loss_fake
            # take step down gradient
            self.g_optim.zero_grad()
            self.d_optim.zero_grad()
            d_loss.backward()
            self.d_optim.step()

            ## Training Generator
            output = self.generator(noisy)
            fake_prob = self.discriminator(output)
            
            # compute losses
            # Do scaling of images, otherwise VGG_loss gives 0.0
            percep_loss = self.global_criterion((sharp + 1.0) / 2.0, (output + 1.0) / 2.0)
            L1_loss = self.local_criterion(output, sharp)
            adversarial_loss = self.adverserial_loss(fake_prob, real_label)
            # L1 loss ~1 order of magnitude smaller
            g_loss = percep_loss + adversarial_loss + 10 * L1_loss
            # step down gradient
            self.g_optim.zero_grad()
            self.d_optim.zero_grad()
            g_loss.backward()
            self.g_optim.step()

            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()

        epoch_g_loss = epoch_g_loss/(batch_iteration*trainloader.batch_size)
        epoch_d_loss = epoch_d_loss/(batch_iteration*trainloader.batch_size)
        return epoch_g_loss, epoch_d_loss


def _validate_epoch(self, validationloader):
        """ Training each epoch.
        Parameters:
        1. validationloader: Validation dataloader.

        Returns:
            epoch_loss(float): Loss calculated for each epoch.
        """

        epoch_g_loss, epoch_d_loss, batch_iteration = 0, 0, 0

        # set models to eval mode
        self.generator.eval()
        self.discriminator.eval()

        for _, data in enumerate(validationloader):
            
            # Keeping track how many iteration is happening.
            batch_iteration += 1
            # Loading data to device used.
            noisy = data['input_image'].to(self.device)
            sharp = data['output_image'].float().to(self.device)
                        
            ## validating Discriminator
            output = self.generator(noisy)
            real_prob = self.discriminator(sharp)
            fake_prob = self.discriminator(output)

            real_label = torch.ones_like(real_prob).to(self.device)
            fake_label = torch.zeros_like(fake_prob).to(self.device)
            # compute adverserial loss
            d_loss_real = self.adverserial_loss(real_prob, real_label)
            d_loss_fake = self.adverserial_loss(fake_prob, fake_label)
            
            d_loss = d_loss_real + d_loss_fake

            ## validating Generator
            output = self.generator(noisy)
            fake_prob = self.discriminator(output)
            
            # compute losses
            # Do scaling of images, otherwise VGG_loss gives 0.0
            percep_loss = self.global_criterion((sharp + 1.0) / 2.0, (output + 1.0) / 2.0)
            L1_loss = self.local_criterion(output, sharp)
            adversarial_loss = self.adverserial_loss(fake_prob, real_label)
            # L1 loss ~1 order of magnitude smaller
            g_loss = percep_loss + adversarial_loss + 10 * L1_loss

            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()

        # set models to training mode again
        self.generator.train()
        self.discriminator.train()
        epoch_g_loss = epoch_g_loss/(batch_iteration*validationloader.batch_size)
        epoch_d_loss = epoch_d_loss/(batch_iteration*validationloader.batch_size)
        return epoch_g_loss, epoch_d_loss
