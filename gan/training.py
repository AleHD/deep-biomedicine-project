import torch
import torch.optim as optim

from src.loss_functions import VGGPerceptualLoss
from src.training import Trainer

class GANTrainer(Trainer):
    def __init__(self, model, discriminator, device, learning_rate):
        """ 
        The Trainer need to receive the model and the device.
        """
        # Model is the generator
        self.model = model
        # discriminator to distinguish nosiy from sharp
        self.discriminator = discriminator
        self.device = device
        
        self.learning_rate = learning_rate
        # loss functions
        self.l1_loss = torch.nn.L1Loss()
        # more global loss fucntion
        self.perceptual_loss = VGGPerceptualLoss(resize=False).to(device)
        # adverserial loss function
        self.adverserial_loss = torch.nn.BCELoss()
        
        # initialize optimizers for generator and discriminator
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.d_optim = optim.Adam(self.discriminator.parameters(), lr=self.learning_rate)
    
    def compute_loss(self, data, do_step=True):
        
        if do_step:
            self.model.train()
            self.discriminator.train()
        else:
            self.model.eval()
            self.discriminator.eval()
            
        ## Training Discriminator
        output = self.model(data["input_image"])
        real_prob = self.discriminator(data["output_image"].float())
        fake_prob = self.discriminator(output)

        # Create the correct labels
        real_label = torch.ones_like(real_prob).to(self.device)
        fake_label = torch.zeros_like(fake_prob).to(self.device)
        # compute adverserial loss
        d_loss_real = self.adverserial_loss(real_prob, real_label)
        d_loss_fake = self.adverserial_loss(fake_prob, fake_label)
        # total loss
        d_loss = d_loss_real + d_loss_fake
        
        if do_step:
            # take step down gradient
            self.optimizer.zero_grad()
            self.d_optim.zero_grad()
            d_loss.backward()
            self.d_optim.step()

        ## Training Generator
        output = self.model(data["input_image"])
        fake_prob = self.discriminator(output)
        
        # compute losses
        # Do scaling of images, otherwise VGG_loss gives 0.0
        percep_loss = self.perceptual_loss((data["output_image"].float() + 1.0) / 2.0, (output + 1.0) / 2.0)
        L1_loss = self.l1_loss(output, data["output_image"].float())
        adversarial_loss = self.adverserial_loss(fake_prob, real_label)
        # L1 loss ~1 order of magnitude smaller
        g_loss = percep_loss + adversarial_loss + 10 * L1_loss
        if do_step:
            # step down gradient
            self.optimizer.zero_grad()
            self.d_optim.zero_grad()
            g_loss.backward()
            self.optimizer.step()
            
        # return generator loss
        return g_loss.item()
    