import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    """ 
    For more information about U-Net Architecture check the paper here.
    Link :- https://arxiv.org/abs/1505.04597
    """

    def __init__(self, filter_num, input_channels=1, output_channels=1):

        """ Constructor for UNet class.
        Parameters:
            filter_num: A list of number of filters (number of input or output channels of each layer).
            input_channels: Input channels for the network.
            output_channels: Output channels for the final network.
        """

        # Since UNet is a child class, we need to first call the __init__ function of its parent class
        super(UNet, self).__init__()

        # We set hyper-parameter padding and kernel size
        padding = 1
        ks = 3

        # Encoding Part of Network.
        # Hint: 
        # 1. For each block, you need two convolution layers and one maxpooling layer.
        # 2. You can check online (torch.nn) and try nn.Conv2d and nn.MaxPool2d.
        # Links!

        # Block 1
        self.conv1_1 = nn.Conv2d(input_channels, filter_num[0], kernel_size=ks, padding=padding)
        self.conv1_2 = nn.Conv2d(filter_num[0], filter_num[0], kernel_size=ks, padding=padding)
        self.maxpool1 = nn.MaxPool2d(2)

        # Block 2
        self.conv2_1 = nn.Conv2d(filter_num[0], filter_num[1], kernel_size=ks, padding=padding)
        self.conv2_2 = nn.Conv2d(filter_num[1], filter_num[1], kernel_size=ks, padding=padding)
        self.maxpool2 = nn.MaxPool2d(2)

        # Block 3
        self.conv3_1 = nn.Conv2d(filter_num[1], filter_num[2], kernel_size=ks, padding=padding)
        self.conv3_2 = nn.Conv2d(filter_num[2], filter_num[2], kernel_size=ks, padding=padding)
        self.maxpool3 = nn.MaxPool2d(2)

        # Block 4
        self.conv4_1 = nn.Conv2d(filter_num[2], filter_num[3], kernel_size=ks, padding=padding)
        self.conv4_2 = nn.Conv2d(filter_num[3], filter_num[3], kernel_size=ks, padding=padding)
        self.maxpool4 = nn.MaxPool2d(2)
        
        # Bottleneck Part of Network.
        self.conv5_1 = nn.Conv2d(filter_num[3], filter_num[4], kernel_size=ks, padding=padding)
        self.conv5_2 = nn.Conv2d(filter_num[4], filter_num[4], kernel_size=ks, padding=padding)

        # Decoding Part of Network.
        # Hint: 
        # 1. For each block, you need two convolution layers and one upsampling layer.
        # 2. You can check online and try nn.ConvTranspose2d for upsampling.
        # Links!

        # Block 4
        self.conv6_t = nn.ConvTranspose2d(filter_num[4], filter_num[3], 2, stride=2)
        self.conv6_1 = nn.Conv2d(filter_num[4], filter_num[3], kernel_size=ks, padding=padding)
        self.conv6_2 = nn.Conv2d(filter_num[3], filter_num[3], kernel_size=ks, padding=padding)

        # Block 3
        self.conv7_t = nn.ConvTranspose2d(filter_num[3], filter_num[2], 2, stride=2)
        self.conv7_1 = nn.Conv2d(filter_num[3], filter_num[2], kernel_size=ks, padding=padding)
        self.conv7_2 = nn.Conv2d(filter_num[2], filter_num[2], kernel_size=ks, padding=padding)

        # Block 2
        self.conv8_t = nn.ConvTranspose2d(filter_num[2], filter_num[1], 2, stride=2)
        self.conv8_1 = nn.Conv2d(filter_num[2], filter_num[1], kernel_size=ks, padding=padding)
        self.conv8_2 = nn.Conv2d(filter_num[1], filter_num[1], kernel_size=ks, padding=padding)

        # Block 1
        self.conv9_t = nn.ConvTranspose2d(filter_num[1], filter_num[0], 2, stride=2)
        self.conv9_1 = nn.Conv2d(filter_num[1], filter_num[0], kernel_size=ks, padding=padding)
        self.conv9_2 = nn.Conv2d(filter_num[0], filter_num[0], kernel_size=ks, padding=padding)

        # Output Part of Network.
        self.conv10 = nn.Conv2d(filter_num[0], output_channels, kernel_size=ks, padding=padding)

    def forward(self, x):
        """ 
        Forward propagation of the network.
        """

        # Encoding Part of Network.
        # Block 1
        conv1 = F.relu(self.conv1_1(x))
        conv1 = F.relu(self.conv1_2(conv1))
        pool1 = self.maxpool1(conv1)
        # Block 2
        conv2 = F.relu(self.conv2_1(pool1))
        conv2 = F.relu(self.conv2_2(conv2))
        pool2 = self.maxpool2(conv2)
        # Block 3
        conv3 = F.relu(self.conv3_1(pool2))
        conv3 = F.relu(self.conv3_2(conv3))
        pool3 = self.maxpool3(conv3)
        # Block 4
        conv4 = F.relu(self.conv4_1(pool3))
        conv4 = F.relu(self.conv4_2(conv4))
        pool4 = self.maxpool4(conv4)

        # Bottleneck Part of Network.
        conv5 = F.relu(self.conv5_1(pool4))
        conv5 = F.relu(self.conv5_2(conv5))

        # Decoding Part of Network.
        # Hint: Do not forget to concatnate in outputs from the encoder.

        # Block 4
        up6 = torch.cat((self.conv6_t(conv5), conv4), dim=1)
        conv6 = F.relu(self.conv6_1(up6))
        conv6 = F.relu(self.conv6_2(conv6))

        # Block 3
        up7 = torch.cat((self.conv7_t(conv6), conv3), dim=1)
        conv7 = F.relu(self.conv7_1(up7))
        conv7 = F.relu(self.conv7_2(conv7))

        # Block 2
        up8 = torch.cat((self.conv8_t(conv7), conv2), dim=1)
        conv8 = F.relu(self.conv8_1(up8))
        conv8 = F.relu(self.conv8_2(conv8))

        # Block 1
        up9 = torch.cat((self.conv9_t(conv8), conv1), dim=1)
        conv9 = F.relu(self.conv9_1(up9))
        output = F.relu(self.conv9_2(conv9))

        # Output Part of Network.
        # Hint: 
        # 1. Segmentation is to classify each pixel into foreground or backfround. 
        # 2. Sigmoid is a useful activate function for binary classification.
        
        output = self.conv10(conv9)

        return output
    
    def predict(self, x_T: torch.Tensor) -> torch.Tensor:
        return self(x_T.to(self.device)).to(x_T.device)
    
    @property
    def dtype(self):
        return next(p.dtype for p in self.parameters())

    @property
    def device(self):
        return next(p.device for p in self.parameters())
    

def get_pretrained(path: str) -> UNet:
    filter_num = [16,32,64,128,256]
    model = UNet(filter_num=filter_num)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.eval().requires_grad_(False)
