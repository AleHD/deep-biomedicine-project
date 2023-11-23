import torch
import torch.nn as nn
import torch.nn.functional as F


class MWCNN(nn.Module):
    """ 
    For more information about U-Net Architecture check the paper here.
    Link :- https://arxiv.org/abs/1505.04597
    """

    def __init__(self, n_feats, down_sample, up_sample, input_channels=1, output_channels=1):

        """ Constructor for UNet class.
        Parameters:
            filter_num: A list of number of filters (number of input or output channels of each layer).
            down_sample: func
            up_sample: func
            input_channels: Input channels for the network.
            output_channels: Output channels for the final network.
        """

        # Since UNet is a child class, we need to first call the __init__ function of its parent class
        super(MWCNN, self).__init__()

        # We set hyper-parameter padding and kernel size
        padding = 1
        ks = 3

        # Encoding Part of Network.
        # Hint: 
        # 1. For each block, you need two convolution layers and one maxpooling layer.
        # 2. input_channels is the number of the input channels. The filter_num is a list of number of filters. 
        #    For example, in block 1, the input channels should be input_channels, and the output channels should be filter_num[0];
        #    in block 2, the input channels should be filter_num[0], and the output channels should be filter_num[1]
        # 3. torch.nn contains the layers you need. You can check torch.nn from link: https://pytorch.org/docs/stable/nn.html

        # Block 1
        self.conv1_1 = nn.Conv2d(input_channels, n_feats, padding=padding, kernel_size=ks)
        self.conv1_2 = nn.Conv2d(n_feats, n_feats, padding=padding, kernel_size=ks)
        self.downsample1 = down_sample

        # Block 2
        self.conv2_1 = nn.Conv2d(n_feats * 4, n_feats * 2, padding=padding, kernel_size=ks)
        self.conv2_2 = nn.Conv2d(n_feats * 2, n_feats * 2, padding=padding, kernel_size=ks)
        self.downsample2 = down_sample

        # Block 3
        self.conv3_1 = nn.Conv2d(n_feats * 8, n_feats * 4, padding=padding, kernel_size=ks)
        self.conv3_2 = nn.Conv2d(n_feats * 4, n_feats * 4, padding=padding, kernel_size=ks)
        self.downsample3 = down_sample

        # Block 4
        self.conv4_1 = nn.Conv2d(n_feats * 16, n_feats * 8, padding=padding, kernel_size=ks)
        self.conv4_2 = nn.Conv2d(n_feats * 8, n_feats * 8, padding=padding, kernel_size=ks)
        self.downsample4 = down_sample
        
        # Bottleneck Part of Network.
        # Hint: 
        # 1. You only need two convolution layers.
        self.conv5_1 = nn.Conv2d(n_feats * 32, n_feats * 32, padding=padding, kernel_size=ks)
        self.conv5_2 = nn.Conv2d(n_feats * 32, n_feats * 32, padding=padding, kernel_size=ks)

        # Decoding Part of Network.
        # Hint: 
        # 1. For each block, you need one upsample+convolution layer and two convolution layers.
        # 2. output_channels is the number of the output channels. The filter_num is a list of number of filters. 
        #    However, we need to use it reversely.
        #    For example, in block 4 of decoder, the input channels should be filter_num[4], and the output channels should be filter_num[3];
        #    in Output Part of Network, the input channels should be filter_num[0], and the output channels should be output_channels.
        # 3. torch.nn contains the layers you need. You can check torch.nn from link: https://pytorch.org/docs/stable/nn.html
        # 4. Using nn.ConvTranspose2d is one way to do upsampling and convolution at the same time.

        # Block 4
        self.conv6_up = up_sample
        self.conv6_1 = nn.Conv2d(n_feats * 8, n_feats * 8, padding=padding, kernel_size=ks)
        self.conv6_2 = nn.Conv2d(n_feats * 8, n_feats * 16, padding=padding, kernel_size=ks)
        
        # Block 3
        self.conv7_up = up_sample
        self.conv7_1 = nn.Conv2d(n_feats * 4, n_feats * 4, padding=padding, kernel_size=ks)
        self.conv7_2 = nn.Conv2d(n_feats * 4, n_feats * 8, padding=padding, kernel_size=ks)
        
        # Block 2
        self.conv8_up = up_sample
        self.conv8_1 = nn.Conv2d(n_feats * 2, n_feats * 2, padding=padding, kernel_size=ks)
        self.conv8_2 = nn.Conv2d(n_feats * 2, n_feats * 4, padding=padding, kernel_size=ks)
        
        # Block 1
        self.conv9_up = up_sample
        self.conv9_1 = nn.Conv2d(n_feats, n_feats, padding=padding, kernel_size=ks)
        self.conv9_2 = nn.Conv2d(n_feats, n_feats, padding=padding, kernel_size=ks)

        # Output Part of Network.
        self.conv10 = nn.Conv2d(n_feats, output_channels, padding=padding, kernel_size=ks)

    def forward(self, x):
        """ 
        Forward propagation of the network.
        """

        # Encoding Part of Network.
        # Hint: Do not forget to add activate function, e.g. ReLU, between to convolution layers. (same for the bottlenect and decoder)

        #   Block 1
        conv1 = F.relu(self.conv1_1(x))
        conv1 = F.relu(self.conv1_2(conv1))
        pool1 = self.downsample1(conv1)
        #   Block 2
        conv2 = F.relu(self.conv2_1(pool1))
        conv2 = F.relu(self.conv2_2(conv2))
        pool2 = self.downsample2(conv2)
        #   Block 3
        conv3 = F.relu(self.conv3_1(pool2))
        conv3 = F.relu(self.conv3_2(conv3))
        pool3 = self.downsample3(conv3)
        #   Block 4
        conv4 = F.relu(self.conv4_1(pool3))
        conv4 = F.relu(self.conv4_2(conv4))
        pool4 = self.downsample4(conv4) 
        # Bottleneck Part of Network.
        conv5 = F.relu(self.conv5_1(pool4))
        conv5 = F.relu(self.conv5_2(conv5))
        # Decoding Part of Network.
        # Hint: 
        # 1. Do not forget to concatnate the outputs from the previous decoder and the corresponding encoder.
        # 2. You can try torch.cat() to concatnate the tensors.
        
        #   Block 4      
        # up6 = torch.cat((self.conv6_up(conv5), conv4), dim=1)
        up6 = self.conv6_up(conv5) + conv4
        conv6 = F.relu(self.conv6_1(up6))
        conv6 = F.relu(self.conv6_2(conv6))
        #   Block 3
        up7 =  self.conv7_up(conv6) + conv3
        conv7 = F.relu(self.conv7_1(up7))
        conv7 = F.relu(self.conv7_2(conv7))
        #   Block 2
        up8 =  self.conv8_up(conv7) + conv2
        conv8 = F.relu(self.conv8_1(up8))
        conv8 = F.relu(self.conv8_2(conv8))
        #   Block 1
        up9 = self.conv9_up(conv8) +  conv1
        conv9 = F.relu(self.conv9_1(up9))
        conv9 = F.relu(self.conv9_2(conv9))
        # Output Part of Network.
        # Hint: 
        # 1. Our task is a binary segmentation, and it is to classify each pixel into foreground or backfround. 
        # 2. Sigmoid is a useful activate function for binary classification.
        
        output = F.sigmoid(self.conv10(conv9))

        return output
