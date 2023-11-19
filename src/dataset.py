from torch.utils.data import Dataset
from torchvision.transforms import v2

from PIL import Image

import os

'''
Here we try to build a Dataset class.
A common way is to define a Dataset class inherits from torch.utils.data.Dataset class.
Then, we override:
1. __init__: initialization
2. __getitem__: get a sample
3. __len__: get the length (number of samples) of the dataset
'''

class ImageDataset(Dataset):
    """ 
    A TumorDataset class.
    The object of this class represents tumor dataset.
    TumorDataset inherits from torch.utils.data.Dataset class.
    """

    def __init__(self, root_dir, dataset, image_size=1024):

        """
        Inputs:
        1. root_dir: Directory with all the images.
        """

        # set root directory to root_dir
        self.root_dir = root_dir
        self.dataset = dataset
        
        # set image size
        self.image_size= image_size

        # 1. Transform the data to the form that we need. E.g. use ToTensor to convert the datatype to Tensor
        # 2. Augment the data. E.g. flip and rotation.
        self.default_transformation = v2.Compose([
            v2.RandomCrop(size=image_size),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            # v2.RandomRotation(degrees=180),
            v2.ToTensor(),
        ])

    def __getitem__(self, index):
        """
        __getitem__ is the function we need to override.
        Given an index, this function is to output the sample with that index.

        Here, a sample is a dictionary which contains:
        1. 'index': Index of the image.
        2. 'image': Contains the tumor image torch.Tensor.
        3. 'mask' : Contains the mask image torch.Tensor.
        """

        # load image and mask, give each of them an index, and return image, mask, and index.
        input_image_name = os.path.join(self.root_dir, self.dataset + '_tile' + str(index)+'_mip.tif')
        output_image_name = os.path.join(self.root_dir, self.dataset + '_tile' + str(index)+'_edof.tif')
        
        input_image = Image.open(input_image_name)
        output_image = Image.open(output_image_name)
        # apply transform to both input and output
        input_image, output_image = self.default_transformation(input_image, output_image)
        sample = {'index': int(index), 'input_image': input_image, 'output_image': output_image}

        return sample

    def __len__(self):
        """
        Returns the size of the dataset.
        """

        # Get the size of the datasets (The number of samples in the dataset.)
        # Hint: The folder we provide contains samples and their mask, which means we have two images for each samples.
        
        size_of_dataset = int(
            len(
                [name for name in os.listdir(self.root_dir) if os.path.isfile(os.path.join(self.root_dir, name))]
            ) / 2   # imput and output
        )


        return size_of_dataset