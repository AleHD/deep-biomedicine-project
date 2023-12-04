from torch.utils.data import Dataset
from torchvision.transforms import v2
import torch
import numpy as np
from os import listdir
from os.path import isfile, join

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

def get_files(directory, keyword):
    """Returns all files in directory if the file contains the keyword."""
    file_list = []
    for file in listdir(directory):
        if isfile(join(directory, file)) and keyword in file:
            file_list.append(join(directory, file))
    return file_list


class ImageDataset(Dataset):
    """ 
    A TumorDataset class.
    The object of this class represents tumor dataset.
    TumorDataset inherits from torch.utils.data.Dataset class.
    """

    def __init__(self, root_dir, datasets, normalize="standard", image_size=1024):

        """
        Inputs:
        1. root_dir: Directory with all the images.
        2.
        3. normalize: standard, minmax or none, normalization method
        """

        # set root directory to root_dir
        self.root_dir = root_dir
        self.normalize = normalize
        # Get edof and mip file names
        self.mip_files = []
        self.edof_files = []
        for dataset in datasets:
            directory = join(root_dir, dataset)
            self.mip_files += get_files(directory, 'mip')
            self.edof_files += get_files(directory, 'edof')

        
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
            v2.ToDtype(torch.float32),
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
        # input_image_name = os.path.join(self.root_dir, self.dataset + '_tile' + str(index)+'_mip.tif')
        # output_image_name = os.path.join(self.root_dir, self.dataset + '_tile' + str(index)+'_edof.tif')
        input_image_name = self.mip_files[index]
        output_image_name = self.edof_files[index]
        input_image = Image.open(input_image_name)
        output_image = Image.open(output_image_name)
        # apply transform to both input and output
        input_image, output_image = self.default_transformation(input_image, output_image)
        # normalize input
        if self.normalize == "standard":
            input_image = standardize(input_image)
            output_image = standardize(output_image)
        elif self.normalize == "minmax":
            input_image = minmax(input_image)
            output_image = minmax(output_image)
        elif self.normalize == "percentile":
            input_image = percentile(input_image, low=0, up=100)
            output_image = percentile(output_image, low=0, up=100)
        
        #create sample to return
        sample = {'index': int(index), 'input_image': input_image, 'output_image': output_image}

        return sample

    def __len__(self):
        """
        Returns the size of the dataset.
        """

        # Get the size of the datasets (The number of samples in the dataset.)
        # Hint: The folder we provide contains samples and their mask, which means we have two images for each samples.
        
        # size_of_dataset = int(
        #     len(
        #         [name for name in os.listdir(self.root_dir) if os.path.isfile(os.path.join(self.root_dir, name))]
        #     ) / 2   # imput and output
        # )
        size_of_dataset = int(len(self.edof_files))
           
        return size_of_dataset
    

def standardize(image):
    image = image.float()
    return (image - image.mean()) / image.std()

def minmax(image):
    return (image - image.min()) / (image.max() - image.min())

def percentile(image, low=0, up=100):
    lower = np.percentile(image, low)
    upper = np.percentile(image, up)
    return (image - lower) / (upper - lower)
