from torch.utils.data import Dataset
from torchvision.transforms import v2
import torch
from os import listdir
from os.path import isfile, join
from pathlib import Path

from PIL import Image

def get_files(directory, keyword):
    """Returns all files in directory if the file contains the keyword."""
    file_list = []
    for file in sorted(listdir(directory)):
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
        assert len(self.mip_files) == len(self.edof_files)
        for mf, ef in zip(self.mip_files, self.edof_files):
            ef = Path(ef).parent/Path(ef).name.replace("edof", "mip")
            assert str(Path(mf)) == str(ef), f"{mf} != {ef}"

        
        # set image size
        self.image_size= image_size

        # 1. Transform the data to the form that we need. E.g. use ToTensor to convert the datatype to Tensor
        # 2. Augment the data. E.g. flip and rotation.
        self.default_transformation = v2.Compose([
            v2.RandomCrop(size=image_size),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
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
        input_image_name = self.mip_files[index]
        output_image_name = self.edof_files[index]
        input_image = Image.open(input_image_name)
        output_image = Image.open(output_image_name)
        # apply transform to both input and output
        input_image, output_image = self.default_transformation(input_image, output_image)
        # normalize input
        if self.normalize == "standard":
            mean = input_image.mean()
            std = input_image.std()
            input_image = (input_image - mean) / std
            output_image = (output_image - mean) / std
        elif self.normalize == "minmax":
            mi = input_image.min()
            ma = input_image.max()
            input_image = (input_image - mi) / (ma - mi)
            output_image = (output_image - mi) / (ma - mi)
        
        #create sample to return
        sample = {'index': int(index), 'input_image': input_image, 'output_image': output_image}

        return sample

    def __len__(self):
        """
        Returns the size of the dataset.
        """

        # Get the size of the datasets (The number of samples in the dataset.)
        size_of_dataset = int(len(self.edof_files))
           
        return size_of_dataset

    
def get_splits(path: str, **kwargs) -> tuple[ImageDataset, ImageDataset]:
    return ImageDataset(path, ["train"], **kwargs), ImageDataset(path, ["val"], **kwargs)
