"""
This module contain different fucntions to load dataset, transform, and process images 
"""
# Import the neccessary libraries 
import torch 
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
import torch.nn.functional as F
import os
import argparse
import matplotlib.pyplot as plt
from collections import OrderedDict
from PIL import Image
import numpy as np


def data_prep(args):
    """ 
    It accept dataset directory, load the required datasets, transform it and 
    return dictionary of the dataset and the dataloader
    Parameters
    ----------
    filename: str
        Dataset directory 
        .. You should include the right directory where the dataset exist!
    Returns
    -------
    Dictionary
       It returns dictionary of two variables
        
    Examples
    --------
    >>>x, y = data_prep( "../user/dataset/flowers")
    """
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    image_datasets= dict()
    dataloaders = dict()
    # transformation for the training datasets
    train_data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                                transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # transformation for the test and validation datasets
    testval_data_transforms = transforms.Compose([transforms.Resize(256),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(), 
                                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # Loading the datasets with ImageFolder
    image_datasets['train'] = datasets.ImageFolder(train_dir, transform=train_data_transforms)
    image_datasets['valid'] = datasets.ImageFolder(valid_dir, transform=testval_data_transforms)
    image_datasets['test'] = datasets.ImageFolder(test_dir, transform=testval_data_transforms)

    # TODO: Define the dataloaders
    dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size = 32, shuffle=True)
    dataloaders['valid'] = torch.utils.data.DataLoader(image_datasets['valid'], batch_size = 32)
    dataloaders['test'] = torch.utils.data.DataLoader(image_datasets['test'], batch_size = 32)
    
    return image_datasets, dataloaders

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    imgtopil = Image.open(image)
    transform_image = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                         ])
    tfd_image = transform_image(imgtopil)
    # conver to numpy array 
    arr_tfd_image = np.array(tfd_image)
    
    return arr_tfd_image
    
    

