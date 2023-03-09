"""
This module is simply save and load output parameters 
"""
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

# Function that save the checkpoint,
def save_checklist(args,image_datasets, model):
    model.epochs = args.epochs
    model.class_to_idx = image_datasets['train'].class_to_idx
    checkpoint = {'model':model,
                  'state_dict':model.state_dict(), 
                  'classifier': model.classifier,
                  'class_to_idx':model.class_to_idx,
                  'output_size':102,
                  'epoch':model.epochs}
    torch.save(checkpoint,args.saved_model)
    
# Function that loads a checkpoint and rebuilds the model
def load_checkpoint(args):
    checkpoint = torch.load(args.saved_model)
    model = checkpoint["model"]
    model.load_state_dict(checkpoint['state_dict'])
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    return model