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
from workspace_utils import active_session
from preprocessing import process_image
from saveload import load_checkpoint
import json


def predict(args, model, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    # preprocess the image
    image = process_image(args.image_path)
    topk = args.topk
    # From numpy to torch tensor 
    imgto_torch = torch.from_numpy(image).type(torch.FloatTensor)
    imgto_torch = imgto_torch.unsqueeze(0)
    imgto_torch = imgto_torch.to(device)
    # Enable evaluation mode
    model.to(device)
    model.eval()
    with torch.no_grad():
        output = model.forward(imgto_torch)
    #calculate probablility 
        prb = torch.exp(output)
        probs, indices = torch.topk(prb, topk)
        top_probs = [float(probs) for probs in probs[0]]
        iv_map = {i:j for j, i in model.class_to_idx.items()}
        top_classes = [iv_map[int(idx)] for idx in indices[0]]
        
    return top_probs, top_classes

def main():
    parser = argparse.ArgumentParser(description = "Classification Predictor")
    parser.add_argument('--gpu', default = False, action ='store_true', help = 'enable or disable GPU')
    parser.add_argument('--image_path', type = str, help = 'Path to image')
    parser.add_argument('--saved_model', type = str, default = 'checkpoint.pth', help = 'path to saved model')
    parser.add_argument('--topk', type = int, default = 5, help = 'display top k probabilities')
    parser.add_argument('--load_json', type =str, default = 'cat_to_name.json', help='path to json file')
    
    args = parser.parse_args()
    
    with open(args.load_json,'r') as f:
        cat_to_name = json.load(f)
    
    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda") 
        print("Using GPU:{}".format(device))
    else:
        device =  torch.device("cpu")
        print("GPU is not available..now running on CPU")

    model = load_checkpoint(args)
    
    top_probability, top_class = predict(args,model, device) 
    
    print('Predicted Classes: ', top_class)
    print('Predicted Probability: ', top_probability)
    class_name = []
    [class_name.append(cat_to_name[i]) for i in top_class]
    print ('Class Names:',class_name )
   
    
if __name__ == "__main__":
    main()
    