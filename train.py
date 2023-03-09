"""
This module contain different fucntions to train a model 
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
from workspace_utils import active_session
from preprocessing import data_prep 
from saveload import save_checklist
import json 

def build_network(args, pretrained_model,device, dropout = 0.1):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pretrained_model == "vgg16":
        model = models.vgg16(pretrained=True)
        input_future = 25088
        hidden_unit = args.hidden_units
        
    elif pretrained_model == "densenet121":
        model = models.densenet121(pretrained = True)
        input_future = 1024
        hidden_unit = args.hidden_units
    # Freeze the parameters of pretrained model and define new forward network at the end of it 
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_future,hidden_unit)),
                                                  ('relu',nn.ReLU()),
                                                  ('dropout',nn.Dropout(dropout)),
                                                  ('fc2', nn.Linear(hidden_unit,102)),
                                                  ('output', nn.LogSoftmax(dim=1))]))
   
    criterion = torch.nn.NLLLoss() #CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), args.lr)
    model.to(device)
    return model, criterion, optimizer


# Function that measure the validation loss and accuracy
def validation(model, dataloader, criterion, device):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss = 0
    accuracy = 0
    with torch.no_grad():
        for inputs, labels in iter(dataloader):
            inputs, labels = inputs.to(device), labels.to(device) # Move input and label tensors to the GPU
            
            output = model.forward(inputs)
            loss = criterion(output, labels)
            total_loss = loss.item()

            prb = torch.exp(output) # get the class probabilities from log-softmax
            top_p, top_class = prb.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
   
    return total_loss, accuracy


# TODO: Build and train your network
#def train_model(model_struct, dataloaders, criterion, optimizer, epochs = 2, print_every = 40):
def train_model(args):
    
    if args.arch == 'vgg':
        pretrained_model = "vgg16"
    elif args.arch == 'densenet':
        pretrained_model = "densenet121"
    
    if args.gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using GPU:{}".format(device))
        else:
            device =  torch.device("cpu")
            print("GPU is not available..now running on CPU")
            
    # start with pre-trained network
    model,criterion,optimizer = build_network(args,pretrained_model,device) 
    image_datasets, dataloaders = data_prep(args) 
   
    with active_session():
        print("The training process is started....\n") 
        steps = 0
        running_loss = 0
        training_accuracy = 0
        print_every = 40
        for ep in range(args.epochs):
            for inputs, labels in dataloaders['train']:
                steps += 1
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                # Forward and backward propagation
                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                px = torch.exp(outputs)
                equal = (labels.data == px.max(dim=1)[1])
                training_accuracy += equal.type(torch.FloatTensor).mean()
                if steps % print_every == 0:
                    model.eval()
                    valid_loss = 0
                    accuracy = 0
                    valid_loss, accuracy = validation(model, dataloaders['valid'], criterion, device)
                    print("Epoch: {}/{} |".format(ep+1,args.epochs),
                          "Training Loss:{:.3f} |".format(running_loss/print_every),
                          "Training Accuracy:{:3f} |".format(training_accuracy/print_every),
                          "Validation Loss: {:.3f} |".format(valid_loss/len(dataloaders['valid'])),
                          "Validation Accuracy: {:3f}".format(accuracy/len(dataloaders['valid'])))
                    running_loss = 0
                    training_accuracy = 0
                    model.train()
                  
        save_checklist(args,image_datasets, model) 

def main():
    parser = argparse.ArgumentParser(description = "Classification Trainer")
    parser.add_argument('--gpu', type = bool, default = False, help = 'enable or disable GPU')
    parser.add_argument('--arch', type = str, default = "vgg", help = " choose from vgg or densenet", required = True)
    parser.add_argument('--lr', type = float, default = 0.001, help = 'set learning rate ')
    parser.add_argument('--hidden_units', type = int, default = 4096, help = 'hidden unit')
    parser.add_argument('--epochs', type = int, default = 5, help = 'number of epochs')
    parser.add_argument('--data_dir', type = str, default = 'flowers', help = 'dataset directory')
    parser.add_argument('--saved_model', type = str, default = 'checkpoint.pth', help = 'path to saved model')
    
    args = parser.parse_args()
    
    with open('cat_to_name.json','r') as f:
        cat_to_name = json.load(f)
    
    train_model(args)
    
if __name__ == "__main__":
    main()
    


                                                  
