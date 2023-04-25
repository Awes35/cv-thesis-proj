#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 22:13:31 2023

@author: kollengruizenga
"""

from torchvision import models, transforms
import torch

from PIL import Image #Pillow is the default image backend supported by TorchVision

dir(models)


#%% Functions

def load_img(file):
    
    #Load image
    img = Image.open(file)
    
    #Create transformer
    transform = transforms.Compose([
        transforms.Resize(256),                    #Resize the image to 256×256 pixels.
        transforms.CenterCrop(224),                #Crop the image to 224×224 pixels about the center.
        transforms.ToTensor(),                     #Convert the image to PyTorch Tensor data type.
        transforms.Normalize(mean=[0.485, 0.456, 0.406], # Normalize the image by setting mean 
                             std=[0.229, 0.224, 0.225]) # and std to the specified values.
        ])
    
    #Transform image
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0)
    
    return img, img_t, batch_t


def get_prediction(out_vector):
    
    #get index where the max score in the output vector "out" occurs
    _, index = torch.max(out_vector, 1)
    percentage = torch.nn.functional.softmax(out_vector, dim=1)[0]
    pred = (classes[index[0]], percentage[index[0]].item())
    
    #get top 5 classes
    _, indices = torch.sort(out_vector, descending=True)
    top5 = [(classes[idx], percentage[idx].item()) for idx in indices[0][:5]]
    
    return pred, top5


#%% Get classes (labels) for ImageNet dataset (models trained on)

#ImageNet classes -- https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
with open('imagenet_classes.txt') as f:
    classes = [line.strip() for line in f.readlines()]


#%% AlexNet - CNN

alexnet = models.alexnet(pretrained=True) #weights=True
print(alexnet)

alexnet.eval() # put model in eval mode


#%% ResNet 101 - 101 layer CNN

resnet = models.resnet101(pretrained=True)
print(resnet)

resnet.eval() # put model in eval mode


#%% Load image - Dog
#https://github.com/pytorch/hub/raw/master/images/dog.jpg

img, img_t, batch_t = load_img("dog.jpg")

#AlexNet output vector & predictions
alex_vector = alexnet(batch_t)
print(alex_vector.shape)
alex_pred, alex_top5 = get_prediction(alex_vector)

#ResNet output vector & predictions
res101_vector = resnet(batch_t)
print(res101_vector.shape)
res101_pred, res101_top5 = get_prediction(res101_vector)


#%% Load image - guydude

img, img_t, batch_t = load_img("guydude.jpeg")

#AlexNet output vector & predictions
alex_vector = alexnet(batch_t)
print(alex_vector.shape)
alex_pred, alex_top5 = get_prediction(alex_vector)

#ResNet output vector & predictions
res101_vector = resnet(batch_t)
print(res101_vector.shape)
res101_pred, res101_top5 = get_prediction(res101_vector)


#%% Load image - flowerbug

img, img_t, batch_t = load_img("flowerbug.jpeg")

#AlexNet output vector & predictions
alex_vector = alexnet(batch_t)
print(alex_vector.shape)
alex_pred, alex_top5 = get_prediction(alex_vector)

#ResNet output vector & predictions
res101_vector = resnet(batch_t)
print(res101_vector.shape)
res101_pred, res101_top5 = get_prediction(res101_vector)


#%% Load image - labradorsX3

img, img_t, batch_t = load_img("labradorsX3.jpg")

#AlexNet output vector & predictions
alex_vector = alexnet(batch_t)
print(alex_vector.shape)
alex_pred, alex_top5 = get_prediction(alex_vector)

#ResNet output vector & predictions
res101_vector = resnet(batch_t)
print(res101_vector.shape)
res101_pred, res101_top5 = get_prediction(res101_vector)


#%% Load image - soldiers

img, img_t, batch_t = load_img("soldiers.jpeg")

#AlexNet output vector & predictions
alex_vector = alexnet(batch_t)
print(alex_vector.shape)
alex_pred, alex_top5 = get_prediction(alex_vector)

#ResNet output vector & predictions
res101_vector = resnet(batch_t)
print(res101_vector.shape)
res101_pred, res101_top5 = get_prediction(res101_vector)


#%% Load image - artgirl

img, img_t, batch_t = load_img("artgirl.jpeg")

#AlexNet output vector & predictions
alex_vector = alexnet(batch_t)
print(alex_vector.shape)
alex_pred, alex_top5 = get_prediction(alex_vector)

#ResNet output vector & predictions
res101_vector = resnet(batch_t)
print(res101_vector.shape)
res101_pred, res101_top5 = get_prediction(res101_vector)



