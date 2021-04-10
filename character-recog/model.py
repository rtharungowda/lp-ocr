import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import time
import os
import copy
from dataloader import loader
from utils import save_ckp, plot
import pandas as pd

import albumentations as A
from albumentations.pytorch import ToTensorV2

from efficientnet_pytorch import EfficientNet

import config

def calc_padding(in_height, in_width,filter_height, filter_width, strides=(None,1,1)):
    out_height = np.ceil(float(in_height) / float(strides[1]))
    out_width  = np.ceil(float(in_width) / float(strides[2]))

    #The total padding applied along the height and width is computed as:
    if (in_height % strides[1] == 0):
        pad_along_height = max(filter_height - strides[1], 0)
    else:
        pad_along_height = max(filter_height - (in_height % strides[1]), 0)
    if (in_width % strides[2] == 0):
        pad_along_width = max(filter_width - strides[2], 0)
    else:
        pad_along_width = max(filter_width - (in_width % strides[2]), 0)

    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left

    return (pad_left, pad_right, pad_top, pad_bottom)

class akbhd(nn.Module):
    def __init__(self):
        super(akbhd, self).__init__()
        self.conv1 = nn.Conv2d(1,32,kernel_size=(5,5))
        self.maxpool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(32,64,kernel_size=(5,5))
        self.maxpool2 = nn.MaxPool2d(kernel_size=5,stride=5)
        self.linear1 = nn.Linear(64*2*2,config.NUM_CLASSES)
    
    def forward(self,x):
        x = self.conv1(x)
        # x = torch.sigmoid(x)
        x = F.relu(x)
        x = F.pad(x,calc_padding(x.size(2),x.size(3),2,2,strides=(None,2,2)))
        x = self.maxpool1(x)

        x = self.conv2(x)
        # x = torch.sigmoid(x)
        x = F.relu(x)
        x = F.pad(x,calc_padding(x.size(2),x.size(3),5,5,strides=(None,5,5)))
        x = self.maxpool2(x)

        x = x.view(x.size(0),-1)
        x = self.linear1(x)

        return x

class vatch(nn.Module):
    def __init__(self):
        super(vatch, self).__init__()
        self.conv0 = nn.Conv2d(1,64,kernel_size=(3,3))
        self.conv1 = nn.Conv2d(64,64,kernel_size=(3,3))
        self.conv2 = nn.Conv2d(64,128,kernel_size=(3,3))
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(128,128,kernel_size=(3,3))
        self.conv4 = nn.Conv2d(128,256,kernel_size=(5,5))
        self.conv5 = nn.Conv2d(256,256,kernel_size=(5,5))
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        self.linear0 = nn.Linear(16384,512)
        self.linear1 = nn.Linear(512,128)
        self.linear2 = nn.Linear(128,64)
        self.linear3 = nn.Linear(64,config.NUM_CLASSES)
    
    def forward(self,x):
        x = F.pad(x,calc_padding(32,32,3,3))
        x = F.relu(self.conv0(x))

        x = F.pad(x,calc_padding(x.size(2),x.size(3),3,3))
        x = F.relu(self.conv1(x))

        x = F.pad(x,calc_padding(x.size(2),x.size(3),3,3))
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)

        x = F.pad(x,calc_padding(x.size(2),x.size(3),3,3))
        x = F.relu(self.conv3(x))

        x = F.pad(x,calc_padding(x.size(2),x.size(3),5,5))
        x = F.relu(self.conv4(x))
        x = self.maxpool4(x)

        x = F.pad(x,calc_padding(x.size(2),x.size(3),5,5))
        x = F.relu(self.conv5(x))
        
        x = x.view(x.size(0),-1)
        x = F.relu(self.linear0(x))
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x

class drklrd(nn.Module):
    def __init__(self):
        super(drklrd, self).__init__()
        self.conv0 = nn.Conv2d(1,32,kernel_size=(3,3))
        self.batchnorm2d0 = nn.BatchNorm2d(32)
        self.conv1 = nn.Conv2d(32,32,kernel_size=(3,3))
        self.batchnorm2d1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,64,kernel_size=(3,3))
        self.batchnorm2d2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,64,kernel_size=(3,3))
        self.batchnorm2d3 = nn.BatchNorm2d(64)
        self.linear1 = nn.Linear(256,128)
        self.batchnorm1d1 = nn.BatchNorm1d(128)
        self.linear2 = nn.Linear(128,64)
        self.batchnorm1d2 = nn.BatchNorm1d(64)
        self.linear3 = nn.Linear(64,config.NUM_CLASSES)

        self.maxpool = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        
    
    def forward(self,x):
        x = self.batchnorm2d0(F.relu(self.conv0(x)))
        # x = F.pad(x,calc_padding(x.size(2),x.size(3),2,2,strides=(None,2,2)))
        x = self.maxpool(x)
        
        x = self.batchnorm2d1(F.relu(self.conv1(x)))
        # x = F.pad(x,calc_padding(x.size(2),x.size(3),2,2,strides=(None,2,2)))
        x = self.maxpool(x)
        
        x = self.batchnorm2d2(F.relu(self.conv2(x)))
        # x = F.pad(x,calc_padding(x.size(2),x.size(3),2,2,strides=(None,2,2)))
        x = self.maxpool(x)
        
        # x = self.batchnorm2d3(F.relu(self.conv3(x)))
        # x = F.pad(x,calc_padding(x.size(2),x.size(3),2,2,strides=(None,2,2)))
        # x = self.maxpool(x)
        
        x = x.view(x.size(0),-1)
        x = self.batchnorm1d1(F.relu(self.linear1(x)))
        x = self.batchnorm1d2(F.relu(self.linear2(x)))
        x = self.linear3(x)
        
        return x

def mdl(type):
    if type == "res18":
        model_ft = models.resnet18(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, config.NUM_CLASSES)
    
    elif type == "res34":
        model_ft = models.resnet34(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, config.NUM_CLASSES)
    
    elif type == "res50":
        model_ft = models.resnet50(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, config.NUM_CLASSES)

    elif type == "eff-b0":
        model_ft = EfficientNet.from_pretrained('efficientnet-b0', num_classes=config.NUM_CLASSES)
    
    elif type == "eff-b1":
        model_ft = EfficientNet.from_pretrained('efficientnet-b1', num_classes=config.NUM_CLASSES)

    return model_ft
    
if __name__ == '__main__':

    mdl = drklrd()
    x = torch.rand((2,1,32,32))
    print(mdl(x).size())