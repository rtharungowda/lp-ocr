import cv2
import glob
import numpy as np
from PIL import Image
import os

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch.nn as nn
import torch
import torch.optim as optim
from torchvision import transforms, models

import argparse

NOT = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REV_MAPPING = {
    0:'0',
    1:'1',
    2:'2',
    3:'3',
    4:'4',
    5:'5',
    6:'6',
    7:'7',
    8:'8',
    9:'9',
    10:'A',
    11:'B',
    12:'C',
    13:'D',
    14:'E',
    15:'F',
    16:'G',
    17:'H',
    18:'I',
    19:'J',
    20:'K',
    21:'L',
    22:'M',
    23:'N',
    24:'O',
    25:'P',
    26:'Q',
    27:'R',
    28:'S',
    29:'T',
    30:'U',
    31:'V',
    32:'W',
    33:'X',
    34:'Y',
    35:'Z',
}

def load_ckp(checkpoint_fpath, model, optimizer, device):
    """load saved model  and optimizer 

    Args:
        checkpoint_fpath (str): path of saved model
        model (torch.model): model to be loaded
        optimizer (torch.optim): optimizer to be loaded
        device (torch.device): load model on device

    Returns:
        torch.model: pytorch model
        torch.optim: pytorch optimizer
        int: epoch number 
        float: validation accuracy
    """

    checkpoint = torch.load(checkpoint_fpath,map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    valid_acc = checkpoint['valid_acc'] 
    return model, optimizer, checkpoint['epoch'], valid_acc

def preprocess(img):
    """return preprocessed image

    Args:
        img (np.array): image

    Returns:
        torch.tensor: processed image tensor
    """
    img = np.array(img)
    h = img.shape[0]
    w = img.shape[1]
    #create 3 channels, as segmented image is gray scale and imagenet models take 3 channels
    new_img = np.zeros((h,w,3))
    for i in range(h):
        for j in range(w):
            new_img[i][j][0]=img[i][j]
            new_img[i][j][1]=img[i][j]
            new_img[i][j][2]=img[i][j]
    transform = A.Compose([
                    A.Resize(width=224, height=224),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ])
    img = transform(image=new_img)['image'].float().unsqueeze(0)
    return img

def predict_charac(img):
    """predicts the character in image

    Args:
        img (np.array): image

    Returns:
        torch.tensor: predicted character label
    """
    #define model
    model_ft = models.resnet34(pretrained=False)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 36)

    #load model and place in eval mode
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)
    checkpoint_path_res34 = "res34_albu_seg_3.pt"
    model, _, _, val_acc = load_ckp(checkpoint_path_res34, model_ft, optimizer_ft, DEVICE)
    model = model.to(DEVICE)
    model.eval()

    img = preprocess(img)
    img = img.to(DEVICE)
    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)
    
    if REV_MAPPING[preds.item()] == 'A' and NOT:
        return preds.item(),'1'
    return preds.item(),REV_MAPPING[preds.item()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="predict character")
    parser.add_argument("--folder",type=str,help="path to folder containing images",required=False)
    parser.add_argument("--image",type=str,help="path to image",required=False)
    args = parser.parse_args()
    folder = []
    if args.folder is not None:
        folder = args.folder
    
    paths = glob.glob(folder+"/*.jpeg")

    if args.image is not None:
        paths.append(args.image)
    for path in paths:
        print(path)
        img = cv2.imread(path,0)
        print(REV_MAPPING[predict_charac(img)])