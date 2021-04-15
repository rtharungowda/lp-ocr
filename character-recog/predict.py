import cv2
import glob
import numpy as np
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch.nn as nn
import torch
import torch.optim as optim
from torchvision import transforms, models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    model_ft = models.resnet18(pretrained=False)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 36)

    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)
    checkpoint_path_res34 = "/content/drive/MyDrive/competitions/mosaic-r2/weights/res18_albu_seg.pt"
    model, _, _, _ = load_ckp(checkpoint_path_res34, model_ft, optimizer_ft, DEVICE)
    model = model.to(DEVICE)
    model.eval()

    img = preprocess(img)
    img = img.to(DEVICE)
    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)

    return preds


if __name__ == "__main__":
    folder = "/content/drive/MyDrive/competitions/mosaic-r2/test_charac"
    paths = glob.glob(folder+"/*.jpeg")
    # imgs = perform_segmentation(path)
    for path in paths:
        print(path)
        img = cv2.imread(path,0)
        print(predict_charac(img))