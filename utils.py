import pandas as pd
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt

import torch

def plot(loss_p,acc_p,epochs):
    x = [i for i in range(epochs)]
    plt.plot(x,loss_p['train'],color='red', marker='o')
    plt.title('Train loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.grid(True) 
    plt.savefig('/content/train_loss.png')
    plt.clf()

    plt.plot(x, loss_p['val'],color='red', marker='o')
    plt.title('Val loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.grid(True) 
    plt.savefig('/content/val_loss.png')
    plt.clf()
    
    #acc
    plt.plot(x, acc_p['train'],color='red', marker='o')
    plt.title('Train acc')
    plt.xlabel('epochs')
    plt.ylabel('acc')
    plt.grid(True) 
    plt.savefig('/content/train_acc.png')
    plt.clf()

    plt.plot(x, acc_p['val'],color='red', marker='o')
    plt.title('Val acc')
    plt.xlabel('epochs')
    plt.ylabel('acc')
    plt.grid(True) 
    plt.savefig('/content/val_acc.png')
    plt.clf()

def save_ckp(state, checkpoint_path):
    f_path = checkpoint_path
    torch.save(state, f_path)

def load_ckp(checkpoint_fpath, model, optimizer, device):

    checkpoint = torch.load(checkpoint_fpath,map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    valid_acc = checkpoint['valid_acc'] 
    return model, optimizer, checkpoint['epoch'], valid_acc
