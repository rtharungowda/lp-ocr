import os
import torch

LR = 0.0001
NUM_EPOCHS = 20
BATCH_SIZE = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 36
TRAIN_IMAGES = "/content/drive/MyDrive/competitions/mosaic-r2/dataset_characters"
TRAIN_CSV = '/content/drive/MyDrive/competitions/mosaic-r2/char_train_seg.csv'
VAL_CSV = '/content/drive/MyDrive/competitions/mosaic-r2/char_val_seg.csv'

MAPPING = {
    '0':0,
    '1':1,
    '2':2,
    '3':3,
    '4':4,
    '5':5,
    '6':6,
    '7':7,
    '8':8,
    '9':9,
    'A':10,
    'B':11,
    'C':12,
    'D':13,
    'E':14,
    'F':15,
    'G':16,
    'H':17,
    'I':18,
    'J':19,
    'K':20,
    'L':21,
    'M':22,
    'N':23,
    'O':24,
    'P':25,
    'Q':26,
    'R':27,
    'S':28,
    'T':29,
    'U':30,
    'V':31,
    'W':32,
    'X':33,
    'Y':34,
    'Z':35,
}

if __name__ == "__main__":
    print(NUM_CLASSES)