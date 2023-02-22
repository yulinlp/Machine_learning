# Numerical Operations
import math
import numpy as np

# Reading/Writing Data
import pandas as pd
import os
import csv
#
# For Progress Bar
from tqdm import tqdm

# Pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# For plotting learning curve
from torch.utils.tensorboard import SummaryWriter

device = 'cpu'

folder_path = 'D:/yulin/学习/Dataset/covid_regression'
testData_path = folder_path + '/' + 'covid.test.csv'
trainData_path = folder_path + '/' + 'covid.train.csv'

trainFrame = pd.read_csv(trainData_path)
testFrame = pd.read_csv(testData_path)
# print(trainFrame)


def same_seed(seed = 0):
    '''Fixes random number generator seeds for reproducibility.'''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_valid_split(dataset, valid_radio, seed = 0):
    '''Split provided training data into training set and validation set'''
    train_set,valid_set = random_split(dataset,[int (len(dataset)*(1-valid_radio)), len(dataset)*valid_radio],generator=torch.Generator().manual_seed(seed))
    return train_set,valid_set

def predict(test_loader, model):
    model.eval()
    preds = []
    for x in test_loader:
        x = x.to(device)
        with torch.no_grad():
            pred = model(x)
            preds.append(pred.detach().cpu())
    preds = torch.cat(preds,dim=0).numpy()
    return preds