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
from sklearn.feature_selection import VarianceThreshold
# Pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# For plotting learning curve
from torch.utils.tensorboard import SummaryWriter

device = 'cpu'
epoch = 100

# Some Utility Functions
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
    train_set,valid_set = random_split(dataset,[int (len(dataset)*(1-valid_radio)), math.ceil(len(dataset)*valid_radio)],generator=torch.Generator().manual_seed(seed))
    return train_set,valid_set


def get_data(folder_path='D:/yulin/学习/Dataset/covid_regression'):
    '''
    get data from folder_path and return train_set_X, train_set_Y, valid_set_X, valid_set_Y, test_set
    '''

    folder_path = 'D:/yulin/学习/Dataset/covid_regression'
    testData_path = folder_path + '/' + 'covid.test.csv'
    trainData_path = folder_path + '/' + 'covid.train.csv'
    trainFrame = pd.read_csv(trainData_path)
    testFrame = pd.read_csv(testData_path)
    train_set, valid_set = train_valid_split(trainFrame,valid_radio=0.3)
    train_set = torch.FloatTensor(np.array(train_set.dataset)).to(device)
    valid_set = torch.FloatTensor(np.array(valid_set.dataset)).to(device)
    test_set = torch.FloatTensor(np.array(testFrame)).to(device)
    # print(train_set.dataset)
    # print(train_set.shape)
    train_set_Y = train_set[:,-1]
    train_set_X = train_set[:,:-1]
    valid_set_Y = train_set[:,-1]
    # print(valid_set_Y)
    valid_set_X = valid_set[:,:-1]
    # test_set_X = testFrame
    # print(testFrame.shape)
    train_set_ret = MyDataset(train_set_X, train_set_Y)
    valid_set_ret = MyDataset(valid_set_X, valid_set_Y)
    test_set_ret = MyDataset(test_set)

    return train_set_ret,valid_set_ret,test_set_ret


def predict(test_loader, model):
    model.eval()
    preds = []
    for x in tqdm(test_loader):
        x = x.to(device)
        with torch.no_grad():
            pred = model(x)
            preds.append(pred.detach().cpu())
    preds = torch.cat(preds,dim=0).numpy()
    return preds

# Dataset
class MyDataset(Dataset):
    '''
        x: Features.
        y: Targets, if none, do prediction.
    '''
    def __init__(self,x,y = None,):
        if y==None:
            self.y = None
            self.x = torch.FloatTensor(x)
            return
        # tensor_x = torch.FloatTensor(x) <=> convert x to tensor_x
        self.x = torch.FloatTensor(x)
        self.y = torch.FloatTensor(y)

    def __getitem__(self, index):
        if self.y == None:
            return self.x[index]
        return self.x[index],self.y[index]

    def __len__(self):
        return len(self.x)


# Neural Network Model
class MyModel(nn.Module):
    def __init__(self, input_dim=None):
        super(MyModel, self).__init__()
        assert input_dim, "Error! input_dim is None!"
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32,8),
            nn.ReLU(),
            nn.Linear(8,1)
        )

    def forward(self,x):
        return self.model(x)    # the significance of squeeze() is ?


# Feature Selection
def features_select(dataset,):
    print(pd.DataFrame(dataset).shape)

    #1 VarianceThreshold is not necessary

    # selector = VarianceThreshold()
    # X_var0 = selector.fit_transform(X=dataset)
    # print(X_var0.shape)
    # print(pd.DataFrame(X_var0).head())

    #2


if __name__ == '__main__':

    train_set, valid_set, test_set = get_data()
    features_select(train_set.x)
    model = MyModel()


