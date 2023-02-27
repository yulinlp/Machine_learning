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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'seed': 521,
    'select_all': True,
    'valid_ratio': 0.3,   # validation_size = train_size * valid_ratio
    'n_epochs': 1000,     # Number of epochs.
    'batch_size': 256,
    'learning_rate': 1e-5,
    'early_stop': 400,    # If model has not improved for this many consecutive epochs, stop training.
    'save_path': './models/model.ckpt'  # Your model will be saved here.
}

# Some Utility Functions
def same_seed(seed = config['seed']):
    '''Fixes random number generator seeds for reproducibility.'''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_valid_split(dataset, valid_radio=config['valid_ratio'], seed=config['seed']):
    '''Split provided training data into training set and validation set'''
    train_set,valid_set = random_split(dataset,[int (len(dataset)*(1-valid_radio)), math.ceil(len(dataset)*valid_radio)],generator=torch.Generator().manual_seed(seed))
    return np.array(train_set),np.array(valid_set)


def get_data(folder_path='D:/yulin/学习/Dataset/covid_regression'):
    '''
    get data from folder_path and return train_dataset,valid_dataset,test_dataset
    '''

    folder_path = 'D:/yulin/学习/Dataset/covid_regression'
    testData_path = folder_path + '/' + 'covid.test.csv'
    trainData_path = folder_path + '/' + 'covid.train.csv'
    trainFrame = pd.read_csv(trainData_path).values
    testFrame = pd.read_csv(testData_path).values
    train_set, valid_set = train_valid_split(trainFrame)

    print(f"""
    train_data size: {train_set.shape} 
    valid_data size: {valid_set.shape} 
    test_data size: {testFrame.shape}
    """)

    # Select features
    x_train, x_valid, x_test, y_train, y_valid = select_feat(train_set, valid_set, testFrame, config['select_all'])

    # Print out the number of features.
    print(f'number of features: {x_train.shape[1]}')
    # train_set = torch.FloatTensor(np.array(train_set)).to(device)
    # valid_set = torch.FloatTensor(np.array(valid_set)).to(device)
    # test_set = torch.FloatTensor(np.array(testFrame)).to(device)
    train_dataset, valid_dataset, test_dataset = MyDataset(x_train,y_train), MyDataset(x_valid, y_valid), MyDataset(x_test)

    return train_dataset,valid_dataset,test_dataset



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
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)
        # tensor_x = torch.FloatTensor(x) <=> convert x to tensor_x
        self.x = torch.FloatTensor(x)


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
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16,64),
            nn.ReLU(),
            nn.Linear(64,8),
            nn.ReLU(),
            nn.Linear(8,1)
        )

    def forward(self,x):
        return self.model(x)    # the significance of squeeze() is ?


# Feature Selection
def select_feat(train_data, valid_data, test_data, select_all=True):
    '''Selects useful features to perform regression'''
    y_train, y_valid = train_data[:, -1], valid_data[:, -1]
    raw_x_train, raw_x_valid, raw_x_test = train_data[:, :-1], valid_data[:, :-1], test_data

    if select_all:
        feat_idx = list(range(raw_x_train.shape[1]))
    else:
        feat_idx = [0, 1, 2, 3, 4]

    return raw_x_train[:, feat_idx], raw_x_valid[:, feat_idx], raw_x_test[:, feat_idx], y_train, y_valid

    #1 VarianceThreshold is not necessary

    # selector = VarianceThreshold()
    # X_var0 = selector.fit_transform(X=dataset)
    # print(X_var0.shape)
    # print(pd.DataFrame(X_var0).head())

    #2

def trainer(train_loader, valid_loader, model):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    writer = SummaryWriter()    # tensorboard

    if not os.path.isdir('./models'):
        os.mkdir('./models')

    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf, 0, 0
    for epoch in range(n_epochs):
        model.train()
        loss_records = []
        train_pbar = tqdm(train_loader, position=0)
        for x, y in train_pbar:
            optimizer.zero_grad()   # set gradient to zero
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(y,pred)
            loss.backward()     # backpropagation
            optimizer.step()    # update parameters
            loss_records.append(loss.detach().item())

            # Display current epoch number and loss on tqdm progress bar.
            train_pbar.set_description(f'Epoch [{epoch + 1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})

        mean_train_loss = sum(loss_records) / len(loss_records)
        writer.add_scalar('Loss/train', mean_train_loss, step)

        model.eval()  # Set your model to evaluation mode.
        loss_records = []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)

            loss_records.append(loss.item())

        mean_valid_loss = sum(loss_records) / len(loss_records)
        print(f'Epoch [{epoch + 1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')
        writer.add_scalar('Loss/valid', mean_valid_loss, step)

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict, config['save_path'])
            print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1
        if early_stop_count > config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
            return


def save_pred(preds, file):
    ''' Save predictions to specified file '''
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])

def tester(model,test_loader):
    model = MyModel(input_dim=117).to(device)
    model.load_state_dict(torch.load(config['save_path']))
    preds = predict(test_loader, model)
    save_pred(preds, 'pred.csv')

if __name__ == '__main__':

    train_set, valid_set, test_set = get_data()
    # Pytorch data loader loads pytorch dataset into batches.
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, pin_memory=False)   # pin_memory is True if training on a GPU
    valid_loader = DataLoader(valid_set, batch_size=config['batch_size'], shuffle=True, pin_memory=False)
    test_loader = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False, pin_memory=False)
    # print(len(train_set.x[1]))
    model = MyModel(input_dim=len(train_set.x[1])).to(device)  # input_dim = num of features
    trainer(train_loader,valid_loader,model)
