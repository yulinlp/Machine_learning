import torch
from torch.utils.data import DataLoader,Dataset

# Hyperparameter
batch_size = 5
epoch_size = 100
lr = 0.001
device = 'cpu' # or 'cuda'

# Dataset
class MyDataset(Dataset):
    def __init__(self,file):
        self.data = []

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

import torch.nn as nn

# Model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10,32),
            nn.Sigmoid(),
            nn.Linear(32,1)
        )

    def forward(self,x):
        return self.net(x)



# Training Setup
dataset = MyDataset("file")
dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
model = MyModel()
MSELoss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr)


# Training Loop
def train(train_dataloader):
    # training pattern
    model.train()
    for epoch in range(epoch_size):
        for x,y in train_dataloader:  # each loop is one batch
            optimizer.zero_grad()
            x,y = x.to(device),y.to(device)
            pred = model(x)
            loss = MSELoss(pred,y)
            loss.backward()
            optimizer.step()


# Evaluate Loop
def evaluate(evaluate_dataloader):
    model.eval()
    total_loss = 0
    for x,y in evaluate_dataloader:
        x,y = x.to(device),y.to(device)
        with torch.no_grad():
            pred = model(x)
            loss = MSELoss(pred,y)
            total_loss += loss.item() * len(x)
            avg_loss = total_loss / len(evaluate_dataloader.dataset)

    print(total_loss,avg_loss)


# Test loop
def test(test_dataloader):
    model.eval()
    preds = []
    for x in test_dataloader:
        x = x.to(device)
        with torch.no_grad():
            pred = model(x)
            preds.append(pred)

    print(preds)

# Save model
def Save(model,path):
    if torch.save(model.state_dict(),path):
        return True
    return False

# Load model
def Load(path):
    modelFile = torch.load(path)
    model.load_state_dict(modelFile)
