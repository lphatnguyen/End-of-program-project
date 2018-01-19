import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

dataset = pd.read_csv("F:/Projet fin d'Etudes/Neural Network trial/data.csv")

value = dataset.iloc[:,0].values
label = dataset.iloc[:,1].values

class Net(nn.Module):
    def __init__(self):
        self.fc1=nn.Linear(1,1)
        self.fc2=nn.Linear(1,1)
        self.fc3=nn.Linear(1,1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
net = Net()