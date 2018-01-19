# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt

# Building data of 3 classes
N = 100 # number of points per class
d0 = 2 # dimensionality
C = 3 # number of classes
X = np.zeros((d0, N*C)) # data matrix (each row = single example)
y = np.zeros(N*C) # class labels

for j in range(C):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[:,ix] = np.c_[r*np.sin(t), r*np.cos(t)].T
  y[ix] = j
# lets visualize the data:
# plt.scatter(X[:N, 0], X[:N, 1], c=y[:N], s=40, cmap=plt.cm.Spectral)

plt.plot(X[0, :N], X[1, :N], 'bs', markersize = 7);
plt.plot(X[0, N:2*N], X[1, N:2*N], 'ro', markersize = 7);
plt.plot(X[0, 2*N:], X[1, 2*N:], 'g^', markersize = 7);
# plt.axis('off')
plt.xlim([-1.5, 1.5])
plt.ylim([-1.5, 1.5])
cur_axes = plt.gca()
cur_axes.axes.get_xaxis().set_ticks([])
cur_axes.axes.get_yaxis().set_ticks([])

plt.savefig('EX.png', bbox_inches='tight', dpi = 600)
plt.show()
X = X.transpose()
dataset = np.column_stack((X,y))
dataset = np.random.permutation(dataset)

# begin to build the neural network

import torch
from torch import autograd, nn, optim
import torch.nn.functional as F

data = torch.from_numpy(dataset[:,0:2]).float()
label = torch.from_numpy(dataset[:,2]).long()

inputSize = 2
batchSize = N*C
hiddenSize = 20
numClass = C
learningRate = 0.001
torch.manual_seed(123)

torch.manual_seed(123)
inputData = autograd.Variable(data)
target = autograd.Variable(label)

class Net(nn.Module):
    def __init__(self, inputSize, hiddenSize, numClass):
        super().__init__()
        self.h1 = nn.Linear(inputSize, hiddenSize)
        self.h2 = nn.Linear(hiddenSize, hiddenSize)
        self.h3 = nn.Linear(hiddenSize, numClass)
        
    def forward(self,x):
        x=self.h1(x)
        x=F.tanh(x)
        x=self.h2(x)
        x=F.softmax(x)
        x=self.h3(x)
        x=F.softmax(x)
        
        return x

model = Net(inputSize = inputSize, hiddenSize = hiddenSize, numClass = numClass)
opt = optim.Adam(params = model.parameters(), lr = learningRate)

for epoche in range(2000):
    out = model(inputData)
    output, pred = out.max(1)
    print("Target", str(target.view(1,-1)).split("\n")[1])
    print("Prediction", str(pred.view(1,-1)).split("\n")[1])
    loss = F.nll_loss(out,target)
    print("Loss Data", loss.data[0])
    model.zero_grad()
    loss.backward()
    opt.step()
