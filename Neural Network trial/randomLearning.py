import torch
from torch import autograd, nn, optim
import torch.nn.functional as F

batchSize = 5
inputSize = 6
hiddenSize = 5
numClass = 2
learningRate = 0.001

torch.manual_seed(123)
input = autograd.Variable(torch.rand(batchSize, inputSize))
target = autograd.Variable((torch.rand(batchSize)*numClass).long())

class Net(nn.Module):
    def __init__(self, inputSize, hiddenSize, numClass):
        super().__init__()
        self.h1 = nn.Linear(inputSize, hiddenSize)
        self.h2 = nn.Linear(hiddenSize, numClass)
        
    def forward(self,x):
        x=self.h1(x)
        x=F.relu(x)
        x=self.h2(x)
        x=F.softmax(x)
        return x

model = Net(inputSize = inputSize, hiddenSize = hiddenSize, numClass = numClass)
opt = optim.Adam(params = model.parameters(), lr = learningRate)

for epoche in range(5000):
    out = model(input)
    output, pred = out.max(1)
    print("Target", str(target.view(1,-1)).split("\n")[1])
    print("Prediction", str(pred.view(1,-1)).split("\n")[1])
    loss = F.nll_loss(out,target)
    print("Loss Data", loss.data[0])
    model.zero_grad()
    loss.backward()
    opt.step()
    