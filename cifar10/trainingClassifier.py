import cv2
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Loading and Normalizing CIFAR10 images
transform = transforms.Compose([transforms.ToTensor()])#,
                               #transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

trainset = torchvision.datasets.CIFAR10(root = "./data", train =True,
                                        download =True, transform = transform)
trainLoader = torch.utils.data.DataLoader(trainset, batch_size = 8,
                                          shuffle = True)

testset = torchvision.datasets.CIFAR10(root = "./data", train = False,
                                       download = True, transform = transform)
testLoader = torch.utils.data.DataLoader(testset, batch_size = 20, 
                                         shuffle = True)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 
           'ship', 'truck')

# Show some images vefore preprocessing
def imshow(img):
    img = img/2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    
dataiter = iter(trainLoader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s' % classes[labels[j]] for j in range(8)))

# Define a CNN
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class modul(nn.Module):
    def __init__(self):
        super(modul,self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.h1 = nn.Linear(16*5*5, 120)
        self.h2 = nn.Linear(120,84)
        self.h3 = nn.Linear(84,10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,16*5*5)
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        x = self.h3(x)
        return x
    
net = modul()

# Define optimisation and loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr = 0.001, momentum = 0.9)
lossValue = []
# Train the neural network
for epoch in range(20):
    runningLoss = 0.0
    for i,data in enumerate(trainLoader, 0):
        inputs, labels = data
        #labels = labels.numpy()
        #labels = np.matrix(labels)
        #labels = torch.from_numpy(labels.transpose())
        inputs = Variable(inputs)
        labels = Variable(labels)
        
        optimizer.zero_grad()
        outputs = net(inputs)
        #_,pred = outputs.max(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        runningLoss += loss.data[0]
        if i%2000 == 1999:
            print("Loss: ", runningLoss/2000)
            lossValue.append(runningLoss/2000)
            runningLoss = 0.0
print("Finish Training!")

# Testing the training model
dataiter = iter(testLoader)
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ',' '.join('%5s' % classes[labels[j]] for j in range(20)))

outputs = net(Variable(images))
_,pred = torch.max(outputs.data,1)
pred = pred.numpy()
pred = pred[:,0]

print('Predicted: ',' '.join('%5s' % classes[pred[j]] for j in range(20)))
plt.figure()
plt.plot(lossValue)
