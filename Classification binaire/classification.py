from torchvision import transforms
from torch.autograd import Variable
import torchvision

path = "F:/Projet fin d'Etudes/DogsCatsClassification/Convolutional_Neural_Networks/dataset"

# Preparing the preprocess step and the training 
normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225])
preprocess = transforms.Compose([transforms.Scale(50),
                                 transforms.CenterCrop(40),
                                 transforms.ToTensor(),
                                 normalize])

# Data Loading
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np


trainingSet = datasets.ImageFolder(root = "./dataset/training_set",
                                   transform = preprocess)

trainLoader = data.DataLoader(trainingSet, shuffle = True, batch_size = 4)

testingSet = datasets.ImageFolder(root = "./dataset/test_set",
                                  transform = preprocess)

testLoader = data.DataLoader(testingSet, shuffle = True, batch_size = 4)

classes = ('cats', 'dogs')

# Show some images vefore preprocessing
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) 
    
dataiter = iter(trainLoader)
images, labels = dataiter.next()
labels = labels.numpy()
#labels = labels[0,0]
imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s' % classes[labels[j]]for j in range(4)))

# Define a CNN with 2 convolutional layers and 1 layer of maxpooling
import torch.nn.functional as F

class modul(nn.Module):
    def __init__(self):
        super(modul,self).__init__()
        self.conv1 = nn.Conv2d(3,18,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(18,36,5)
        self.h1 = nn.Linear(36*7*7, 120)
        self.h2 = nn.Linear(120,84)
        self.h3 = nn.Linear(84,2)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # Flatenning steps 
        x = x.view(-1,self.num_flat_features(x))
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        # Linearize the input layer
        x = self.h3(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = modul().cuda()



criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr = 0.001, momentum = 0.9)
lossValue = []
# Train the neural network
for epoch in range(20):
    runningLoss = 0.0
    for i,datum in enumerate(trainLoader, 0):
        inputs, labels = datum
        #labels = labels.numpy()
        #labels = np.matrix(labels)
        #labels = torch.from_numpy(labels.transpose())
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
        
        optimizer.zero_grad()
        outputs = net(inputs)
        #_,pred = outputs.max(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        runningLoss += loss.data[0]
        if i%500 == 499:
            print("Loss: ", runningLoss/500)
            lossValue.append(runningLoss/500)
            runningLoss = 0.0
print("Finish Training!")
torch.save(net,'trainedModel1.pt')

plt.figure()
plt.plot(lossValue)
plt.title('Les pertes')