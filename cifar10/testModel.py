from torchvision import transforms
from torch.autograd import Variable
import torchvision
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

# Preparing the preprocess step and the training 
normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225])
preprocess = transforms.Compose([transforms.Scale(50),
                                 transforms.CenterCrop(40),
                                 transforms.ToTensor(),
                                 normalize])

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

# Importer le modele
net = torch.load('trainedModel1.pt')
testingSet = datasets.ImageFolder(root = "./dataset/test_set",
                                  transform = preprocess)

testLoader = data.DataLoader(testingSet, shuffle = True, batch_size = 4)

classes = ('cats', 'dogs')
# Testing phase
dataiter = iter(testLoader)
images, labels = dataiter.next()
labels = labels.numpy()
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ',' '.join('%5s' % classes[labels[j]] for j in range(4)))

outputs = net(Variable(images.cuda()))
_,pred = torch.max(outputs.data,1)
pred = pred.numpy()
pred = pred[:,0]
print('Predicted: ',' '.join('%5s' % classes[pred[j]] for j in range(4)))

correct = 0
total = 0
for datum in testLoader:
    images, labels = datum
    outputs = net(Variable(images.cuda()))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (    100 * correct / total))