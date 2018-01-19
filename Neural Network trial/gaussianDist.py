import torch 
from torchvision.transforms import ToTensor
import numpy as np
import random

# Building random datasets of 100 elements in each class
# Building 1st dataset
n = 100
values1 = []
frequencies1 = {}
while len(values1) < n:
    value = random.gauss(5, 4)
    if 0 < value < 10:
        frequencies1[int(value)] = frequencies1.get(int(value), 0) + 1
        values1.append(value)

values1 = np.array(values1)
label1 = np.zeros(100).astype(int)
class1 = np.array([values1,label1]).T

# Building 2nd dataset
values2 = []
frequencies2 = {}
while len(values2) < n:
    value = random.gauss(15, 4)
    if 8 < value < 18:
        frequencies2[int(value)] = frequencies2.get(int(value), 0) + 1
        values2.append(value)

values2 = np.array(values2)
label2 = np.ones(100).astype(int)
class2 = np.array([values2,label2]).T

# Concatenate and permutate the 2 classes together
data = np.concatenate([class1,class2])
data = np.random.permutation(data)

# Transform to tensor
# data = torch.from_numpy(data)
