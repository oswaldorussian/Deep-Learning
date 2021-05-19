import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import numpy as np

NUM_CLASSES = 21

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.conv2 = nn.Conv2d(64, 32, 3)
        self.conv3 = nn.Conv2d(32, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 26 * 26, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, NUM_CLASSES)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size()[0], 16 * 26 * 26)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    '''
    Build two different classifiers similar to the one provided by experimenting with different parts of the network
    each incorporating some or all of these suggestions:
        Add more conv layers
        Change the filter size (channel size, kernel size, etc)
        Reduce the number of FC layers
        Incorporate batch normalization
            e.g.,         
            self.conv1 = nn.Conv2d(input_channel_size, output_size, kernel_size, stride=stride)
            self.bn1 = nn.BatchNorm2d(channel_size)
            ...
            x = F.relu(self.bn1(self.conv1(x)))
        Add residual connections
            e.g., 
            x = F.relu(self.bn3(self.conv3(x)))
            residual = x
            x = F.relu(self.bn4(self.conv4(x)))
            x = F.relu(self.bn5(self.conv5(x)) + residual)
    
    You may also try suggestions not included in the above list
    
    Train the two different classifiers and compare them to the baseline
    '''

class ClassifierA(nn.Module):
    ## Replicate the code above and modify the network using the suggestions

    ##I am adding an additional convolutional layer and changing the filter size

    def __init__(self):
        super(ClassifierA, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.bnorm1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 32, 3)
        self.bnorm2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 26 * 26, 120)
        self.fc2 = nn.Linear(120, NUM_CLASSES)

    def forward(self, x):
        x = self.pool(F.relu(self.bnorm1(self.conv1(x))))
        x = self.pool(F.relu(self.bnorm2(self.conv2(x))))
        x = self.pool(F.relu((self.conv3(x)))
        x = x.view(x.size()[0], 16 * 26 * 26)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class ClassifierB(nn.Module):
    ## Replicate the code above and modify the network using the suggestions
    
    ##I am eliminating one of the fully-connected layers

    def __init__(self):
        super(ClassifierB, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.conv2 = nn.Conv2d(64, 32, 3)
        self.conv3 = nn.Conv2d(32, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 26 * 26, 120)
        self.fc2 = nn.Linear(120, NUM_CLASSES)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size()[0], 16 * 26 * 26)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x