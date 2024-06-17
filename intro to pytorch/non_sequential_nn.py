import numpy as np
import pandas as pd
import torch
import torch.nn as nn

#just like the NN network in neural_networks.py
class NN_Regression(nn.Module):
    def __init__(self):
        super(NN_Regression, self).__init__()

        #specify number of nodes per layer
        #note that each subsequent node need to share dimentionality with previous node before transitioning to a different dimentionality (projection)
        self.layer1 = nn.Linear(3, 16)
        self.layer2 = nn.Linear(16, 8)
        self.layer3 = nn.Linear(8, 4)
        self.layer4 = nn.Linear(4, 1)

        #some activation function to run between transitions
        self.relu = nn.ReLU()
    
    #specify feedforward operation 
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu()
        x = self.layer2(x)
        x = self.relu()
        x = self.layer3(x)
        x = self.relu()
        x = self.layer4(x)
        return x

class OneHidden(nn.Module):
    def __init__(self):
        super(OneHidden, self).__init__()
        self.layer1 = nn.Linear(2, 4)
        self.layer2 = nn.Linear(4, 1)

        self.relu = nn.ReLU()

    #specify number of nodes within hidden layers
    def __init__(self, numHiddenNodes):
        super(OneHidden, self).__init__()
        self.layer1 = nn.Linear(2, numHiddenNodes)
        self.layer2 = nn.Linear(numHiddenNodes, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x