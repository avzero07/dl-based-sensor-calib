import torch
import torch.nn as nn
import torch.nn.Functional as F
import torch.optim as optim

class CNNBasic(nn.Module):

    def __init__(self):
        '''
        Define various layers used in the CNN. Not necessarily in
        order.
        '''
        super(CNNBasic, self).__init__()
        self.conv1 = nn.Conv2d(3,32,3,1)
        self.conv2 = nn.Conv2d(32,64,3,1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.mlp1 = nn.Linear(9216,128) # TODO: Determine actual size
        seld.mlp2 = nn.Linear(128,2)

        self.optimizer = optim.Adadelta(self.parameters(),lr=0.01)

    def forward(self,X):
        '''
        Stack layers in order and add activations to form the forward
        pass.
        '''
        X = self.conv1(X)
        X = F.relu(X)
        X = self.conv2(X)
        X = F.relu(X)
        X = F.max_pool2d(X,2)
        X = self.dropout1(X)
        X = torch.flatten(X,1)
        X = self.mlp1(X)
        X = F.relu(X)
        X = self.dropout2(X)
        X = self.mlp2(X)
        output = 
