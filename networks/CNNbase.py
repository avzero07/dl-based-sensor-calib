import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from networks.common import get_conv_output_size, get_cascaded_output_size

class CNNBasic(nn.Module):

    def __init__(self,alpha=0,ip_dim=torch.Size([3,874,1164])):
        '''
        Define various layers used in the CNN. Not necessarily in
        order.
        '''
        #TODO: Sizes Should be Generic Based on Input_Dim
        # Frames from the car_dataset are 1164x874x3
        layer_dim_list = [
                (32,13,5,0),
                (64,13,5,0),
                (64,3,3,0)] # MaxPool layer below
        mlp1_dim = get_cascaded_output_size(ip_dim,layer_dim_list)

        super(CNNBasic, self).__init__()
        self.conv1 = nn.Conv2d(ip_dim[0],32,13,5)
        self.conv2 = nn.Conv2d(32,64,13,5)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.mlp1 = nn.Linear(mlp1_dim,128) # TODO: Determine actual size
        self.mlp2 = nn.Linear(128,2)

        self.optimizer = optim.Adadelta(self.parameters(),lr=0.01,weight_decay=alpha)

    def forward(self,X):
        '''
        Stack layers in order and add activations to form the forward
        pass.
        '''
        X = self.conv1(X)
        X = F.relu(X)
        X = self.conv2(X)
        X = F.relu(X)
        X = F.max_pool2d(X,3)
        X = self.dropout1(X)
        X = torch.flatten(X,1)
        X = self.mlp1(X)
        X = F.relu(X)
        X = self.dropout2(X)
        X = self.mlp2(X)
        return X
