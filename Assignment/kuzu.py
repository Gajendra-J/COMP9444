"""
   kuzu.py
   COMP9444, CSE, UNSW
   Functions filled in by: Gajendra Jayasekera z5260252
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        self.linear = nn.Linear(28*28, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = F.log_softmax(x, dim=1)
        return x

class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        nodes = 260
        self.hidden_layer = nn.Linear(28*28, nodes)
        self.output_layer = nn.Linear(nodes, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.hidden_layer(x)
        x = torch.tanh(x)
        x = self.output_layer(x)
        x = F.log_softmax(x, dim=1)
        return x

class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.max_pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(1568,260)
        self.fc2 = nn.Linear(260,10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x
