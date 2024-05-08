from cmath import polar
from turtle import forward
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Towers2PModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_planes = 22
        self.hidden_planes = 256 
        self.plane_size = 25
        self.res_blocks = 19
        self.action_count = (
            self.plane_size +      #Build
            self.plane_size +      #Spawn
            self.plane_size * 4    #Move
            )
        self.num_players = 2

        self.start = nn.Sequential(
            nn.Conv2d(self.input_planes, self.hidden_planes, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(self.hidden_planes),
            nn.ReLU()
        )

        self.backBone = nn.ModuleList(
            [ResBlock(self.hidden_planes) for _ in range(0, self.res_blocks)]
        )

        self.policyHead = nn.Sequential(
            nn.Conv2d(self.hidden_planes, self.hidden_planes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.hidden_planes),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.hidden_planes * self.plane_size, self.action_count),
            nn.Softmax(dim=1)
        )

        self.valueHead = nn.Sequential(
            nn.Conv2d(self.hidden_planes, self.hidden_planes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.hidden_planes),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.hidden_planes * self.plane_size, self.num_players),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.start(x)
        for block in self.backBone:
            x = block(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value

class ResBlock(nn.Module):
    def __init__(self, hidden_planes):
        super().__init__()
        self.hidden_planes = hidden_planes
        self.conv1 = nn.Conv2d(self.hidden_planes, self.hidden_planes, kernel_size=3, padding=1, stride=1)
        self.batchN1 = nn.BatchNorm2d(self.hidden_planes)

        self.conv2 = nn.Conv2d(self.hidden_planes, self.hidden_planes, kernel_size=3, padding=1, stride=1)
        self.batchN2 = nn.BatchNorm2d(self.hidden_planes)
    
    def forward(self, x):
        residual = x
        x = F.relu(self.batchN1(self.conv1(x)))
        x = self.batchN2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x