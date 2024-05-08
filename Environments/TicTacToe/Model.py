from cmath import polar
from turtle import forward
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TicTacToeModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.start = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.backBone = nn.ModuleList(
            [ResBlock() for i in range(0, 10)]
        )

        self.policyHead = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 9, 9),
            nn.Softmax(dim=1)
        )

        self.valueHead = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 9, 2),
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
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1)
        self.batchN1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1)
        self.batchN2 = nn.BatchNorm2d(32)
    
    def forward(self, x):
        residual = x
        x = F.relu(self.batchN1(self.conv1(x)))
        x = self.batchN2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x