import torch
import torch.nn as nn
import torch.nn.functional as F

# Discriminator net
class Discriminator(nn.Module):
    def __init__(self, classes, dim=128):
        super(Discriminator, self).__init__()

    def forward(self, input, label):
        return

# Generator net
class Generator(nn.Module):
    def __init__(self, classes, dim=512):
        super(Generator, self).__init__()

    def forward(self, input, label):
        return
