import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# Discriminator net
class Discriminator(nn.Module):
    def __init__(self, classes, dim=128):
        super(Discriminator, self).__init__()
        # Initialize first two layers
        modelLayers = OrderedDict([
            ("conv1", nn.Conv2d(
                classes,
                dim,
                kernel_size=4,
                stride=2,
                padding=1
            )),
            ("relu1",  nn.LeakyReLU(0.2))
        ])
        index=0
        while (dim / 2**index != 4):
            modelLayers["conv{}".format(index)] = self._block(dim * (2**index),
                                                             dim * (2**(index+1)),
                                                             4,
                                                             2,
                                                             1 )
            index+=1
        # 1x1
        modelLayers["convFinal"] = nn.Conv2d(dim * (2**index), 1, kernel_size=4, stride=2, padding=0)
        modelLayers["activationFinal"] = nn.Sigmoid()
        self.discriminator = nn.Sequential(modelLayers)


    def forward(self, input, label):
        # First iteration : no labels
        return self.discriminator(input)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
# Generator net
class Generator(nn.Module):
    def __init__(self, classes, dim=512):
        super(Generator, self).__init__()

    def forward(self, input, label):
        return
