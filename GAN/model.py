import torch
import torch.nn as nn
from collections import OrderedDict

# Discriminator net
class Discriminator(nn.Module):
    def __init__(self, channels_img, features_disc):
        super(Discriminator, self).__init__()
        # Initialize first two layers
        # modelLayers = OrderedDict([
        #     ("conv1", nn.Conv2d(
        #         channels_img,
        #         features_disc,
        #         kernel_size=4,
        #         stride=2,
        #         padding=1
        #     )),
        #     ("relu1",  nn.LeakyReLU(0.2))
        # ])
        # index=0
        # while (features_disc / 2**index != 8):
        #     print(2**index)
        #     modelLayers["conv{}".format(index)] = self._block(features_disc * (2**index),
        #                                                      features_disc * (2**(index+1)),
        #                                                      4,
        #                                                      2,
        #                                                      1 )
        #     index+=1
        # 1x1
        # modelLayers["convFinal"] = nn.Conv2d(features_disc * 8, 1, kernel_size=4, stride=2, padding=0)
        # modelLayers["activationFinal"] = nn.Sigmoid()
        self.discriminator = nn.Sequential(
            nn.Conv2d(
                channels_img, features_disc, kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2),
            self._block(features_disc, features_disc * 2, 4 ,2  ,1),
            self._block(features_disc * 2, features_disc * 4, 4 ,2 ,1),
            self._block(features_disc * 4, features_disc * 8, 4 ,2 ,1),
            nn.Conv2d(
                features_disc * 8, 1, kernel_size=4, stride=2, padding=0
            ),
        )


    def forward(self, input):
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
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True), # LayerNorm == InstanceNorm?
            nn.LeakyReLU(0.2)
        )
# Generator net
class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_gen):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # Input : N x z_features_gen x 1 x 1
            self._block(z_dim, features_gen * 16, 4, 1, 0), # N x features_gen x 16 x 4 x 4
            self._block(features_gen * 16, features_gen * 8, 4, 2, 1), # 8x8
            self._block(features_gen * 8, features_gen * 4, 4, 2, 1), # 16x16
            self._block(features_gen * 4, features_gen * 2, 4, 2, 1), # 32x32
            nn.ConvTranspose2d(
                features_gen * 2, channels_img, 4, 2, 1
            ),
            nn.Tanh(),
        )

    def forward(self, input):
        return self.gen(input)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        # We upscale here
        return  nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False, # Because we are batching
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(),
        )

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def test():
    N, in_channels, H, W = 8, 3, 64, 64
    z_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(in_channels, 8)
    initialize_weights(disc)
    assert disc(x).shape == (N, 1 ,1 ,1)
    gen = Generator(z_dim, in_channels, 8)
    z = torch.randn((N, z_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W)
    print("Successful test")

test()
