import torch.nn as nn
import torch.nn.functional as F
import torch
import os
        
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
                nn.ReflectionPad2d(padding=1),
                nn.Conv2d(in_channels, in_channels, kernel_size=3),
                nn.InstanceNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.ReflectionPad2d(padding=1),
                nn.Conv2d(in_channels, in_channels, kernel_size=3),
                nn.InstanceNorm2d(in_channels)
        )


    def forward(self, x):
        return x + self.block(x)


class Discriminator(nn.Module):

    def __init__(self, channel, disc_n_filters):
        super(Discriminator, self).__init__()

        self.kernel_size = 4;
        self.leakyrelu_negative_slop = 0.2
        
        def conv4(in_channel, out_channel, stride = 2, norm=True):
            ret = []
            ret += [nn.Conv2d(in_channel, out_channel,
                             kernel_size=self.kernel_size,
                             stride=stride, padding=1)]
            if norm:
                ret += [nn.InstanceNorm2d(out_channel)]
            ret += [nn.LeakyReLU(self.leakyrelu_negative_slop, inplace=True)]

            return ret
            

        model = []
        model += conv4(channel, disc_n_filters, stride=2, norm=False)
        model += conv4(disc_n_filters, disc_n_filters * 2, stride=2)
        model += conv4(disc_n_filters * 2, disc_n_filters * 4, stride=2)
        model += conv4(disc_n_filters * 4, disc_n_filters * 8, stride=2)

        model += [nn.Conv2d(disc_n_filters * 8, 1, kernel_size=self.kernel_size, padding=1)]
        self.model = nn.Sequential(*model)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        x = self.sigmoid(x)
        return x
        # Average pooling and flatten
        #return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


class Generator(nn.Module):
    def __init__(self, in_channels, out_channels,
                 gen_n_filters, n_residual_blocks = 9):
        super(Generator, self).__init__()

        # Start
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(in_channels, gen_n_filters, kernel_size=7),
                 nn.InstanceNorm2d(gen_n_filters),
                 nn.ReLU()]
        
        # Downsampling
        in_feature = gen_n_filters
        out_feature = in_feature * 2
        for _ in range(2):
            model += [nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=2, padding=1),
                      nn.InstanceNorm2d(out_feature),
                      nn.ReLU()]
            in_feature = out_feature
            out_feature = in_feature * 2

        # Residual block
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_feature)]

        # Upsampling
        out_feature = in_feature // 2
        for _ in range(2):
            model += [nn.ConvTranspose2d(in_feature, out_feature, kernel_size=3, stride=2,
                                         padding=1, output_padding=1),
                      nn.InstanceNorm2d(out_feature),
                      nn.ReLU()]
            in_feature = out_feature
            out_feature = in_feature // 2

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(gen_n_filters, out_channels, kernel_size=7),
                  nn.Tanh()]
                  
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

