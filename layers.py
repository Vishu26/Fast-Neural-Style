from torch import nn
import numpy as np
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding=0):
        super(ResBlock, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        fx = self.conv3x3(x)
        fx = self.batchnorm(fx)
        fx = F.relu(fx)
        fx = self.conv3x3(fx)
        fx = self.batchnorm(fx)
        return fx + x


class ResBlockCenterCrop(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlockCenterCrop, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        fx = self.conv3x3(x)
        fx = self.batchnorm(fx)
        fx = F.relu(fx)
        fx = self.conv3x3(fx)
        fx = self.batchnorm(fx)
        return fx + x[:, :, 2:-2, 2:-2]


class ResInstanceCenterCrop(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResInstanceCenterCrop, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3)
        self.instancenorm = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        fx = self.conv3x3(x)
        fx = self.instancenorm(fx)
        fx = F.relu(fx)
        fx = self.conv3x3(fx)
        fx = self.instancenorm(fx)
        return fx + x[:, :, 2:-2, 2:-2]


class ConvBatchRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1):
        super(ConvBatchRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              padding=padding, stride=stride)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(self.batchnorm(x))
        return x


class ConvInstanceRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1):
        super(ConvInstanceRelu, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              padding=padding, stride=stride)
        self.instancenorm = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        x = self.reflection_pad(x)
        x = self.conv(x)
        x = F.relu(self.instancenorm(x))
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(UpBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.instancenorm = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        x = self.upsample(x)
        x = self.reflection_pad(x)
        x = self.conv(x)
        x = self.instancenorm(x)
        return F.relu(x)


class DeConvBatchRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 padding=0, stride=1, output_padding=0):
        super(DeConvBatchRelu, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                         padding=padding, stride=stride, output_padding=output_padding)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.deconv(x)
        x = F.relu(self.batchnorm(x))
        return x[:, :, 1:, 1:]  # hack to upsample by 2 (crop one pixel for 3x3 filter size)


class DeConvInstanceRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 padding=0, stride=1, output_padding=0):
        super(DeConvBatchRelu, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                         padding=padding, stride=stride, output_padding=output_padding)
        self.instancenorm = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        x = self.deconv(x)
        x = F.relu(self.instancenorm(x))
        return x[:, :, 1:, 1:]  # hack to upsample by 2 (crop one pixel for 3x3 filter size)
