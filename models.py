import torch.nn.functional as F
from layers import *
from torch import nn
from torchvision.models import vgg16


class LossNetwork(nn.Module):
    def __init__(self):
        super(LossNetwork, self).__init__()
        vgg = vgg16(pretrained=True)
        features = list(vgg.features)
        self.relu1_2 = nn.Sequential(*features[:4])
        self.relu2_2 = nn.Sequential(*features[4:9])
        self.relu3_3 = nn.Sequential(*features[9:16])
        self.relu4_3 = nn.Sequential(*features[16:23])

    def forward(self, x):
        relu1_2 = self.relu1_2(x)
        relu2_2 = self.relu2_2(relu1_2)
        relu3_3 = self.relu3_3(relu2_2)
        relu4_3 = self.relu4_3(relu3_3)
        return dict(relu1_2=relu1_2,
                    relu2_2=relu2_2,
                    relu3_3=relu3_3,
                    relu4_3=relu4_3)


class FastStyleTransfer(nn.Module):
    def __init__(self):
        super(FastStyleTransfer, self).__init__()
        self.reflection_padding = nn.ReflectionPad2d(40)
        self.conv1 = ConvBatchRelu(3, 32, kernel_size=9, padding=4)
        self.conv2 = ConvBatchRelu(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = ConvBatchRelu(64, 128, kernel_size=3, stride=2, padding=1)
        self.resblock = ResBlockCenterCrop(128, 128)
        self.deconv1 = DeConvBatchRelu(128, 64, kernel_size=3, stride=2)
        self.deconv2 = DeConvBatchRelu(64, 32, kernel_size=3, stride=2)
        self.conv_out = nn.Conv2d(32, 3, kernel_size=9, padding=4)

    def forward(self, x):
        x = self.reflection_padding(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.resblock(x)
        x = self.resblock(x)
        x = self.resblock(x)
        x = self.resblock(x)
        x = self.resblock(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.conv_out(x)
        return F.tanh(x)


class FastStyleTransfer2(nn.Module):
    def __init__(self):
        super(FastStyleTransfer2, self).__init__()
        self.reflection_padding = nn.ReflectionPad2d(40)
        self.conv1 = ConvInstanceRelu(3, 32, kernel_size=9)
        self.conv2 = ConvInstanceRelu(32, 64, kernel_size=3, stride=2)
        self.conv3 = ConvInstanceRelu(64, 128, kernel_size=3, stride=2)
        self.resblock = ResInstanceCenterCrop(128, 128)
        self.upblock1 = UpBlock(128, 64, kernel_size=3)
        self.upblock2 = UpBlock(64, 32, kernel_size=3)
        self.reflection_pad_out = nn.ReflectionPad2d(4)
        self.conv_out = nn.Conv2d(32, 3, kernel_size=9)

    def forward(self, x):
        x = self.reflection_padding(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.resblock(x)
        x = self.resblock(x)
        x = self.resblock(x)
        x = self.resblock(x)
        x = self.resblock(x)
        x = self.upblock1(x)
        x = self.upblock2(x)
        x = self.reflection_pad_out(x)
        x = self.conv_out(x)
        return F.tanh(x)
