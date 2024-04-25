import torch
import torch.nn as nn
import torch.nn.functional as F
from .complexPyTorch.complexLayers import *
from .complexPyTorch.complexFunctions import *

def Complex_MSELoss(data, label):

    mat = (data - label).abs()
    l = torch.sum( mat*mat )

    return l

class ComplexFCN(nn.Module):
    def __init__(self, in_channels, out_channels, num_hidden=1000):
        super(ComplexFCN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_hidden = num_hidden

        self.linear1 = ComplexLinear(self.in_channels, self.num_hidden)
        # self.ComplexReLU1 = ComplexReLU()
        self.linear2 = ComplexLinear(self.num_hidden, self.out_channels)
        # self.Complex_sigmoid1 = complex_sigmoid()


    def forward(self, x):
        x1 = self.linear1(x)
        x1 = complex_relu(x1)
        x2 = self.linear2(x1)
        x2 = complex_sigmoid(x2)

        return x2

    def use_checkpointing(self):
        self.linear1 = torch.utils.checkpoint(self.linear1)
        self.linear2 = torch.utils.checkpoint(self.linear2)


class ComplexUnet(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super(ComplexUnet, self).__init__()
        self.n_channels = in_channels
        self.n_classes = out_channels
        self.bilinear = bilinear

        self.inc = (ComplexDoubleConv(self.n_channels, 16))
        self.down1 = (ComplexDown(16, 32))
        self.down2 = (ComplexDown(32, 64))
        # self.down3 = (ComplexDown(64, 128))
        factor = 2 if bilinear else 1
        self.down3 = (ComplexDoubleConv(64, 128 // factor))
        self.up1 = (ComplexUp(128, 64 // factor, bilinear))
        self.up2 = (ComplexUp(64, 32 // factor, bilinear))
        self.up3 = (ComplexUp(32, 16 // factor, bilinear))
        # self.up4 = (ComplexUp(16, 1, bilinear))
        self.outc = (ComplexOutConv(16, self.n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # x5 = self.down4(x4)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        # x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)



class ComplexDoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            ComplexConv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            ComplexBatchNorm2d(mid_channels),
            ComplexReLU(),
            ComplexConv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            ComplexBatchNorm2d(out_channels),
            ComplexReLU()
        )

    def forward(self, x):
        return self.double_conv(x)


class ComplexDown(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            ComplexMaxPool2d(2),
            ComplexDoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class ComplexUp(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = ComplexDoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = ComplexConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = ComplexDoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ComplexOutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ComplexOutConv, self).__init__()
        self.conv = ComplexConv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
