import torch
from torch import nn

from .utils import conv_block_3d as conv_block, up_conv_3d as up_conv


class UNet_3d(nn.Module):
    def __init__(self, input_dim=3, num_classes=1) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.Maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch=input_dim, out_ch=64)
        self.Conv2 = conv_block(in_ch=64, out_ch=128)
        self.Conv3 = conv_block(in_ch=128, out_ch=256)
        self.Conv4 = conv_block(in_ch=256, out_ch=512)
        self.Conv5 = conv_block(in_ch=512, out_ch=1024)

        self.Up5 = up_conv(in_ch=1024, out_ch=512)
        self.Up_conv5 = conv_block(in_ch=1024, out_ch=512)

        self.Up4 = up_conv(in_ch=512, out_ch=256)
        self.Up_conv4 = conv_block(in_ch=512, out_ch=256)

        self.Up3 = up_conv(in_ch=256, out_ch=128)
        self.Up_conv3 = conv_block(in_ch=256, out_ch=128)

        self.Up2 = up_conv(in_ch=128, out_ch=64)
        self.Up_conv2 = conv_block(in_ch=128, out_ch=64)

        self.Conv_1x1 = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        e1 = self.Conv1(x)  #

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        # decoding + concat path
        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1
