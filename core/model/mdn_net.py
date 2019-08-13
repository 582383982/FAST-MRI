import torch
import torch.nn as nn
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2+dilation-1, dilation=dilation)
        self.activation = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.activation(self.conv(x))

class MDN_Block(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(MDN_Block, self).__init__()
        self.bn_0 = nn.BatchNorm2d(in_channels)
        self.conv_0 = ConvBlock(in_channels, 32, 9, stride=1, dilation=1)
        self.conv_1 = ConvBlock(32, 64, 3, stride=1, dilation=2)
        self.conv_2 = ConvBlock(64, 32, 3, stride=1, dilation=3)
        self.conv_3 = ConvBlock(32, 64, 3, stride=1, dilation=2)
        self.conv_4 = ConvBlock(64, 32, 3, stride=1, dilation=3)
        self.conv_5 = ConvBlock(32, 64, 3, stride=1, dilation=2)
        self.bn_5 = nn.BatchNorm2d(64)
        
        self.conv_6 = ConvBlock(64, 32, 3, stride=1, dilation=3)
        self.bn_6 = nn.BatchNorm2d(32)

        self.cat_conv = nn.Conv2d(32+64+1, 1, 3, dilation=1, padding=1)
    def forward(self, x):
        feats_bn0 = self.bn_0(x)
        feats_0 = self.conv_0(feats_bn0)
        feats_1 = self.conv_1(feats_0)

        feats_2 = self.conv_2(feats_1)
        res_02 = feats_0 + feats_2

        feats_3 = self.conv_3(res_02)
        res_13 = feats_1 + feats_3

        feats_4 = self.conv_4(res_13)
        res_24 = feats_2 + feats_4

        feats_5 = self.conv_5(res_24)
        res_35 = feats_3 + feats_5
        feats_bn5 = self.bn_5(res_35)

        feats_6 = self.conv_6(feats_5)
        res_46 = feats_4 + feats_6
        feats_bn6 = self.bn_6(res_46)
        # print(feats_bn0.size(), feats_1.size(), feats_2.size(), feats_3.size(), feats_bn5.size(), feats_bn6.size())
        feats_cat = torch.cat([feats_bn0, feats_bn5, feats_bn6], 1)
        return self.cat_conv(feats_cat)

class MDN_NET(nn.Module):
    def __init__(self, n_mdn=3):
        super(MDN_NET, self).__init__()
        self.block_1 = MDN_Block()
        self.block_2 = MDN_Block()
        self.block_3 = MDN_Block()
    def forward(self, x):
        feats_1 = self.block_1(x)
        # res_1 = feats_1 + x
        feats_2 = self.block_2(feats_1)
        # res_2 = res_1 + x
        feats_3 = self.block_2(feats_2)
        # res_3 = res_1 + x
        return feats_3

def get_model(name):
    return MDN_NET



