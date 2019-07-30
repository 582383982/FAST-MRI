import torch
from torch import nn
class UpProject(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_sizes=[6, 8], strides=[2, 4], padding=2):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.conv1 = nn.Conv2d(in_channels, out_chans, kernel_size=kernel_sizes[0], padding=padding)
        self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=kernel_sizes[1], stride=strides[1], padding=padding)
    pass