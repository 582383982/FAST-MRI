import torch
from torch import nn
from core.model import common
class UpProject(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size=6, stride=2, padding=2):
        super(UpProject, self).__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.conv1 = common.DeconvBlock(in_chans, out_chans, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = common.ConvBlock(out_chans, out_chans, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv3 = common.DeconvBlock(out_chans, out_chans, kernel_size=kernel_size, stride=stride, padding=padding)
        
    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        F2 = self.conv1(input)
        F3 = self.conv2(F2)
        F_residual = input + F3
        F5 = self.conv3(F_residual)
        F4 = F2+F5
        return F4


class DownProject(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size=6, stride=2, padding=2):
        super(DownProject, self).__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.conv1 = common.ConvBlock(in_chans, out_chans, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = common.DeconvBlock(out_chans, out_chans, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv3 = common.ConvBlock(out_chans, out_chans, kernel_size=kernel_size, stride=stride, padding=padding)
    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        F2 = self.conv1(input)
        F3 = self.conv2(F2)
        F_residual = input + F3
        F5 = self.conv3(F_residual)
        F4 = F2+F5
        return F4
    pass

class ResidualBlock(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.residual = nn.Sequential(
            common.ConvBlock(in_chans, out_chans, kernel_size, stride, padding),
            nn.PReLU(),
            common.ConvBlock(out_chans, out_chans, kernel_size, stride, padding),
        )
    def forward(self, input):
        output = self.residual(input)
        output = output + input
        return output

class CPRN(nn.Module):
    def __init__(self, in_chans=1, out_chans=1, num_cp=6, cp_chans=32, num_res=16, res_chans=64):
        super(CPRN, self).__init__()
        self.num_cp = num_cp
        self.num_res = num_res

        self.conv1 = nn.Conv2d(in_chans, cp_chans, 3,1,1)

        self.up_projections = nn.Sequential(*[UpProject(cp_chans, cp_chans) for i in range(num_cp)])
        self.down_projections = nn.Sequential(*[DownProject(cp_chans, cp_chans) for i in range(num_cp)])

        self.conv2 = nn.Conv2d(cp_chans, res_chans, 3,1,1)

        self.residuals = nn.Sequential(*[ResidualBlock(res_chans, res_chans) for i in range(num_res)])

        self.final_conv = nn.Sequential(
            nn.Conv2d(res_chans, res_chans // 2, kernel_size=1),
            nn.Conv2d(res_chans // 2, out_chans, kernel_size=1),
            nn.Conv2d(out_chans, out_chans, kernel_size=1),
        )
    
    def forward(self, input):
        input = self.conv1(input)
        sum_up = self.up_projections[0](input)
        sum_down = self.down_projections[0](sum_up)
        for i in range(1, self.num_cp):
            o_up = self.up_projections[i](sum_down)
            sum_up = sum_up + o_up
            o_down = self.down_projections[i](sum_up)
            sum_down = sum_down + o_down
        FR = self.conv2(sum_down)
        FR_res = self.residuals(FR)
        output = FR + FR_res
        output = self.final_conv(output)
        return output

class CPRN_Test(nn.Module):
    def __init__(self, in_chans=1, out_chans=1, num_cp=6, cp_chans=32, num_res=16, res_chans=64):
        super().__init__()
        self.up = UpProject(1,2)
        self.down = DownProject(2, 2)
        self.conv = nn.Conv2d(2,1,3,1,1)
    def forward(self, x):
        x = self.up(x)
        x = self.down(x)
        return self.conv(x)

def get_model(name):
    if name == 'cprn':
        return CPRN
    return None
            
# def forward(self, input):
#     out_conv = []
#     out_convtrans = []
#     o_up = self.up_projections[0](input)
#     o_down = self.down_projections[0][o_up]
#     out_conv.append(o_up)
#     out_convtrans.append(o_down)
#     for i in range(1, self.num_cp):
#         o_up = self.up_projections[i](o_down)
#         o_down = self.down_projections[i][o_up]
#         out_conv.append(o_up)
#         out_convtrans.append(o_down)