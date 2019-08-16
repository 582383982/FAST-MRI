import torch
import torch.nn as nn
import torch.nn.functional as F


def passthrough(x, **kwargs):
    return x

def ELUCons(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)

# normalization between sub-volumes is necessary
# for good performance
class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
        super(ContBatchNorm3d, self)._check_input_dim(input)

    def forward(self, input):
        # self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)


class LUConv(nn.Module):
    def __init__(self, nchan, elu):
        super(LUConv, self).__init__()
        self.relu1 = ELUCons(elu, nchan)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(nchan)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


def _make_nConv(nchan, depth, elu):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, elu))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(self, inChans, outChans, elu):
        super(InputTransition, self).__init__()
        self.outChans = outChans
        self.conv1 = nn.Conv3d(inChans, outChans, kernel_size=5, padding=2)
        self.bn1 = nn.InstanceNorm3d(outChans)
        self.relu1 = ELUCons(elu, outChans)

    def forward(self, x):
        # do we want a PRELU here as well?
        out = self.bn1(self.conv1(x))
        out = self.relu1(out)
        return out


class DownTransition(nn.Module):
    def __init__(self, inChans, nConvs, elu, dropout=False):
        super(DownTransition, self).__init__()
        outChans = 2*inChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=3, stride=(1, 2, 2), padding=1)
        self.bn1 = ContBatchNorm3d(outChans)
        self.do1 = passthrough
        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, dropout=False):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=(1,2,2), stride=(1, 2, 2))
        self.bn1 = nn.InstanceNorm3d(outChans // 2)
        self.do1 = passthrough
        self.do2 = nn.Dropout3d()
        self.relu1 = ELUCons(elu, outChans // 2)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x, skipx):
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        out = self.relu1(self.bn1(self.up_conv(out)))
        xcat = torch.cat((out, skipxdo), 1)
        out = self.ops(xcat)
        out = self.relu2(torch.add(out, xcat))
        return out


class OutputTransition(nn.Module):
    def __init__(self, inChans, outChans=1):
        super(OutputTransition, self).__init__()
        ch = inChans
        self.conv = nn.Sequential(
            nn.Conv3d(ch, ch//2, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.Conv3d(ch//2, ch//2, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.Conv3d(ch//2, outChans, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        )

    def forward(self, x):
        return self.conv(x)


class VNet(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, inChans, startChans, num_pool_layers=4, elu=True, nll=False):
        super(VNet, self).__init__()
        self.in_tr = InputTransition(inChans, startChans, elu)

        self.num_pool_layers = num_pool_layers
        after_down_convs = [1, 2, 3, 2]
        after_up_convs = [2, 2, 1, 1]

        self.down_sample_layers = nn.ModuleList([DownTransition(startChans, after_down_convs[0], elu)])
        ch = startChans*2
        for i in range(1, num_pool_layers):
            self.down_sample_layers += [DownTransition(ch, after_down_convs[i], elu)]
            ch *= 2
        
        self.up_sample_layers = nn.ModuleList([UpTransition(ch, ch, after_up_convs[0], elu)])

        for i in range(1, num_pool_layers):
            self.up_sample_layers += [UpTransition(ch, ch//2, after_up_convs[i], elu)]
            ch //= 2
        self.out_tr = OutputTransition(ch)
    def forward(self, x):
        
        output = self.in_tr(x)
        # print(output.size())

        stack = [output]
        for layer in self.down_sample_layers:
            output = layer(output)
            # print(output.size())
            stack.append(output)
        stack.pop()
        for layer in self.up_sample_layers:
            skip_x = stack.pop()
            output = layer(output, skip_x)
        output = self.out_tr(output)
        return output

def get_model(name):
    return VNet