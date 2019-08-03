import torch
from torch import nn
from core.model.senet.base_module import ConvBlock, SEResBlock
class SE_RES(nn.Module):
    def __init__(self, in_chans, out_chans, n_resblocks, n_feats):
        super(SE_RES, self).__init__()
        m_head = [ConvBlock(in_chans, n_feats)]
        m_body = [SEResBlock(n_feats, n_feats) for i in range(n_resblocks)]
        m_tail = [ConvBlock(n_feats, out_chans)]

        self.conv = nn.Sequential(*(m_head+m_body+m_tail))
    def forward(self, x):
        return self.conv(x)

def get_model(name):
    return SE_RES
        

        