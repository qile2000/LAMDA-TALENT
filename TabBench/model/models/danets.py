import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import model.lib.danets.sparsemax as sparsemax
# Source: https://github.com/WhatAShot/DANet/blob/main/model/DANet.py
def initialize_glu(module, input_dim, output_dim):
    gain_value = np.sqrt((input_dim + output_dim) / np.sqrt(input_dim))
    torch.nn.init.xavier_normal_(module.weight, gain=gain_value)
    return

class GBN(torch.nn.Module):
    """
    Ghost Batch Normalization
    https://arxiv.org/abs/1705.08741
    """
    def __init__(self, input_dim, virtual_batch_size=512):
        super(GBN, self).__init__()
        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = nn.BatchNorm1d(self.input_dim)

    def forward(self, x):
        if self.training == True:
            chunks = x.chunk(int(np.ceil(x.shape[0] / self.virtual_batch_size)), 0)
            res = [self.bn(x_) for x_ in chunks]
            return torch.cat(res, dim=0)
        else:
            return self.bn(x)

class LearnableLocality(nn.Module):

    def __init__(self, input_dim, k):
        super(LearnableLocality, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.rand(k, input_dim)))
        self.smax = sparsemax.Entmax15(dim=-1)

    def forward(self, x):
        mask = self.smax(self.weight)
        masked_x = torch.einsum('nd,bd->bnd', mask, x)  # [B, k, D]
        return masked_x

class AbstractLayer(nn.Module):
    def __init__(self, base_input_dim, base_output_dim, k, virtual_batch_size, bias=True):
        super(AbstractLayer, self).__init__()
        self.masker = LearnableLocality(input_dim=base_input_dim, k=k)
        self.fc = nn.Conv1d(base_input_dim * k, 2 * k * base_output_dim, kernel_size=1, groups=k, bias=bias)
        initialize_glu(self.fc, input_dim=base_input_dim * k, output_dim=2 * k * base_output_dim)
        self.bn = GBN(2 * base_output_dim * k, virtual_batch_size)
        self.k = k
        self.base_output_dim = base_output_dim

    def forward(self, x):
        b = x.size(0)
        x = self.masker(x)  # [B, D] -> [B, k, D]
        x = self.fc(x.view(b, -1, 1))  # [B, k, D] -> [B, k * D, 1] -> [B, k * (2 * D'), 1]
        x = self.bn(x)
        chunks = x.chunk(self.k, 1)  # k * [B, 2 * D', 1]
        x = sum([F.relu(torch.sigmoid(x_[:, :self.base_output_dim, :]) * x_[:, self.base_output_dim:, :]) for x_ in chunks])  # k * [B, D', 1] -> [B, D', 1]
        return x.squeeze(-1)


class BasicBlock(nn.Module):
    def __init__(self, input_dim, base_outdim, k, virtual_batch_size, fix_input_dim, drop_rate):
        super(BasicBlock, self).__init__()
        self.conv1 = AbstractLayer(input_dim, base_outdim // 2, k, virtual_batch_size)
        self.conv2 = AbstractLayer(base_outdim // 2, base_outdim, k, virtual_batch_size)

        self.downsample = nn.Sequential(
            nn.Dropout(drop_rate),
            AbstractLayer(fix_input_dim, base_outdim, k, virtual_batch_size)
        )

    def forward(self, x, pre_out=None):
        if pre_out == None:
            pre_out = x
        out = self.conv1(pre_out)
        out = self.conv2(out)
        identity = self.downsample(x)
        out += identity
        return F.leaky_relu(out, 0.01)


class DANet(nn.Module):
    def __init__(self, input_dim, num_classes, n_layers, base_outdim, k, virtual_batch_size, dropout=0.1):
        super(DANet, self).__init__()
        params = {'base_outdim': base_outdim, 'k': k, 'virtual_batch_size': virtual_batch_size,
                  'fix_input_dim': input_dim, 'drop_rate': dropout}
        self.init_layer = BasicBlock(input_dim, **params)
        self.lay_num = n_layers
        self.layer = nn.ModuleList()
        for i in range((n_layers // 2) - 1):
            self.layer.append(BasicBlock(base_outdim, **params))
        self.drop = nn.Dropout(0.1)

        self.fc = nn.Sequential(nn.Linear(base_outdim, 256),
                                nn.ReLU(inplace=True),
                                nn.Linear(256, 512),
                                nn.ReLU(inplace=True),
                                nn.Linear(512, num_classes))

    def forward(self, x, x_cat = None):
        out = self.init_layer(x)
        for i in range(len(self.layer)):
            out = self.layer[i](x, out)
        out = self.drop(out)
        out = self.fc(out)
        return out