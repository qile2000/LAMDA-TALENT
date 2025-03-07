import torch
import numpy as np
import torch.nn as nn
import TALENT.model.lib.tabcaps.model.sparsemax as sparsemax
import torch.nn.functional as F

def initialize_glu(module, input_dim, output_dim):
    gain_value = np.sqrt((input_dim + output_dim) / np.sqrt(input_dim))
    torch.nn.init.xavier_normal_(module.weight, gain=gain_value)
    return

class GBN(nn.Module):
    """
    Ghost Batch Normalization
    https://arxiv.org/abs/1705.08741
    """

    def __init__(self, input_dim, virtual_batch_size=256):
        super(GBN, self).__init__()
        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = nn.BatchNorm1d(self.input_dim)

    def forward(self, x):
        chunks = x.chunk(int(np.ceil(x.shape[0] / self.virtual_batch_size)), 0)
        res = [self.bn(x_) for x_ in chunks]
        return torch.cat(res, dim=0)
        # return self.bn(x)

class LearnableLocality(nn.Module):
    def __init__(self, input_dim, n_path):
        super(LearnableLocality, self).__init__()
        self.weight = nn.Parameter(torch.rand((n_path, input_dim)))
        # self.smax = sparsemax.Sparsemax(dim=-1)
        self.smax = sparsemax.Entmax15(dim=-1)

    def forward(self, x):
        mask = self.smax(self.weight)
        masked_x = torch.einsum('nd,bd->bnd', mask, x)  # [B, n_path, D]
        return masked_x

class AbstractLayer(nn.Module):
    def __init__(self, base_input_dim, base_output_dim, n_path, virtual_batch_size=256):
        super(AbstractLayer, self).__init__()
        self.masker = LearnableLocality(input_dim=base_input_dim, n_path=n_path)
        self.fc = nn.Conv1d(base_input_dim * n_path, 2 * n_path * base_output_dim, kernel_size=1, groups=n_path)
        initialize_glu(self.fc, input_dim=base_input_dim * n_path, output_dim=2 * n_path * base_output_dim)
        self.n_path = n_path
        self.base_output_dim = base_output_dim
        self.bn = GBN(2 * base_output_dim * n_path, virtual_batch_size)
        # self.bn = nn.LayerNorm(2 * base_output_dim * n_path)


    def forward(self, x):
        b = x.size(0)
        x = self.masker(x)  # [B, D] -> [B, n_path, D]
        x = self.fc(x.view(b, -1, 1)) # [B, n_path, D] -> [B, n_path * D, 1] -> [B, n_path * (2 * D'), 1]
        x = self.bn(x.squeeze())
        chunks = x.chunk(self.n_path, 1) # n_path * [B, 2 * D', 1]
        x = [F.relu(torch.sigmoid(x_[:, :self.base_output_dim]) * x_[:, self.base_output_dim:]) for x_ in chunks] # n_path * [B, D', 1]
        # x = torch.cat(x, dim=1).squeeze()
        return sum(x)
