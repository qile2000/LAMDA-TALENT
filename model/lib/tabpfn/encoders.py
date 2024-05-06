import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

def torch_masked_mean(x, mask, dim=0, return_share_of_ignored_values=False):
    """
    Returns the mean of a torch tensor and only considers the elements, where the mask is true.
    If return_share_of_ignored_values is true it returns a second tensor with the percentage of ignored values
    because of the mask.
    """
    num = torch.where(mask, torch.full_like(x, 1), torch.full_like(x, 0)).sum(dim=dim)
    value = torch.where(mask, x, torch.full_like(x, 0)).sum(dim=dim)
    if return_share_of_ignored_values:
        return value / num, 1.-num/x.shape[dim]
    return value / num

def torch_masked_std(x, mask, dim=0):
    """
    Returns the std of a torch tensor and only considers the elements, where the mask is true.
    If get_mean is true it returns as a first Tensor the mean and as a second tensor the std.
    """
    num = torch.where(mask, torch.full_like(x, 1), torch.full_like(x, 0)).sum(dim=dim)
    value = torch.where(mask, x, torch.full_like(x, 0)).sum(dim=dim)
    mean = value / num
    mean_broadcast = torch.repeat_interleave(mean.unsqueeze(dim), x.shape[dim], dim=dim)
    quadratic_difference_from_mean = torch.square(torch.where(mask, mean_broadcast - x, torch.full_like(x, 0)))
    return torch.sqrt(torch.sum(quadratic_difference_from_mean, dim=dim) / (num - 1))

def torch_nanmean(x, dim=0, return_nanshare=False):
    return torch_masked_mean(x, ~torch.isnan(x), dim=dim, return_share_of_ignored_values=return_nanshare)

def torch_nanstd(x, dim=0):
    return torch_masked_std(x, ~torch.isnan(x), dim=dim)

def normalize_data(data, normalize_positions=-1):
    if normalize_positions > 0:
        mean = torch_nanmean(data[:normalize_positions], dim=0)
        std = torch_nanstd(data[:normalize_positions], dim=0) + .000001
    else:
        mean = torch_nanmean(data, dim=0)
        std = torch_nanstd(data, dim=0) + .000001
    data = (data - mean) / std
    data = torch.clip(data, min=-100, max=100)

    return data

class _PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.device_test_tensor = nn.Parameter(torch.tensor(1.))

    def forward(self, x):# T x B x num_features
        assert self.d_model % x.shape[-1]*2 == 0
        d_per_feature = self.d_model // x.shape[-1]
        pe = torch.zeros(*x.shape, d_per_feature, device=self.device_test_tensor.device)
        #position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        interval_size = 10
        div_term = (1./interval_size) * 2*math.pi*torch.exp(torch.arange(0, d_per_feature, 2, device=self.device_test_tensor.device).float()*math.log(math.sqrt(2)))
        #print(div_term/2/math.pi)
        pe[..., 0::2] = torch.sin(x.unsqueeze(-1) * div_term)
        pe[..., 1::2] = torch.cos(x.unsqueeze(-1) * div_term)
        return self.dropout(pe).view(x.shape[0],x.shape[1],self.d_model)


Positional = lambda _, emsize: _PositionalEncoding(d_model=emsize)

class EmbeddingEncoder(nn.Module):
    def __init__(self, num_features, em_size, num_embs=100):
        super().__init__()
        self.num_embs = num_embs
        self.embeddings = nn.Embedding(num_embs * num_features, em_size, max_norm=True)
        self.init_weights(.1)
        self.min_max = (-2,+2)

    @property
    def width(self):
        return self.min_max[1] - self.min_max[0]

    def init_weights(self, initrange):
        self.embeddings.weight.data.uniform_(-initrange, initrange)

    def discretize(self, x):
        split_size = self.width / self.num_embs
        return (x - self.min_max[0] // split_size).int().clamp(0, self.num_embs - 1)

    def forward(self, x):  # T x B x num_features
        x_idxs = self.discretize(x)
        x_idxs += torch.arange(x.shape[-1], device=x.device).view(1, 1, -1) * self.num_embs
        # print(x_idxs,self.embeddings.weight.shape)
        return self.embeddings(x_idxs).mean(-2)



Linear = nn.Linear
MLP = lambda num_features, emsize: nn.Sequential(nn.Linear(num_features+1,emsize*2),
                                                 nn.ReLU(),
                                                 nn.Linear(emsize*2,emsize))

class NanHandlingEncoder(nn.Module):
    def __init__(self, num_features, emsize, keep_nans=True):
        super().__init__()
        self.num_features = 2 * num_features if keep_nans else num_features
        self.emsize = emsize
        self.keep_nans = keep_nans
        self.layer = nn.Linear(self.num_features, self.emsize)

    def forward(self, x):
        if self.keep_nans:
            x = torch.cat([torch.nan_to_num(x, nan=0.0), normalize_data(torch.isnan(x) * -1
                                                          + torch.logical_and(torch.isinf(x), torch.sign(x) == 1) * 1
                                                          + torch.logical_and(torch.isinf(x), torch.sign(x) == -1) * 2
                                                          )], -1)
        else:
            x = torch.nan_to_num(x, nan=0.0)
        return self.layer(x)


class Linear(nn.Linear):
    def __init__(self, num_features, emsize, replace_nan_by_zero=False):
        super().__init__(num_features, emsize)
        self.num_features = num_features
        self.emsize = emsize
        self.replace_nan_by_zero = replace_nan_by_zero

    def forward(self, x):
        if self.replace_nan_by_zero:
            x = torch.nan_to_num(x, nan=0.0)
        return super().forward(x)

    def __setstate__(self, state):
        super().__setstate__(state)
        self.__dict__.setdefault('replace_nan_by_zero', True)


class Conv(nn.Module):
    def __init__(self, input_size, emsize):
        super().__init__()
        self.convs = torch.nn.ModuleList([nn.Conv2d(64 if i else 1, 64, 3) for i in range(5)])
        self.linear = nn.Linear(64,emsize)

    def forward(self, x):
        size = math.isqrt(x.shape[-1])
        assert size*size == x.shape[-1]
        x = x.reshape(*x.shape[:-1], 1, size, size)
        for conv in self.convs:
            if x.shape[-1] < 4:
                break
            x = conv(x)
            x.relu_()
        x = nn.AdaptiveAvgPool2d((1,1))(x).squeeze(-1).squeeze(-1)
        return self.linear(x)


class CanEmb(nn.Embedding):
    def __init__(self, num_features, num_embeddings: int, embedding_dim: int, *args, **kwargs):
        assert embedding_dim % num_features == 0
        embedding_dim = embedding_dim // num_features
        super().__init__(num_embeddings, embedding_dim, *args, **kwargs)

    def forward(self, x):
        lx = x.long()
        assert (lx == x).all(), "CanEmb only works with tensors of whole numbers"
        x = super().forward(lx)
        return x.view(*x.shape[:-2], -1)


def get_Canonical(num_classes):
    return lambda num_features, emsize: CanEmb(num_features, num_classes, emsize)


def get_Embedding(num_embs_per_feature=100):
    return lambda num_features, emsize: EmbeddingEncoder(num_features, emsize, num_embs=num_embs_per_feature)
