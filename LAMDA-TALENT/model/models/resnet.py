import math
import typing as ty

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
# adapted from https://github.com/yandex-research/rtdl-revisiting-models/blob/main/bin/resnet.py
# %%
def reglu(x):
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)


def geglu(x):
    a, b = x.chunk(2, dim=-1)
    return a * F.gelu(b)

def get_nonglu_activation_fn(name):
    return (
        F.relu
        if name == 'reglu'
        else F.gelu
        if name == 'geglu'
        else get_activation_fn(name)
    )

def get_activation_fn(name):
    return (
        reglu
        if name == 'reglu'
        else geglu
        if name == 'geglu'
        else torch.sigmoid
        if name == 'sigmoid'
        else getattr(F, name)
    )

class ResNet(nn.Module):
    def __init__(
        self,
        *,
        d_in: int,
        d: int,
        d_hidden_factor: float,
        n_layers: int,
        activation: str,
        normalization: str,
        hidden_dropout: float,
        residual_dropout: float,
        d_out: int,
        ) -> None:
        super().__init__()

        def make_normalization():
            return {'batchnorm': nn.BatchNorm1d, 'layernorm': nn.LayerNorm}[
                normalization
                ](d)

        self.main_activation = get_activation_fn(activation)
        self.last_activation = get_nonglu_activation_fn(activation)
        self.residual_dropout = residual_dropout
        self.hidden_dropout = hidden_dropout

        d_hidden = int(d * d_hidden_factor)

        self.first_layer = nn.Linear(d_in, d)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        'norm': make_normalization(),
                        'linear0': nn.Linear(
                            d, d_hidden * (2 if activation.endswith('glu') else 1)
                            ),
                        'linear1': nn.Linear(d_hidden, d),
                    }
                )
                for _ in range(n_layers)
            ]
        )
        self.last_normalization = make_normalization()
        self.head = nn.Linear(d, d_out)

    def forward(self, x: Tensor, x_cat: Tensor) -> Tensor:

        x = self.first_layer(x)
        for layer in self.layers:
            layer = ty.cast(ty.Dict[str, nn.Module], layer)
            z = x
            z = layer['norm'](z)
            z = layer['linear0'](z)
            z = self.main_activation(z)
            if self.hidden_dropout:
                z = F.dropout(z, self.hidden_dropout, self.training)
            z = layer['linear1'](z)
            if self.residual_dropout:
                z = F.dropout(z, self.residual_dropout, self.training)
            x = x + z
        x = self.last_normalization(x)
        x = self.last_activation(x)
        x = self.head(x)
        x = x.squeeze(-1)
        return x