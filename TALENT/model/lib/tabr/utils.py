import torch
import torch.nn as nn
from typing import  Optional, Union
import math
import statistics
from functools import partial
from typing import Any, Callable, Optional, Union, cast
from torch import Tensor
from torch.nn.parameter import Parameter
import math

# adapted from https://github.com/yandex-research/tabular-dl-tabr
def _initialize_embeddings(weight: Tensor, d: Optional[int]) -> None:
    if d is None:
        d = weight.shape[-1]
    d_sqrt_inv = 1 / math.sqrt(d)
    nn.init.uniform_(weight, a=-d_sqrt_inv, b=d_sqrt_inv)


def make_trainable_vector(d: int) -> Parameter:
    x = torch.empty(d)
    _initialize_embeddings(x, None)
    return Parameter(x)





class CLSEmbedding(nn.Module):
    def __init__(self, d_embedding: int) -> None:
        super().__init__()
        self.weight = make_trainable_vector(d_embedding)

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 3
        assert x.shape[-1] == len(self.weight)
        return torch.cat([self.weight.expand(len(x), 1, -1), x], dim=1)


class ResNet(nn.Module):
    def __init__(
        self,
        *,
        d_in: None | int = None,
        d_out: None | int = None,
        n_blocks: int,
        d_block: int,
        dropout: float,
        d_hidden_multiplier: float | int,
        n_linear_layers_per_block: int = 2,
        activation: str = 'ReLU',
        normalization: str,
        first_normalization: bool,
    ) -> None:
        assert n_linear_layers_per_block in (1, 2)
        if n_linear_layers_per_block == 1:
            assert d_hidden_multiplier == 1
        super().__init__()

        Activation = getattr(nn, activation)
        Normalization = (
            Identity if normalization == 'none' else getattr(nn, normalization)
        )
        d_hidden = int(d_block * d_hidden_multiplier)

        self.proj = None if d_in is None else nn.Linear(d_in, d_block)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    Normalization(d_block) if first_normalization else Identity(),
                    (
                        nn.Linear(d_block, d_hidden)
                        if n_linear_layers_per_block == 2
                        else nn.Linear(d_block, d_block)
                    ),
                    Activation(),
                    nn.Dropout(dropout),
                    (
                        nn.Linear(d_hidden, d_block)
                        if n_linear_layers_per_block == 2
                        else Identity()
                    ),
                )
                for _ in range(n_blocks)
            ]
        )
        self.preoutput = nn.Sequential(Normalization(d_block), Activation())
        self.output = None if d_out is None else nn.Linear(d_block, d_out)

    def forward(self, x: Tensor) -> Tensor:
        if self.proj is not None:
            x = self.proj(x)
        for block in self.blocks:
            x = x + block(x)
        x = self.preoutput(x)
        if self.output is not None:
            x = x + self.output(x)
        return x


class LinearEmbeddings(nn.Module):
    def __init__(self, n_features: int, d_embedding: int, bias: bool = True):
        super().__init__()
        self.weight = Parameter(Tensor(n_features, d_embedding))
        self.bias = Parameter(Tensor(n_features, d_embedding)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for parameter in [self.weight, self.bias]:
            if parameter is not None:
                _initialize_embeddings(parameter, parameter.shape[-1])

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 2
        x = self.weight[None] * x[..., None]
        if self.bias is not None:
            x = x + self.bias[None]
        return x


class PeriodicEmbeddings(nn.Module):
    def __init__(
        self, n_features: int, n_frequencies: int, frequency_scale: float
    ) -> None:
        super().__init__()
        self.frequencies = Parameter(
            torch.normal(0.0, frequency_scale, (n_features, n_frequencies))
        )

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 2
        x = 2 * torch.pi * self.frequencies[None] * x[..., None]
        x = torch.cat([torch.cos(x), torch.sin(x)], -1)
        return x


class NLinear(nn.Module):
    def __init__(
        self, n_features: int, d_in: int, d_out: int, bias: bool = True
    ) -> None:
        super().__init__()
        self.weight = Parameter(Tensor(n_features, d_in, d_out))
        self.bias = Parameter(Tensor(n_features, d_out)) if bias else None
        with torch.no_grad():
            for i in range(n_features):
                layer = nn.Linear(d_in, d_out)
                self.weight[i] = layer.weight.T
                if self.bias is not None:
                    self.bias[i] = layer.bias

    def forward(self, x):
        assert x.ndim == 3
        x = x[..., None] * self.weight[None]
        x = x.sum(-2)
        if self.bias is not None:
            x = x + self.bias[None]
        return x


class LREmbeddings(nn.Sequential):
    """The LR embeddings from the paper 'On Embeddings for Numerical Features in Tabular Deep Learning'."""  # noqa: E501

    def __init__(self, n_features: int, d_embedding: int) -> None:
        super().__init__(LinearEmbeddings(n_features, d_embedding), nn.ReLU())


class PLREmbeddings(nn.Sequential):
    """The PLR embeddings from the paper 'On Embeddings for Numerical Features in Tabular Deep Learning'.

    Additionally, the 'lite' option is added. Setting it to `False` gives you the original PLR
    embedding from the above paper. We noticed that `lite=True` makes the embeddings
    noticeably more lightweight without critical performance loss, and we used that for our model.
    """  # noqa: E501

    def __init__(
        self,
        n_features: int,
        n_frequencies: int,
        frequency_scale: float,
        d_embedding: int,
        lite: bool,
    ) -> None:
        super().__init__(
            PeriodicEmbeddings(n_features, n_frequencies, frequency_scale),
            (
                nn.Linear(2 * n_frequencies, d_embedding)
                if lite
                else NLinear(n_features, 2 * n_frequencies, d_embedding)
            ),
            nn.ReLU(),
        )
import numpy as np
class PBLDEmbeddings(nn.Module):
    def __init__(self, n_features: int,
                 n_frequencies: int,
                 frequency_scale: float,
                 d_embedding: int,
                 lite: bool,
                 plr_act_name: str = 'relu',
                 plr_use_densenet: bool = True):
        super().__init__()
        print(f'Constructing PBLD embeddings')
        hidden_2 = d_embedding-1 if plr_use_densenet else d_embedding
        self.weight_1 = nn.Parameter(frequency_scale * torch.randn(n_features, 1, n_frequencies))
        self.weight_2 = nn.Parameter((-1 + 2 * torch.rand(n_features, n_frequencies, hidden_2))
                / np.sqrt(n_frequencies))
        self.bias_1 = nn.Parameter(np.pi * (-1 + 2 * torch.rand(n_features, 1, n_frequencies)))
        self.bias_2 = nn.Parameter((-1 + 2 * torch.rand(n_features, 1, hidden_2)) / np.sqrt(n_frequencies))
        self.plr_act_name = plr_act_name
        self.plr_use_densenet = plr_use_densenet

    def forward(self, x):
        # transpose to treat the continuous feature dimension like a batched dimension
        # then add a new channel dimension
        # shape will be (vectorized..., n_cont, batch, 1)
        x_orig = x
        x = x.transpose(-1, -2).unsqueeze(-1)
        x = 2 * torch.pi * x.matmul(self.weight_1)  # matmul is automatically batched
        x = x + self.bias_1
        # x = torch.sin(x)
        x = torch.cos(x)
        x = x.matmul(self.weight_2)  # matmul is automatically batched
        x = x + self.bias_2
        if self.plr_act_name == 'relu':
            x = torch.relu(x)
        elif self.plr_act_name == 'linear':
            pass
        else:
            raise ValueError(f'Unknown plr_act_name "{self.plr_act_name}"')
        # bring back n_cont dimension after n_batch
        # then flatten the last two dimensions
        x = x.transpose(-2, -3)
        x = x.reshape(*x.shape[:-2], x.shape[-2] * x.shape[-1])
        if self.plr_use_densenet:
            x = torch.cat([x, x_orig], dim=-1)
        return x




# class MLP(nn.Module):
#     class Block(nn.Module):
#         def __init__(
#             self,
#             *,
#             d_in: int,
#             d_out: int,
#             bias: bool,
#             activation: str,
#             dropout: float,
#         ) -> None:
#             super().__init__()
#             self.linear = nn.Linear(d_in, d_out, bias)
#             self.activation = make_module(activation)
#             self.dropout = nn.Dropout(dropout)

#         def forward(self, x: Tensor) -> Tensor:
#             return self.dropout(self.activation(self.linear(x)))

#     Head = nn.Linear

#     def __init__(
#         self,
#         *,
#         d_in: int,
#         d_out: Optional[int],
#         n_blocks: int,
#         d_layer: int,
#         activation: str,
#         dropout: float,
#     ) -> None:
#         assert n_blocks > 0
#         super().__init__()

#         self.blocks = nn.Sequential(
#             *[
#                 MLP.Block(
#                     d_in=d_layer if block_i else d_in,
#                     d_out=d_layer,
#                     bias=True,
#                     activation=activation,
#                     dropout=dropout,
#                 )
#                 for block_i in range(n_blocks)
#             ]
#         )
#         self.head = None if d_out is None else MLP.Head(d_layer, d_out)

#     @property
#     def d_out(self) -> int:
#         return (
#             self.blocks[-1].linear.out_features  # type: ignore[code]
#             if self.head is None
#             else self.head.out_features
#         )

#     def forward(self, x: Tensor) -> Tensor:
#         x = self.blocks(x)
#         if self.head is not None:
#             x = self.head(x)
#         return x

class MLP(nn.Module):
    def __init__(
        self,
        *,
        d_in: None | int = None,
        d_out: None | int = None,
        n_blocks: int,
        d_block: int,
        dropout: float,
        activation: str = 'SELU',
    ) -> None:
        super().__init__()

        d_first = d_block if d_in is None else d_in
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_first if i == 0 else d_block, d_block),
                    getattr(nn, activation)(),
                    nn.Dropout(dropout),
                )
                for i in range(n_blocks)
            ]
        )
        self.output = None if d_out is None else nn.Linear(d_block, d_out)

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        if self.output is not None:
            x = self.output(x)
        return x


_CUSTOM_MODULES = {
    x.__name__: x
    for x in [
        LinearEmbeddings,
        LREmbeddings,
        PLREmbeddings,
        MLP,
        PBLDEmbeddings,
    ]
}

def make_module(spec, *args, **kwargs) -> nn.Module:
    """
    >>> make_module('ReLU')
    >>> make_module(nn.ReLU)
    >>> make_module('Linear', 1, out_features=2)
    >>> make_module((lambda *args: nn.Linear(*args)), 1, out_features=2)
    >>> make_module({'type': 'Linear', 'in_features' 1}, out_features=2)
    """
    if isinstance(spec, str):
        Module = getattr(nn, spec, None)
        if Module is None:
            Module = _CUSTOM_MODULES[spec]
        else:
            assert spec not in _CUSTOM_MODULES
        return make_module(Module, *args, **kwargs)
    elif isinstance(spec, dict):
        assert not (set(spec) & set(kwargs))
        spec = spec.copy()
        return make_module(spec.pop('type'), *args, **spec, **kwargs)
    elif callable(spec):
        return spec(*args, **kwargs)
    else:
        raise ValueError()
    
def make_module1(type: str, *args, **kwargs) -> nn.Module:
    Module = getattr(nn, type, None)
    if Module is None:
        Module = _CUSTOM_MODULES[type]
    return Module(*args, **kwargs)