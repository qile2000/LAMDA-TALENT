import math
import typing as ty

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import model.lib.node as node
# adapted from https://github.com/yandex-research/rtdl-revisiting-models/blob/main/bin/node.py
# %%
class NODE(nn.Module):
    def __init__(
        self,
        *,
        d_in: int,
        num_layers: int,
        layer_dim: int,
        depth: int,
        tree_dim: int,
        choice_function: str,
        bin_function: str,
        d_out: int,
        ) -> None:
        super().__init__()


        self.d_out = d_out
        self.block = node.DenseBlock(
            input_dim=d_in,
            num_layers=num_layers,
            layer_dim=layer_dim,
            depth=depth,
            tree_dim=max(tree_dim,self.d_out),
            bin_function=getattr(node, bin_function),
            choice_function=getattr(node, choice_function),
            flatten_output=False,
        )

    def forward(self, x_num: Tensor, x_cat: Tensor) -> Tensor:
        x = x_num

        x = self.block(x)
        x = x[..., : self.d_out].mean(dim=-2)
        x = x.squeeze(-1)
        return x
