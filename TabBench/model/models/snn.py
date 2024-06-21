import typing as ty

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
# Source: https://github.com/yandex-research/rtdl-revisiting-models/blob/main/bin/snn.py
# %%
class SNN(nn.Module):
    def __init__(
        self,
        *,
        d_in: int,
        d_layers: ty.List[int],
        dropout: float,
        d_out: int,
        categories: ty.Optional[ty.List[int]],
        d_embedding: int,
    ) -> None:
        super().__init__()
        assert d_layers

        if categories is not None:
            d_in += len(categories) * d_embedding
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_embedding)
            nn.init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
            print(f'{self.category_embeddings.weight.shape=}')

        self.layers = (
            nn.ModuleList(
                [
                    nn.Linear(d_layers[i - 1] if i else d_in, x)
                    for i, x in enumerate(d_layers)
                ]
            )
            if d_layers
            else None
        )

        self.normalizations = None
        self.activation = nn.SELU()
        self.dropout = dropout
        self.head = nn.Linear(d_layers[-1] if d_layers else d_in, d_out)

        # Ensure correct initialization
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight.data, mode='fan_in', nonlinearity='linear'
                )
                nn.init.zeros_(m.bias.data)

        self.apply(init_weights)

    @property
    def d_embedding(self) -> int:
        return self.head.id_in  # type: ignore[code]

    def encode(self, x_num, x_cat):
        x = []
        if x_num is not None:
            x.append(x_num)
        if x_cat is not None:
            x.append(
                self.category_embeddings(x_cat + self.category_offsets[None]).view(
                    x_cat.size(0), -1
                )
            )
        x = torch.cat(x, dim=-1)

        layers = self.layers or []
        for i, m in enumerate(layers):
            x = m(x)
            if self.normalizations:
                x = self.normalizations[i](x)
            x = self.activation(x)
            if self.dropout:
                x = F.alpha_dropout(x, self.dropout, self.training)
        return x

    def calculate_output(self, x: Tensor) -> Tensor:
        x = self.head(x)
        x = x.squeeze(-1)
        return x

    def forward(self, x_num: Tensor, x_cat) -> Tensor:
        return self.calculate_output(self.encode(x_num, x_cat))