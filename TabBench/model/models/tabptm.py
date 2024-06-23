import math
import typing as ty

import torch
import torch.nn as nn
import torch.nn.functional as F

# %%
class TabPTM(nn.Module):
    def __init__(
        self,
        *,
        distance: str,
        is_regression: bool,
        d_in: int,
        d_out: int,
        d_layers: ty.List[int],
        dropout: float,
        ) -> None:
        super().__init__()
        self.is_regression = is_regression
        if is_regression:
            d_in_repeat = len([distance[i:i+3] for i in range(0, len(distance), 3)]) * 2
            self.is_regression = True   
        else:
            d_in_repeat = len([distance[i:i+3] for i in range(0, len(distance), 3)]) * 2

        self.layers = nn.ModuleList(
            [
                nn.Linear(d_layers[i - 1] if i else d_in * d_in_repeat, x, dtype = torch.float64)
                for i, x in enumerate(d_layers)
            ]
        )
        self.dropout = dropout
        self.head = nn.Linear(d_layers[-1] if d_layers else d_in, 1, dtype = torch.float64)
        self.d_out = d_out

    def forward(self, x):
        if self.is_regression:
            '''batch x 1 x distance_num x (numK * 2)'''
            num_batch, num_class = x.shape[:2]
            x = x.view(num_batch * num_class, -1)
            for layer in self.layers:
                x = layer(x)
                x = F.relu(x)
                if self.dropout:
                    x = F.dropout(x, self.dropout, self.training)
            x = self.head(x)        
            x = x.view(num_batch, self.d_out)
            x = x.view(-1)
            return x              
        else:
            '''batch x num_class x distance_num x numK'''
            num_batch, num_class = x.shape[:2]
            x = x.view(num_batch * num_class, -1)
            for layer in self.layers:
                x = layer(x)
                x = F.relu(x)
                if self.dropout:
                    x = F.dropout(x, self.dropout, self.training)
            x = self.head(x)        
            x = x.view(num_batch, num_class)
            return x