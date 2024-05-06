import math
import typing as ty

import torch
import torch.nn as nn
import torch.nn.functional as F

# %%
class MLP(nn.Module):
    def __init__(
        self,
        *,
        d_in: int,
        d_out: int, 
        d_layers: ty.List[int],    
        dropout: float,
        
        ) -> None:
        super().__init__()
        self.dropout = dropout
        self.d_out = d_out 
        self.layers = nn.ModuleList(
            [
                nn.Linear(d_layers[i - 1] if i else d_in, x)
                for i, x in enumerate(d_layers)
            ]
        )
        self.head = nn.Linear(d_layers[-1] if d_layers else d_in, d_out)


    def forward(self, x, x_cat = None):

        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, self.dropout, self.training)
        logit = self.head(x)        
        if self.d_out == 1:
            logit = logit.squeeze(-1)
        return  logit