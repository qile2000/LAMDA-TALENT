import math
import typing as ty
from TALENT.model.lib.tabr.utils import make_module
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union
# %%
class MLP(nn.Module):
    def __init__(
        self,
        *,
        d_in: int,
        d_num:int,
        d_out: int, 
        d_layers: ty.List[int],    
        dropout: float,
        num_embeddings: Optional[dict],
        ) -> None:
        super().__init__()
        self.dropout = dropout
        self.d_out = d_out
        self.d_num=d_num
        self.d_in = d_in if num_embeddings is None else d_num*num_embeddings['d_embedding']+d_in-d_num   
        self.layers = nn.ModuleList(
            [
                nn.Linear(d_layers[i - 1] if i else self.d_in, x)
                for i, x in enumerate(d_layers)
            ]
        )
        self.head = nn.Linear(d_layers[-1] if d_layers else d_in, d_out)
        self.num_embeddings = (
            None
            if num_embeddings is None
            else make_module(num_embeddings, n_features=d_num)
        )       

    def forward(self, x_num, x_cat):
        if self.num_embeddings is not None and self.d_num >0:
            x_num=self.num_embeddings(x_num).flatten(1)
        if x_num is not None and x_cat is not None:
            x=torch.cat([x_num,x_cat],dim=-1)
        elif x_num is not None:
            x=x_num
        elif x_cat is not None:
            x=x_cat
        x = x.to(self.head.weight.dtype)
        for layer in self.layers:
            # print(x.shape,self.d_in)
            x = layer(x)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, self.dropout, self.training)
        logit = self.head(x)        
        if self.d_out == 1:
            logit = logit.squeeze(-1)
        return  logit