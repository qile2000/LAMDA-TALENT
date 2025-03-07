import math
import typing as ty

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from TALENT.model.lib.trompt.trompt import TromptCell, TromptDecoder


class Trompt(nn.Module): #Trompt
    def __init__(self, n_num_features, cat_cardinalities,d_out, P, d, n_cycles):
        super().__init__()
        # self.tcell = TromptCell(n_num_features,cat_cardinalities, P, d)
        self.tcell = nn.ModuleList(
            [
                TromptCell(n_num_features,cat_cardinalities, P, d) for i in range(n_cycles)
            ]
        )
        self.tdown = TromptDecoder(d,d_out)
        self.init_rec = nn.Parameter(torch.empty(P, d))
        nn.init.normal_(self.init_rec, std=0.01)
        self.n_cycles = n_cycles

    def forward_for_training(self, x_num : None | Tensor = None,  x_cat : None | Tensor = None):
        O = self.init_rec.unsqueeze(0).repeat(x_num.shape[0], 1, 1)
        outputs = []
        for i in range(self.n_cycles):
            O = self.tcell[i](x_num, x_cat, O)
            # print(O.shape)
            # print(self.tdown(O).shape)
            outputs.append(self.tdown(O))

        return torch.stack(outputs, dim=1)

    def forward(self, x_num : None | Tensor = None,  x_cat : None | Tensor = None):
        Output = self.forward_for_training(x_num, x_cat)

        return Output.mean(dim=1)
    
    