import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from torch.nn.parameter import Parameter
# adapted from https://github.com/yandex-research/tabred/blob/main/bin/trompt.py

def _init_rsqrt_uniform_(weight: Tensor, dim: None | int, d: None | int = None) -> None:
    if d is None:
        assert dim is not None
        d = weight.shape[dim]
    else:
        assert dim is None
    d_rsqrt = 1 / math.sqrt(d)
    nn.init.uniform_(weight, -d_rsqrt, d_rsqrt)

class LinearEmbeddings(nn.Module):
    """Linear embeddings for continuous features.

    **Shape**

    - Input: `(*, n_features)`
    - Output: `(*, n_features, d_embedding)`

    **Examples**

    >>> batch_size = 2
    >>> n_cont_features = 3
    >>> x = torch.randn(batch_size, n_cont_features)
    >>> d_embedding = 4
    >>> m = LinearEmbeddings(n_cont_features, d_embedding)
    >>> m(x).shape
    torch.Size([2, 3, 4])
    """

    def __init__(self, n_features: int, d_embedding: int) -> None:
        """
        Args:
            n_features: the number of continous features.
            d_embedding: the embedding size.
        """
        if n_features <= 0:
            raise ValueError(f'n_features must be positive, however: {n_features=}')
        if d_embedding <= 0:
            raise ValueError(f'd_embedding must be positive, however: {d_embedding=}')

        super().__init__()
        self.weight = Parameter(torch.empty(n_features, d_embedding))
        self.bias = Parameter(torch.empty(n_features, d_embedding))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        d_rqsrt = self.weight.shape[1] ** -0.5
        nn.init.uniform_(self.weight, -d_rqsrt, d_rqsrt)
        nn.init.uniform_(self.bias, -d_rqsrt, d_rqsrt)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim < 2:
            raise ValueError(
                f'The input must have at least two dimensions, however: {x.ndim=}'
            )

        x = x[..., None] * self.weight
        x = x + self.bias[None]
        return x


class ImportanceGetter(nn.Module): #Figure 3 part 1
    def __init__(self, P, C, d):
        super().__init__()
        self.colemb = nn.Parameter(torch.empty(C, d))
        self.pemb = nn.Parameter(torch.empty(P, d))
        torch.nn.init.normal_(self.colemb, std=0.01)
        torch.nn.init.normal_(self.pemb, std=0.01)
        self.C = C
        self.P = P
        self.d = d
        self.dense = nn.Linear(2 * self.d, self.d)
        self.laynorm1 = nn.LayerNorm(self.d)
        self.laynorm2 = nn.LayerNorm(self.d)
    def forward(self, O):
        eprompt = self.pemb.unsqueeze(0).repeat(O.shape[0], 1, 1)

        dense_out = self.dense(torch.cat((self.laynorm1(eprompt), O), dim=-1))
        
        dense_out = dense_out + eprompt + O

        ecolumn = self.laynorm2(self.colemb.unsqueeze(0).repeat(O.shape[0], 1, 1))

        return torch.softmax(dense_out @ ecolumn.transpose(1, 2), dim=-1)

class CategoricalEmbeddings1d(nn.Module):
    # Input:  (*, n_cat_features=len(cardinalities))
    # Output: (*, n_cat_features, d_embedding)
    def __init__(self, cardinalities: list[int], d_embedding: int) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            # [nn.Embedding(c, d_embedding) for c in cardinalities]
            # NOTE: `+ 1` is here to support unknown values that are expected to have
            # the value `max-known-category + 1`.
            # This is not a good way to handle unknown values. This is just a quick
            # hack to stop failing on some datasets.
            [nn.Embedding(c + 1, d_embedding) for c in cardinalities]
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.embeddings:
            _init_rsqrt_uniform_(m.weight, -1)  # type: ignore[code]

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim >= 1

        return torch.stack(
            [m(x[..., i]) for i, m in enumerate(self.embeddings)], dim=-2
        )
    
class TromptEmbedding(nn.Module): # Figure 3 part 2
    def __init__(self, n_num_features,cat_cardinalities, d):
        super().__init__()
        self.d = d
        self.m_num = LinearEmbeddings(n_num_features, d) if (n_num_features) else None
        self.m_cat = CategoricalEmbeddings1d(cat_cardinalities, d) if cat_cardinalities else None
        self.relu = nn.ReLU()
        self.laynorm1 = nn.LayerNorm(self.d)
        self.laynorm2 = nn.LayerNorm(self.d)

    def forward(self, x_num, x_cat):
        xnc = x_num

        if not(x_cat is None):
            return torch.cat((self.laynorm1(self.relu(self.m_num(xnc))), self.laynorm2(self.m_cat(x_cat))), dim=1)
        return self.laynorm1(self.relu(self.m_num(xnc)))

class Expander(nn.Module): #Figure 3 part 3
    def __init__(self, P):
        super().__init__()
        self.lin = nn.Linear(1, P)
        self.relu = nn.ReLU()
        self.gn = nn.GroupNorm(2, P)

    def forward(self, x):
        res = (self.relu(self.lin(x.unsqueeze(-1))))

        return x.unsqueeze(1) + self.gn(torch.permute(res, (0, 3, 1, 2)))

class TromptCell(nn.Module):
    def __init__(self, n_num_features,cat_cardinalities, P, d):
        super().__init__()
        if cat_cardinalities is None:
            cat_cardinalities = []
        C = n_num_features +  len(cat_cardinalities)
        self.enc = TromptEmbedding(n_num_features, cat_cardinalities, d)
        self.fe = ImportanceGetter(P, C, d)
        self.ex = Expander(P)

    def forward(self, x_num, x_cat, O):
        x_res = self.ex(self.enc(x_num, x_cat))
        
        M = self.fe(O)

        return (M.unsqueeze(-1) * x_res).sum(dim=2)

class TromptDecoder(nn.Module):
    def __init__(self, d,d_out):
        super().__init__()
        self.l1 = nn.Linear(d, 1)
        self.l2 = nn.Linear(d, d)
        self.relu = nn.ReLU()
        self.laynorm1 = nn.LayerNorm(d)
        self.lf = nn.Linear(d, d_out)

    def forward(self, o):
        pw = torch.softmax(self.l1(o).squeeze(-1), dim=-1)

        xnew = (pw.unsqueeze(-1) * o).sum(dim=-2)
        
        return self.lf(self.laynorm1(self.relu(self.l2(xnew))))