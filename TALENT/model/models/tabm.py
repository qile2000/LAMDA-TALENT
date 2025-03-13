import math
import typing as ty

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Literal
from torch import Tensor
import delu
from TALENT.model.lib.tabm.tabm import _init_scaling_by_sections
from TALENT.model.lib.tabm.deep import  ElementwiseAffineEnsemble,make_efficient_ensemble,OneHotEncoding0d
from TALENT.model.lib.tabr.utils import make_module,make_module1,MLP,ResNet

def _get_first_input_scaling(
    backbone):
    if isinstance(backbone, MLP):
        return backbone.blocks[0][0]  # type: ignore[code]
    elif isinstance(backbone, ResNet):
        return backbone.blocks[0][1] if backbone.proj is None else backbone.proj  # type: ignore[code]
    else:
        raise RuntimeError(f'Unsupported backbone: {backbone}')
    
class TabM(nn.Module):
    def __init__(
        self,
        *,
        n_num_features: int,
        cat_cardinalities: list[int],
        n_classes: None | int,
        # bins: None | list[Tensor],
        backbone: dict,
        num_embeddings: None | dict = None,
        arch_type: Literal[
            # Active
            'vanilla', # Simple MLP
            'tabm',  # BatchEnsemble + separate heads + better initialization
            'tabm-mini',  # Minimal: * weight
            # BatchEnsemble
            'tabm-naive'
        ],
        k: None | int = None,
    ) -> None:
        # >>> Validate arguments.
        assert n_num_features >= 0
        assert n_num_features or cat_cardinalities
        if arch_type == 'vanilla':
            assert k is None
        else:
            assert k is not None
            assert k > 0
        if cat_cardinalities is None:
            cat_cardinalities = []
        super().__init__()

        # >>> Continuous (numerical) features
        scaling_init_sections = []
        print(num_embeddings)
        if n_num_features == 0:
            # assert bins is None
            self.num_module = None
            d_num = 0

        elif num_embeddings is None:
            # assert bins is None
            self.num_module = None
            d_num = n_num_features
            scaling_init_sections.extend(1 for _ in range(n_num_features))

        else:
            self.num_module = make_module(
                num_embeddings, n_features=n_num_features
            )
            d_num = n_num_features * num_embeddings['d_embedding']
            scaling_init_sections.extend(
                num_embeddings['d_embedding'] for _ in range(n_num_features)
            )

        # >>> Categorical features
        self.cat_module = (
            OneHotEncoding0d(cat_cardinalities) if cat_cardinalities else None
        )
        scaling_init_sections.extend(cat_cardinalities)
        d_cat = sum(cat_cardinalities)

        # >>> Backbone
        d_flat = d_num + d_cat
        self.affine_ensemble = None
        self.backbone = make_module1(d_in=d_flat,**backbone)

        if arch_type != 'vanilla':
            assert k is not None
            scaling_init = (
                'random-signs'
                if num_embeddings is None
                else 'normal'
            )

            if arch_type == 'tabm-mini':
                # The minimal possible efficient ensemble.
                self.affine_ensemble = ElementwiseAffineEnsemble(
                    k,
                    d_flat,
                    weight=True,
                    bias=False,
                    weight_init=(
                        'random-signs'
                        if num_embeddings is None
                        else 'normal'
                    ),
                )
                _init_scaling_by_sections(
                    self.affine_ensemble.weight,  # type: ignore[code]
                    scaling_init,
                    scaling_init_sections,
                )

            elif arch_type == 'tabm-naive':
                # The original BatchEnsemble.
                make_efficient_ensemble(
                    self.backbone,
                    k=k,
                    ensemble_scaling_in=True,
                    ensemble_scaling_out=True,
                    ensemble_bias=True,
                    scaling_init='random-signs',
                )
            elif arch_type == 'tabm':
                # Like BatchEnsemble, but all scalings, except for the first one,
                # are initialized with ones.
                make_efficient_ensemble(
                    self.backbone,
                    k=k,
                    ensemble_scaling_in=True,
                    ensemble_scaling_out=True,
                    ensemble_bias=True,
                    scaling_init='ones',
                )
                _init_scaling_by_sections(
                    _get_first_input_scaling(self.backbone).r,  # type: ignore[code]
                    scaling_init,
                    scaling_init_sections,
                )

            else:
                raise ValueError(f'Unknown arch_type: {arch_type}')

        # >>> Output
        d_block = backbone['d_block']
        d_out = 1 if n_classes is None else n_classes
        self.output = (
            nn.Linear(d_block, d_out)
            if arch_type == 'vanilla'
            else delu.nn.NLinear(k, d_block, d_out)  # type: ignore[code]
        )
        self.d_out = d_out
        # >>>
        self.arch_type = arch_type
        self.k = k

    def forward(
        self, x_num: None | Tensor = None, x_cat: None | Tensor = None
    ) -> Tensor:
        x = []
        if x_num is not None:
            x.append(x_num if self.num_module is None else self.num_module(x_num))
        if x_cat is None:
            assert self.cat_module is None
        else:
            assert self.cat_module is not None
            x.append(self.cat_module(x_cat))
        x = torch.column_stack([x_.flatten(1, -1) for x_ in x])
        if x.dtype == torch.int64:
            x = x.float()

        if self.k is not None:
            x = x[:, None].expand(-1, self.k, -1)  # (B, D) -> (B, K, D)
            if self.affine_ensemble is not None:
                x = self.affine_ensemble(x)
        else:
            assert self.affine_ensemble is None

        x = self.backbone(x)
        x = self.output(x)
        # print(x.shape)
        if self.k is None:
            # Adjust the output shape for vanilla networks to make them compatible
            # with the rest of the script (loss, metrics, predictions, ...).
            # (B, D_OUT) -> (B, 1, D_OUT)
            x = x[:, None]
        if self.d_out == 1:
            x = x.squeeze(-1)
        return x