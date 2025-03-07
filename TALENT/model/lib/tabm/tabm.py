import torch
from torch import Tensor
from typing import Any, Literal
import torch.nn as nn
from TALENT.model.lib.tabm.deep import init_random_signs_

@torch.inference_mode()
def _init_scaling_by_sections(
    weight: Tensor,
    distribution: Literal['normal', 'random-signs'],
    init_sections: list[int],
) -> None:
    """Initialize the (typically, first) scaling in a special way.

    For a given efficient emsemble member, all weights within one section
    are initialized with the same value.
    Typically, one section corresponds to one feature.
    """
    assert weight.ndim == 2
    print(weight.shape)
    print(init_sections)
    assert weight.shape[1] == sum(init_sections)

    if distribution == 'normal':
        init_fn_ = nn.init.normal_
    elif distribution == 'random-signs':
        init_fn_ = init_random_signs_
    else:
        raise ValueError(f'Unknown distribution: {distribution}')

    section_bounds = [0, *torch.tensor(init_sections).cumsum(0).tolist()]
    for i in range(len(init_sections)):
        w = torch.empty((len(weight), 1), dtype=weight.dtype, device=weight.device)
        init_fn_(w)
        weight[:, section_bounds[i] : section_bounds[i + 1]] = w