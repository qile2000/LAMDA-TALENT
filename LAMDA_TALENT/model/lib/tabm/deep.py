import itertools
import math
from collections.abc import Callable
from typing import Any, Literal, cast

import delu
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
_TORCH = None


def _torch():
    global _TORCH
    if _TORCH is None:
        import torch

        _TORCH = torch
    return _TORCH

def is_oom_exception(err: RuntimeError) -> bool:
    return isinstance(err, _torch().cuda.OutOfMemoryError) or any(
        x in str(err)
        for x in [
            'CUDA out of memory',
            'CUBLAS_STATUS_ALLOC_FAILED',
            'CUDA error: out of memory',
        ]
    )


# ======================================================================================
# Initialization
# ======================================================================================
def init_rsqrt_uniform_(x: Tensor, d: int) -> Tensor:
    assert d > 0
    d_rsqrt = d**-0.5
    return nn.init.uniform_(x, -d_rsqrt, d_rsqrt)


@torch.inference_mode()
def init_random_signs_(x: Tensor) -> Tensor:
    return x.bernoulli_(0.5).mul_(2).add_(-1)


# ======================================================================================
# Modules
# ======================================================================================
class Identity(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x


class Mean(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return x.mean(dim=self.dim)

class ScaleEnsemble(nn.Module):
    def __init__(
        self,
        k: int,
        d: int,
        *,
        init: Literal['ones', 'normal', 'random-signs'],
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(k, d))
        self._weight_init = init
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self._weight_init == 'ones':
            nn.init.ones_(self.weight)
        elif self._weight_init == 'normal':
            nn.init.normal_(self.weight)
        elif self._weight_init == 'random-signs':
            init_random_signs_(self.weight)
        else:
            raise ValueError(f'Unknown weight_init: {self._weight_init}')

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim >= 2
        return x * self.weight


class ElementwiseAffineEnsemble(nn.Module):
    def __init__(
        self,
        k: int,
        d: int,
        *,
        weight: bool,
        bias: bool,
        weight_init: Literal['ones', 'normal', 'random-signs'],
    ) -> None:
        assert weight or bias
        super().__init__()
        self.weight = nn.Parameter(torch.empty(k, d)) if weight else None
        self.bias = nn.Parameter(torch.empty(k, d)) if bias else None
        self._weight_init = weight_init
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.weight is not None:
            if self._weight_init == 'ones':
                nn.init.ones_(self.weight)
            elif self._weight_init == 'normal':
                nn.init.normal_(self.weight)
            elif self._weight_init == 'random-signs':
                init_random_signs_(self.weight)
            else:
                raise ValueError(f'Unknown weight_init: {self._weight_init}')
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        # x.shape == (B, K, D)
        assert x.ndim == 3
        return (
            x * self.weight
            if self.bias is None
            else x + self.bias
            if self.weight is None
            else torch.addcmul(self.bias, self.weight, x)
        )


class LinearEfficientEnsemble(nn.Module):
    """
    This layer is a more configurable version of the "BatchEnsemble" layer
    from the paper
    "BatchEnsemble: An Alternative Approach to Efficient Ensemble and Lifelong Learning"
    (link: https://arxiv.org/abs/2002.06715).

    First, this layer allows to select only some of the "ensembled" parts:
    - the input scaling  (r_i in the BatchEnsemble paper)
    - the output scaling (s_i in the BatchEnsemble paper)
    - the output bias    (not mentioned in the BatchEnsemble paper,
                          but is presented in public implementations)

    Second, the initialization of the scaling weights is configurable
    through the `scaling_init` argument.
    """

    r: None | Tensor
    s: None | Tensor
    bias: None | Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        k: int,
        ensemble_scaling_in: bool,
        ensemble_scaling_out: bool,
        ensemble_bias: bool,
        scaling_init: Literal['ones', 'random-signs'],
    ):
        assert k > 0
        if ensemble_bias:
            assert bias
        super().__init__()

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.register_parameter(
            'r',
            (
                nn.Parameter(torch.empty(k, in_features))
                if ensemble_scaling_in
                else None
            ),  # type: ignore[code]
        )
        self.register_parameter(
            's',
            (
                nn.Parameter(torch.empty(k, out_features))
                if ensemble_scaling_out
                else None
            ),  # type: ignore[code]
        )
        self.register_parameter(
            'bias',
            (
                nn.Parameter(torch.empty(out_features))  # type: ignore[code]
                if bias and not ensemble_bias
                else nn.Parameter(torch.empty(k, out_features))
                if ensemble_bias
                else None
            ),
        )

        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        self.scaling_init = scaling_init

        self.reset_parameters()

    def reset_parameters(self):
        init_rsqrt_uniform_(self.weight, self.in_features)
        scaling_init_fn = {'ones': nn.init.ones_, 'random-signs': init_random_signs_}[
            self.scaling_init
        ]
        if self.r is not None:
            scaling_init_fn(self.r)
        if self.s is not None:
            scaling_init_fn(self.s)
        if self.bias is not None:
            bias_init = torch.empty(
                # NOTE: the shape of bias_init is (out_features,) not (k, out_features).
                # It means that all biases have the same initialization.
                # This is similar to having one shared bias plus
                # k zero-initialized non-shared biases.
                self.out_features,
                dtype=self.weight.dtype,
                device=self.weight.device,
            )
            bias_init = init_rsqrt_uniform_(bias_init, self.in_features)
            with torch.inference_mode():
                self.bias.copy_(bias_init)

    def forward(self, x: Tensor) -> Tensor:
        # x.shape == (B, K, D)
        assert x.ndim == 3

        # >>> The equation (5) from the BatchEnsemble paper (arXiv v2).
        if self.r is not None:
            x = x * self.r
        x = x @ self.weight.T
        if self.s is not None:
            x = x * self.s
        # <<<

        if self.bias is not None:
            x = x + self.bias
        return x


def make_efficient_ensemble(module: nn.Module, **kwargs) -> None:
    for name, submodule in list(module.named_children()):
        if isinstance(submodule, nn.Linear):
            module.add_module(
                name,
                LinearEfficientEnsemble(
                    in_features=submodule.in_features,
                    out_features=submodule.out_features,
                    bias=submodule.bias is not None,
                    **kwargs,
                ),
            )
        else:
            make_efficient_ensemble(submodule, **kwargs)

class OneHotEncoding0d(nn.Module):
    # Input:  (*, n_cat_features=len(cardinalities))
    # Output: (*, sum(cardinalities))

    def __init__(self, cardinalities: list[int]) -> None:
        super().__init__()
        self._cardinalities = cardinalities

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim >= 1
        assert x.shape[-1] == len(self._cardinalities)

        return torch.cat(
            [
                # NOTE
                # This is a quick hack to support out-of-vocabulary categories.
                #
                # Recall that lib.data.transform_cat encodes categorical features
                # as follows:
                # - In-vocabulary values receive indices from `range(cardinality)`.
                # - All out-of-vocabulary values (i.e. new categories in validation
                #   and test data that are not presented in the training data)
                #   receive the index `cardinality`.
                #
                # As such, the line below will produce the standard one-hot encoding for
                # known categories, and the all-zeros encoding for unknown categories.
                # This may not be the best approach to deal with unknown values,
                # but should be enough for our purposes.
                F.one_hot(x[..., i], cardinality + 1)[..., :-1]
                for i, cardinality in enumerate(self._cardinalities)
            ],
            -1,
        )









'''
class PiecewiseLinearEmbeddingsV2(nn.Module):
    """NOTE: use PiecewiseLinearEmbeddingsV3 instead."""

    def __init__(self, bins: list[Tensor], d_embedding: int) -> None:
        super().__init__()
        self.linear = rtdl_revisiting_models.LinearEmbeddings(len(bins), d_embedding)
        self.ple = rtdl_num_embeddings.PiecewiseLinearEmbeddings(
            bins, d_embedding, activation=False
        )
        nn.init.zeros_(self.ple.linear.weight)
        nn.init.zeros_(self.ple.linear.bias)
        self.ple.linear.bias.requires_grad_(False)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x) + self.ple(x)


class PiecewiseLinearEmbeddingsV3(nn.Module):
    """Piecewise-linear embeddings.

    **Shape**

    - Input: ``(*, n_features)``
    - Output: ``(*, n_features, d_embedding)``

    This class is a reimplementation of the piecewise-linear embeddings from the paper
    "On Embeddings for Numerical Features in Tabular Deep Learning".
    The original implementation:
    https://github.com/yandex-research/rtdl-num-embeddings/blob/24173f7023088d48a1a81765029aedd632316d56/package/rtdl_num_embeddings.py#L550

    The differences are as follows:

    * New parametrization and initialization,
      which leads to better performance on some datasets.
    * Significantly better efficiency, which comes from two things.
      First, from the observation that the piecewise-linear encoding is just
      a linear transformation followed by a clamp-based activation.
      Second, from aligning the last encoding channel across all features,
      which allows applying the activation simultaneously to all features.

    Because of the floating-point arithmetic,
    the new class can produces different output for the same input.
    Because of that, in some cases, that requires retuning hyperparameters of a model.
    """

    weight2: Tensor
    bias2: Tensor
    single_bin_mask: None | Tensor

    def __init__(self, bins: list[Tensor], d_embedding: int) -> None:
        assert len(bins) > 0
        assert d_embedding > 0
        super().__init__()

        n_features = len(bins)
        n_bins = [len(x) - 1 for x in bins]
        max_n_bins = max(n_bins)
        single_bin_mask = torch.tensor(n_bins) == 1
        self.register_buffer(
            'single_bin_mask', single_bin_mask if single_bin_mask.any() else None
        )

        self.weight1 = nn.Parameter(torch.empty(n_features, d_embedding))
        self.bias1 = nn.Parameter(torch.empty(n_features, d_embedding))
        # Making weight2 and bias2 trainable can lead to *worse* performance.
        self.register_buffer('weight2', torch.zeros(n_features, max_n_bins))
        self.register_buffer('bias2', torch.zeros(n_features, max_n_bins))
        self.weight3 = nn.Parameter(torch.empty(n_features, max_n_bins, d_embedding))

        self.reset_parameters()
        for i, bin_edges in enumerate(bins):
            # >>>
            # Recall that the piecewise-linear encoding of a single feature x
            # looks as follows (see Figure 1 in the paper
            # "On Embeddings for Numerical Features in Tabular Deep Learning"):
            #
            # x --> [1, ..., 1, (x - left_bin_edge) / bin_width, 0, ..., 0]
            #
            # where the expression in the middle corresponds to the bin where
            # the feature belongs: left_bin_edge <= x < right_bin_edge
            # <<<
            bin_width = bin_edges.diff()
            w = 1.0 / bin_width
            b = -bin_edges[:-1] / bin_width
            # The last encoding channel is located in the same last column
            # for all features.
            self.weight2[i, : len(bin_edges) - 2] = w[:-1]
            self.weight2[i, -1] = w[-1]
            self.bias2[i, : len(bin_edges) - 2] = b[:-1]
            self.bias2[i, -1] = b[-1]

    def reset_parameters(self) -> None:
        d_rsqrt = self.weight1.shape[-1] ** -0.5
        nn.init.uniform_(self.weight1, -d_rsqrt, d_rsqrt)
        nn.init.uniform_(self.bias1, -d_rsqrt, d_rsqrt)
        # The piecewise-linear term is initialized with zeros.
        # Thus, at initialization, the whole embedding is actually linear,
        # and the piecewise-linear term is gradually learned during training.
        nn.init.zeros_(self.weight3)

    def forward(self, x: Tensor) -> Tensor:
        # >>> The linear term.
        x_linear = torch.addcmul(self.bias1, self.weight1, x[..., None])

        # >>> The piecewise-linear term.
        # Step 1: express feature values through their bins.
        # x --> [1, ..., 1, (x - left_bin_edge) / bin_width, 0, ..., 0]

        x_ple = torch.addcmul(self.bias2, self.weight2, x[..., None])
        if x_ple.shape[-1] > 1:
            x_ple = torch.cat(
                [
                    x_ple[..., :1].clamp_max(1.0),
                    x_ple[..., 1:-1].clamp(0.0, 1.0),
                    (
                        x_ple[..., -1:].clamp_min(0.0)
                        if self.single_bin_mask is None
                        else torch.where(
                            # For features with only one bin,
                            # the whole "piecewise-linear" encoding effectively behaves
                            # like MinMax scaling
                            # (assuming that the edges of the single bin
                            #  are the minimum and maximum feature values).
                            self.single_bin_mask[..., None],
                            x_ple[..., -1:],
                            x_ple[..., -1:].clamp_min(0.0),
                        )
                    ),
                ],
                dim=-1,
            )
        # Step 2: compute the piecewise-linear embeddings.
        # Intuition: embedding of a feature is the aggregation of its bin embeddings.
        x_ple = (x_ple[..., None, :] @ self.weight3).squeeze(-2)

        # >>> Result.
        return x_linear + x_ple


# TODO: redistribute the comments from PiecewiseLinearEmbeddingsV3 among the V4 classes.
class _PiecewiseLinearEncodingV4(nn.Module):
    weight: Tensor
    bias: Tensor
    single_bin_mask: None | Tensor
    mask: None | Tensor

    def __init__(self, bins: list[Tensor]) -> None:
        assert len(bins) > 0
        super().__init__()

        n_features = len(bins)
        n_bins = [len(x) - 1 for x in bins]
        max_n_bins = max(n_bins)

        # Making weight2 and bias2 trainable can lead to *worse* performance.
        self.register_buffer('weight', torch.zeros(n_features, max_n_bins))
        self.register_buffer('bias', torch.zeros(n_features, max_n_bins))

        single_bin_mask = torch.tensor(n_bins) == 1
        self.register_buffer(
            'single_bin_mask', single_bin_mask if single_bin_mask.any() else None
        )

        self.register_buffer(
            'mask',
            # The mask is needed if features have different number of bins.
            None
            if all(len(x) == len(bins[0]) for x in bins)
            else torch.row_stack(
                [
                    torch.cat(
                        [
                            # The number of bins for this feature, minus 1:
                            torch.ones(len(x) - 2, dtype=torch.bool),
                            # Inactive bins (always zeros):
                            torch.zeros(max_n_bins - (len(x) - 1), dtype=torch.bool),
                            # The last bin:
                            torch.ones(1, dtype=torch.bool),
                        ]
                    )
                    for x in bins
                ]
            ),
        )

        for i, bin_edges in enumerate(bins):
            bin_width = bin_edges.diff()
            w = 1.0 / bin_width
            b = -bin_edges[:-1] / bin_width
            self.weight[i, : len(bin_edges) - 2] = w[:-1]
            self.weight[i, -1] = w[-1]
            self.bias[i, : len(bin_edges) - 2] = b[:-1]
            self.bias[i, -1] = b[-1]

    def _encode1d(self, x: Tensor) -> Tensor:
        x = torch.addcmul(self.bias, self.weight, x[..., None])
        if x.shape[-1] > 1:
            x = torch.cat(
                [
                    x[..., :1].clamp_max(1.0),
                    x[..., 1:-1].clamp(0.0, 1.0),
                    (
                        x[..., -1:].clamp_min(0.0)
                        if self.single_bin_mask is None
                        else torch.where(
                            self.single_bin_mask[..., None],
                            x[..., -1:],
                            x[..., -1:].clamp_min(0.0),
                        )
                    ),
                ],
                dim=-1,
            )
        return x


class PiecewiseLinearEncoding0dV4(_PiecewiseLinearEncodingV4):
    def forward(self, x: Tensor) -> Tensor:
        x = self._encode1d(x)
        return x.flatten(-2) if self.mask is None else x[:, self.mask]


class PiecewiseLinearEncoding1dV4(_PiecewiseLinearEncodingV4):
    def forward(self, x: Tensor) -> Tensor:
        return self._encode1d(x)


# TODO: replace with delu.nn.NLinear with delu>=0.0.25
class NLinear(nn.Module):
    def __init__(
        self, n: int, in_features: int, out_features: int, bias: bool = True
    ) -> None:
        super().__init__()
        self.weight = Parameter(torch.empty(n, in_features, out_features))
        self.bias = Parameter(torch.empty(n, out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        d = self.weight.shape[-2]
        init_rsqrt_uniform_(self.weight, d)
        if self.bias is not None:
            init_rsqrt_uniform_(self.bias, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 3
        assert x.shape[-(self.weight.ndim - 1) :] == self.weight.shape[:-1]

        x = x.transpose(0, 1)
        x = x @ self.weight
        x = x.transpose(0, 1)
        if self.bias is not None:
            x = x + self.bias
        return x


class PiecewiseLinearEmbeddingsV4(nn.Module):
    def __init__(self, bins: list[Tensor], d_embedding: int) -> None:
        assert d_embedding > 0

        super().__init__()
        self.linear1 = rtdl_num_embeddings.LinearEmbeddings(len(bins), d_embedding)
        self.encoding = PiecewiseLinearEncoding1dV4(bins)
        self.linear2 = NLinear(
            len(bins), self.encoding.weight.shape[-1], d_embedding, bias=False
        )
        nn.init.zeros_(self.linear2.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear1(x) + self.linear2(self.encoding(x))


class CLSEmbedding(nn.Module):
    def __init__(self, d_embedding: int) -> None:
        super().__init__()
        self.weight = Parameter(torch.empty(d_embedding))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        d_rsqrt = self.weight.shape[-1] ** -0.5
        nn.init.uniform_(self.weight, -d_rsqrt, d_rsqrt)

    def forward(self, batch_dims: tuple[int]) -> Tensor:
        if not batch_dims:
            raise ValueError('The input must be non-empty')

        return self.weight.expand(*batch_dims, 1, -1)


class OneHotEncoding0d(nn.Module):
    # Input:  (*, n_cat_features=len(cardinalities))
    # Output: (*, sum(cardinalities))

    def __init__(self, cardinalities: list[int]) -> None:
        super().__init__()
        self._cardinalities = cardinalities

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim >= 1
        assert x.shape[-1] == len(self._cardinalities)

        return torch.cat(
            [
                # NOTE
                # This is a quick hack to support out-of-vocabulary categories.
                #
                # Recall that lib.data.transform_cat encodes categorical features
                # as follows:
                # - In-vocabulary values receive indices from `range(cardinality)`.
                # - All out-of-vocabulary values (i.e. new categories in validation
                #   and test data that are not presented in the training data)
                #   receive the index `cardinality`.
                #
                # As such, the line below will produce the standard one-hot encoding for
                # known categories, and the all-zeros encoding for unknown categories.
                # This may not be the best approach to deal with unknown values,
                # but should be enough for our purposes.
                F.one_hot(x[..., i], cardinality + 1)[..., :-1]
                for i, cardinality in enumerate(self._cardinalities)
            ],
            -1,
        )


# The following implementation is equivalent, it may be faster on GPU,
# but I did not notice any speedup on CPU.
# class OneHotEncoding0d(nn.Module):
#     # Input:  (*, n_cat_features=len(cardinalities))
#     # Output: (*, sum(cardinalities))

#     def __init__(self, cardinalities: list[int]) -> None:
#         super().__init__()
#         self._cardinalities = cardinalities
#         self._locations = [0, *torch.tensor(cardinalities).cumsum(0).tolist()]
#         self._out_size = sum(cardinalities)

#     def forward(self, x: Tensor) -> Tensor:
#         assert x.ndim >= 1
#         assert x.shape[-1] == len(self._cardinalities)

#         x_out = torch.empty((*x.shape, self._out_size), device=x.device)
#         for i, cardinality in enumerate(self._cardinalities):
#             x_out[..., self._locations[i] : self._locations[i + 1]] = F.one_hot(
#                 x[..., i], cardinality + 1
#             )[..., :-1].unsqueeze(-2)
#         return x_out


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
            m = cast(nn.Embedding, m)
            d_rsqrt = 1 / math.sqrt(m.weight.shape[-1])
            nn.init.uniform_(m.weight, -d_rsqrt, d_rsqrt)

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim >= 1
        return torch.stack(
            [m(x[..., i]) for i, m in enumerate(self.embeddings)], dim=-2
        )


class ScaleEnsemble(nn.Module):
    def __init__(
        self,
        k: int,
        d: int,
        *,
        init: Literal['ones', 'normal', 'random-signs'],
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(k, d))
        self._weight_init = init
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self._weight_init == 'ones':
            nn.init.ones_(self.weight)
        elif self._weight_init == 'normal':
            nn.init.normal_(self.weight)
        elif self._weight_init == 'random-signs':
            init_random_signs_(self.weight)
        else:
            raise ValueError(f'Unknown weight_init: {self._weight_init}')

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim >= 2
        return x * self.weight


class ElementwiseAffineEnsemble(nn.Module):
    def __init__(
        self,
        k: int,
        d: int,
        *,
        weight: bool,
        bias: bool,
        weight_init: Literal['ones', 'normal', 'random-signs'],
    ) -> None:
        assert weight or bias
        super().__init__()
        self.weight = nn.Parameter(torch.empty(k, d)) if weight else None
        self.bias = nn.Parameter(torch.empty(k, d)) if bias else None
        self._weight_init = weight_init
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.weight is not None:
            if self._weight_init == 'ones':
                nn.init.ones_(self.weight)
            elif self._weight_init == 'normal':
                nn.init.normal_(self.weight)
            elif self._weight_init == 'random-signs':
                init_random_signs_(self.weight)
            else:
                raise ValueError(f'Unknown weight_init: {self._weight_init}')
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        # x.shape == (B, K, D)
        assert x.ndim == 3
        return (
            x * self.weight
            if self.bias is None
            else x + self.bias
            if self.weight is None
            else torch.addcmul(self.bias, self.weight, x)
        )


class LinearEfficientEnsemble(nn.Module):
    """
    This layer is a more configurable version of the "BatchEnsemble" layer
    from the paper
    "BatchEnsemble: An Alternative Approach to Efficient Ensemble and Lifelong Learning"
    (link: https://arxiv.org/abs/2002.06715).

    First, this layer allows to select only some of the "ensembled" parts:
    - the input scaling  (r_i in the BatchEnsemble paper)
    - the output scaling (s_i in the BatchEnsemble paper)
    - the output bias    (not mentioned in the BatchEnsemble paper,
                          but is presented in public implementations)

    Second, the initialization of the scaling weights is configurable
    through the `scaling_init` argument.
    """

    r: None | Tensor
    s: None | Tensor
    bias: None | Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        k: int,
        ensemble_scaling_in: bool,
        ensemble_scaling_out: bool,
        ensemble_bias: bool,
        scaling_init: Literal['ones', 'random-signs'],
    ):
        assert k > 0
        if ensemble_bias:
            assert bias
        super().__init__()

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.register_parameter(
            'r',
            (
                nn.Parameter(torch.empty(k, in_features))
                if ensemble_scaling_in
                else None
            ),  # type: ignore[code]
        )
        self.register_parameter(
            's',
            (
                nn.Parameter(torch.empty(k, out_features))
                if ensemble_scaling_out
                else None
            ),  # type: ignore[code]
        )
        self.register_parameter(
            'bias',
            (
                nn.Parameter(torch.empty(out_features))  # type: ignore[code]
                if bias and not ensemble_bias
                else nn.Parameter(torch.empty(k, out_features))
                if ensemble_bias
                else None
            ),
        )

        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        self.scaling_init = scaling_init

        self.reset_parameters()

    def reset_parameters(self):
        init_rsqrt_uniform_(self.weight, self.in_features)
        scaling_init_fn = {'ones': nn.init.ones_, 'random-signs': init_random_signs_}[
            self.scaling_init
        ]
        if self.r is not None:
            scaling_init_fn(self.r)
        if self.s is not None:
            scaling_init_fn(self.s)
        if self.bias is not None:
            bias_init = torch.empty(
                # NOTE: the shape of bias_init is (out_features,) not (k, out_features).
                # It means that all biases have the same initialization.
                # This is similar to having one shared bias plus
                # k zero-initialized non-shared biases.
                self.out_features,
                dtype=self.weight.dtype,
                device=self.weight.device,
            )
            bias_init = init_rsqrt_uniform_(bias_init, self.in_features)
            with torch.inference_mode():
                self.bias.copy_(bias_init)

    def forward(self, x: Tensor) -> Tensor:
        # x.shape == (B, K, D)
        assert x.ndim == 3

        # >>> The equation (5) from the BatchEnsemble paper (arXiv v2).
        if self.r is not None:
            x = x * self.r
        x = x @ self.weight.T
        if self.s is not None:
            x = x * self.s
        # <<<

        if self.bias is not None:
            x = x + self.bias
        return x


def make_efficient_ensemble(module: nn.Module, **kwargs) -> None:
    for name, submodule in list(module.named_children()):
        if isinstance(submodule, nn.Linear):
            module.add_module(
                name,
                LinearEfficientEnsemble(
                    in_features=submodule.in_features,
                    out_features=submodule.out_features,
                    bias=submodule.bias is not None,
                    **kwargs,
                ),
            )
        else:
            make_efficient_ensemble(submodule, **kwargs)


class MLP(nn.Module):
    def __init__(
        self,
        *,
        d_in: None | int = None,
        d_out: None | int = None,
        n_blocks: int,
        d_block: int,
        dropout: float,
        activation: str = 'ReLU',
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


_CUSTOM_MODULES = {
    # https://docs.python.org/3/library/stdtypes.html#definition.__name__
    CustomModule.__name__: CustomModule
    for CustomModule in [
        rtdl_num_embeddings.LinearEmbeddings,
        rtdl_num_embeddings.LinearReLUEmbeddings,
        rtdl_num_embeddings.PeriodicEmbeddings,
        rtdl_num_embeddings.PiecewiseLinearEncoding,
        rtdl_num_embeddings.PiecewiseLinearEmbeddings,
        PiecewiseLinearEmbeddingsV2,
        PiecewiseLinearEmbeddingsV3,
        PiecewiseLinearEncoding0dV4,
        PiecewiseLinearEncoding1dV4,
        PiecewiseLinearEmbeddingsV4,
        MLP,
        ResNet,
    ]
}


def register_module(name: str, factory: Callable[..., nn.Module]) -> None:
    if _CUSTOM_MODULES.setdefault(name, factory) is not factory:
        raise RuntimeError(f'The module "{name}" is already registered.')


def make_module(type: str, *args, **kwargs) -> nn.Module:
    Module = getattr(nn, type, None)
    if Module is None:
        Module = _CUSTOM_MODULES[type]
    return Module(*args, **kwargs)


def get_n_parameters(m: nn.Module):
    return sum(x.numel() for x in m.parameters() if x.requires_grad)


def get_d_out(n_classes: None | int) -> int:
    return 1 if n_classes is None or n_classes == 2 else n_classes


@torch.inference_mode()
def compute_parameter_stats(module: nn.Module) -> dict[str, dict[str, float]]:
    stats = {'norm': {}, 'gradnorm': {}, 'gradratio': {}}
    for name, parameter in module.named_parameters():
        stats['norm'][name] = parameter.norm().item()
        if parameter.grad is not None:
            stats['gradnorm'][name] = parameter.grad.norm().item()
            # Avoid computing statistics for zero-initialized parameters.
            if (parameter.abs() > 1e-6).any():
                stats['gradratio'][name] = (
                    (parameter.grad.abs() / parameter.abs().clamp_min_(1e-6))
                    .mean()
                    .item()
                )
    stats['norm']['model'] = (
        torch.cat([x.flatten() for x in module.parameters()]).norm().item()
    )
    stats['gradnorm']['model'] = (
        torch.cat([x.grad.flatten() for x in module.parameters() if x.grad is not None])
        .norm()
        .item()
    )
    return stats


# ======================================================================================
# Optimization
# ======================================================================================
def default_zero_weight_decay_condition(
    module_name: str, module: nn.Module, parameter_name: str, parameter: Parameter
):
    from rtdl_num_embeddings import _Periodic

    del module_name, parameter
    return (
        parameter_name.endswith('bias')
        or isinstance(
            module,
            nn.BatchNorm1d
            | nn.LayerNorm
            | nn.InstanceNorm1d
            | rtdl_revisiting_models.LinearEmbeddings
            | rtdl_num_embeddings.LinearEmbeddings
            | rtdl_num_embeddings.LinearReLUEmbeddings
            | _Periodic,
        )
        or isinstance(module, PiecewiseLinearEmbeddingsV3)
        and parameter_name in ('weight1', 'bias1')
    )


def make_parameter_groups(
    module: nn.Module,
    zero_weight_decay_condition=default_zero_weight_decay_condition,
    custom_groups: None | list[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    if custom_groups is None:
        custom_groups = []
    custom_params = frozenset(
        itertools.chain.from_iterable(group['params'] for group in custom_groups)
    )
    assert len(custom_params) == sum(
        len(group['params']) for group in custom_groups
    ), 'Parameters in custom_groups must not intersect'
    zero_wd_params = frozenset(
        p
        for mn, m in module.named_modules()
        for pn, p in m.named_parameters()
        if p not in custom_params and zero_weight_decay_condition(mn, m, pn, p)
    )
    default_group = {
        'params': [
            p
            for p in module.parameters()
            if p not in custom_params and p not in zero_wd_params
        ]
    }
    return [
        default_group,
        {'params': list(zero_wd_params), 'weight_decay': 0.0},
        *custom_groups,
    ]


def make_optimizer(
    type: str,
    sam: None | dict[str, Any] = None,
    mechanic: bool = False,
    schedule_free: bool = False,
    **kwargs,
) -> torch.optim.Optimizer:
    assert sum((sam is not None, mechanic, schedule_free)) <= 1


    try:
        Optimizer = getattr(torch.optim, type)
    except AttributeError:
        import pytorch_optimizer # type: ignore

        Optimizer = getattr(pytorch_optimizer, type)

    return Optimizer(**kwargs)


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    return next(iter(optimizer.param_groups))['lr']


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group['lr'] = lr


# ======================================================================================
# Training
# ======================================================================================
def get_loss_fn(task_type: TaskType) -> Callable[..., Tensor]:
    return (
        F.binary_cross_entropy_with_logits
        if task_type == TaskType.BINCLASS
        else F.cross_entropy
        if task_type == TaskType.MULTICLASS
        else F.mse_loss
    )


def zero_grad_forward_backward(
    optimizer: torch.optim.Optimizer,
    step_fn: Callable[[Tensor], Tensor],  # step_fn: chunk_idx -> loss
    batch_idx: Tensor,
    chunk_size: int,
    grad_scaler: None | torch.cuda.amp.GradScaler = None,  # type: ignore[code]
) -> tuple[Tensor, int]:
    backward = (
        Tensor.backward
        if grad_scaler is None
        else lambda x: grad_scaler.scale(x).backward()  # type: ignore[code]
    )
    batch_size = len(batch_idx)
    loss = None
    while chunk_size != 0:
        optimizer.zero_grad()

        try:
            if batch_size <= chunk_size:
                # The simple forward-backward.
                loss = step_fn(batch_idx)
                backward(loss)
            else:
                # Forward-backward by chunks.
                # Mathematically, this is equivalent to the simple forward-backward.
                # Technically, this implementations uses less memory.
                loss = None
                for chunk_idx in batch_idx.split(chunk_size):
                    chunk_loss = step_fn(chunk_idx)
                    chunk_loss = chunk_loss * (len(chunk_idx) / batch_size)
                    backward(chunk_loss)
                    if loss is None:
                        loss = chunk_loss.detach()
                    else:
                        loss += chunk_loss.detach()
        except RuntimeError as err:
            if not is_oom_exception(err):
                raise
            delu.cuda.free_memory()
            chunk_size //= 2

        else:
            break

    if not chunk_size:
        raise RuntimeError('Not enough memory even for chunk_size=1')
    return cast(Tensor, loss), chunk_size
'''