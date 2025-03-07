import math
import warnings
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Union
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch
try:
    import sklearn.tree as sklearn_tree
except ImportError:
    sklearn_tree = None

def _check_bins(bins: List[Tensor]) -> None:
    if not bins:
        raise ValueError('The list of bins must not be empty')
    for i, feature_bins in enumerate(bins):
        if not isinstance(feature_bins, Tensor):
            raise ValueError(
                'bins must be a list of PyTorch tensors. '
                f'However, for {i=}: {type(bins[i])=}'
            )
        if feature_bins.ndim != 1:
            raise ValueError(
                'Each item of the bin list must have exactly one dimension.'
                f' However, for {i=}: {bins[i].ndim=}'
            )
        if len(feature_bins) < 2:
            raise ValueError(
                'All features must have at least two bin edges.'
                f' However, for {i=}: {len(bins[i])=}'
            )
        if not feature_bins.isfinite().all():
            raise ValueError(
                'Bin edges must not contain nan/inf/-inf.'
                f' However, this is not true for the {i}-th feature'
            )
        if (feature_bins[:-1] >= feature_bins[1:]).any():
            raise ValueError(
                'Bin edges must be sorted.'
                f' However, the for the {i}-th feature, the bin edges are not sorted'
            )
        if len(feature_bins) == 2:
            warnings.warn(
                f'The {i}-th feature has just two bin edges, which means only one bin.'
                ' Strictly speaking, using a single bin for the'
                ' piecewise-linear encoding should not break anything,'
                ' but it is the same as using sklearn.preprocessing.MinMaxScaler'
            )

def compute_bins(
    X: torch.Tensor,
    n_bins: int = 48,
    *,
    tree_kwargs: Optional[Dict[str, Any]] = None,
    y: Optional[Tensor] = None,
    regression: Optional[bool] = None,
    verbose: bool = False,
) -> List[Tensor]:
    # Source: https://github.com/yandex-research/rtdl-num-embeddings/blob/main/package/rtdl_num_embeddings.py
    """Compute bin edges for `PiecewiseLinearEmbeddings`.

    **Usage**

    Computing the quantile-based bins (Section 3.2.1 in the paper):

    >>> X_train = torch.randn(10000, 2)
    >>> bins = compute_bins(X_train)

    Computing the tree-based bins (Section 3.2.2 in the paper):

    >>> X_train = torch.randn(10000, 2)
    >>> y_train = torch.randn(len(X_train))
    >>> bins = compute_bins(
    ...     X_train,
    ...     y=y_train,
    ...     regression=True,
    ...     tree_kwargs={'min_samples_leaf': 64, 'min_impurity_decrease': 1e-4},
    ... )

    Args:
        X: the training features.
        n_bins: the number of bins.
        tree_kwargs: keyword arguments for `sklearn.tree.DecisionTreeRegressor`
            (if ``regression`` is `True`) or `sklearn.tree.DecisionTreeClassifier`
            (if ``regression`` is `False`).
            NOTE: requires ``scikit-learn>=1.0,>2`` to be installed.
        y: the training labels (must be provided if ``tree`` is not None).
        regression: whether the labels are regression labels
            (must be provided if ``tree`` is not None).
        verbose: if True and ``tree_kwargs`` is not None, than ``tqdm``
            (must be installed) will report the progress while fitting trees.
    Returns:
        A list of bin edges for all features. For one feature:

        - the maximum possible number of bin edges is ``n_bins + 1``.
        - the minumum possible number of bin edges is ``1``.
    """
    if not isinstance(X, Tensor):
        raise ValueError(f'X must be a PyTorch tensor, however: {type(X)=}')
    if X.ndim != 2:
        raise ValueError(f'X must have exactly two dimensions, however: {X.ndim=}')
    if X.shape[0] < 2:
        raise ValueError(f'X must have at least two rows, however: {X.shape[0]=}')
    if X.shape[1] < 1:
        raise ValueError(f'X must have at least one column, however: {X.shape[1]=}')
    if not X.isfinite().all():
        raise ValueError('X must not contain nan/inf/-inf.')
    if (X == X[0]).all(dim=0).any():
        raise ValueError(
            'All columns of X must have at least two distinct values.'
            ' However, X contains columns with just one distinct value.'
        )
    if n_bins <= 1 or n_bins >= len(X):
        raise ValueError(
            'n_bins must be more than 1, but less than len(X), however:'
            f' {n_bins=}, {len(X)=}'
        )

    if tree_kwargs is None:
        if y is not None or regression is not None or verbose:
            raise ValueError(
                'If tree_kwargs is None, then y must be None, regression must be None'
                ' and verbose must be False'
            )

        # NOTE[DIFF]
        # The original implementation in the official paper repository has an
        # unintentional divergence from what is written in the paper.
        # This package implements the algorithm described in the paper,
        # and it is recommended for future work
        # (this may affect the optimal number of bins
        #  reported in the official repository).
        #
        # Additional notes:
        # - this is the line where the divergence happens:
        #   (the thing is that limiting the number of quantiles by the number of
        #   distinct values is NOT the same as removing identical quantiles
        #   after computing them)
        #   https://github.com/yandex-research/tabular-dl-num-embeddings/blob/c1d9eb63c0685b51d7e1bc081cdce6ffdb8886a8/bin/train4.py#L612C30-L612C30
        # - for the tree-based bins, there is NO such divergence;
        bins = [
            q.unique()
            for q in torch.quantile(
                X, torch.linspace(0.0, 1.0, n_bins + 1).to(X), dim=0
            ).T
        ]
        _check_bins(bins)
        return bins
    else:
        if sklearn_tree is None:
            raise RuntimeError(
                'The scikit-learn package is missing.'
                ' See README.md for installation instructions'
            )
        if y is None or regression is None:
            raise ValueError(
                'If tree_kwargs is not None, then y and regression must not be None'
            )
        if y.ndim != 1:
            raise ValueError(f'y must have exactly one dimension, however: {y.ndim=}')
        if len(y) != len(X):
            raise ValueError(
                f'len(y) must be equal to len(X), however: {len(y)=}, {len(X)=}'
            )
        if y is None or regression is None:
            raise ValueError(
                'If tree_kwargs is not None, then y and regression must not be None'
            )
        if 'max_leaf_nodes' in tree_kwargs:
            raise ValueError(
                'tree_kwargs must not contain the key "max_leaf_nodes"'
                ' (it will be set to n_bins automatically).'
            )

        if verbose:
            if tqdm is None:
                raise ImportError('If verbose is True, tqdm must be installed')
            tqdm_ = tqdm
        else:
            tqdm_ = lambda x: x  # noqa: E731

        if X.device.type != 'cpu' or y.device.type != 'cpu':
            warnings.warn(
                'Computing tree-based bins involves the conversion of the input PyTorch'
                ' tensors to NumPy arrays. The provided PyTorch tensors are not'
                ' located on CPU, so the conversion has some overhead.',
                UserWarning,
            )
        X_numpy = X.cpu().numpy()
        y_numpy = y.cpu().numpy()
        bins = []
        for column in tqdm_(X_numpy.T):
            feature_bin_edges = [float(column.min()), float(column.max())]
            tree = (
                (
                    sklearn_tree.DecisionTreeRegressor
                    if regression
                    else sklearn_tree.DecisionTreeClassifier
                )(max_leaf_nodes=n_bins, **tree_kwargs)
                .fit(column.reshape(-1, 1), y_numpy)
                .tree_
            )
            for node_id in range(tree.node_count):
                # The following condition is True only for split nodes. Source:
                # https://scikit-learn.org/1.0/auto_examples/tree/plot_unveil_tree_structure.html#tree-structure
                if tree.children_left[node_id] != tree.children_right[node_id]:
                    feature_bin_edges.append(float(tree.threshold[node_id]))
            bins.append(torch.as_tensor(feature_bin_edges).unique())
        _check_bins(bins)
        return [x.to(device=X.device, dtype=X.dtype) for x in bins]
    
class _PiecewiseLinearEncodingImpl(nn.Module):
    # NOTE
    # 1. DO NOT USE THIS CLASS DIRECTLY (ITS OUTPUT CONTAINS INFINITE VALUES).
    # 2. This implementation is not memory efficient for cases when there are many
    #    features with low number of bins and only few features
    #    with high number of bins. If this becomes a problem,
    #    just split features into groups and encode the groups separately.

    # The output of this module has the shape (*batch_dims, n_features, max_n_bins),
    # where max_n_bins = max(map(len, bins)) - 1.
    # If the i-th feature has the number of bins less than max_n_bins,
    # then its piecewise-linear representation is padded with inf as follows:
    # [x_1, x_2, ..., x_k, inf, ..., inf]
    # where:
    #            x_1 <= 1.0
    #     0.0 <= x_i <= 1.0 (for i in range(2, k))
    #     0.0 <= x_k
    #     k == len(bins[i]) - 1  (the number of bins for the i-th feature)

    # If all features have the same number of bins, then there are no infinite values.

    edges: Tensor
    width: Tensor
    mask: Tensor
    # Source: https://github.com/yandex-research/rtdl-num-embeddings/blob/main/package/rtdl_num_embeddings.py
    def __init__(self, bins: List[Tensor]) -> None:
        _check_bins(bins)

        super().__init__()
        # To stack bins to a tensor, all features must have the same number of bins.
        # To achieve that, for each feature with a less-than-max number of bins,
        # its bins are padded with additional phantom bins with infinite edges.
        max_n_edges = max(len(x) for x in bins)
        padding = torch.full(
            (max_n_edges,),
            math.inf,
            dtype=bins[0].dtype,
            device=bins[0].device,
        )
        edges = torch.row_stack([torch.cat([x, padding])[:max_n_edges] for x in bins])

        # The rightmost edge is needed only to compute the width of the rightmost bin.
        self.register_buffer('edges', edges[:, :-1])
        self.register_buffer('width', edges.diff())
        # mask is false for the padding values.
        self.register_buffer(
            'mask',
            torch.row_stack(
                [
                    torch.cat(
                        [
                            torch.ones(len(x) - 1, dtype=torch.bool, device=x.device),
                            torch.zeros(
                                max_n_edges - 1, dtype=torch.bool, device=x.device
                            ),
                        ]
                    )[: max_n_edges - 1]
                    for x in bins
                ]
            ),
        )
        self._bin_counts = tuple(len(x) - 1 for x in bins)
        self._same_bin_count = all(x == self._bin_counts[0] for x in self._bin_counts)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim < 2:
            raise ValueError(
                f'The input must have at least two dimensions, however: {x.ndim=}'
            )

        # See Equation 1 in the paper.
        x = (x[..., None] - self.edges) / self.width

        # If the number of bins is greater than 1, then, the following rules must
        # be applied to a piecewise-linear encoding of a single feature:
        # - the leftmost value can be negative, but not greater than 1.0.
        # - the rightmost value can be greater than 1.0, but not negative.
        # - the intermediate values must stay within [0.0, 1.0].
        n_bins = x.shape[-1]
        if n_bins > 1:
            if self._same_bin_count:
                x = torch.cat(
                    [
                        x[..., :1].clamp_max(1.0),
                        *([] if n_bins == 2 else [x[..., 1:-1].clamp(0.0, 1.0)]),
                        x[..., -1:].clamp_min(0.0),
                    ],
                    dim=-1,
                )
            else:
                # In this case, the rightmost values for all features are located
                # in different columns.
                x = torch.stack(
                    [
                        x[..., i, :]
                        if count == 1
                        else torch.cat(
                            [
                                x[..., i, :1].clamp_max(1.0),
                                *(
                                    []
                                    if n_bins == 2
                                    else [x[..., i, 1 : count - 1].clamp(0.0, 1.0)]
                                ),
                                x[..., i, count - 1 : count].clamp_min(0.0),
                                x[..., i, count:],
                            ],
                            dim=-1,
                        )
                        for i, count in enumerate(self._bin_counts)
                    ],
                    dim=-2,
                )
        return x


class PiecewiseLinearEncoding(nn.Module):
    """Piecewise-linear encoding.

    **Shape**

    - Input: ``(*, n_features)``
    - Output: ``(*, n_features, total_n_bins)``,
      where ``total_n_bins`` is the total number of bins for all features:
      ``total_n_bins = sum(len(b) - 1 for b in bins)``.
    """
    # Source: https://github.com/yandex-research/rtdl-num-embeddings/blob/main/package/rtdl_num_embeddings.py
    def __init__(self, bins: List[Tensor]) -> None:
        """
        Args:
            bins: the bins computed by `compute_bins`.
        """
        _check_bins(bins)

        super().__init__()
        self.impl = _PiecewiseLinearEncodingImpl(bins)

    def forward(self, x: Tensor) -> Tensor:
        x = self.impl(x)
        return x.flatten(-2) if self.impl._same_bin_count else x[:, self.impl.mask]
    

class _UnaryEncodingImpl(nn.Module):
    edges: Tensor
    mask: Tensor

    def __init__(self, bins: List[Tensor]) -> None:
        _check_bins(bins)

        super().__init__()
        # To stack bins to a tensor, all features must have the same number of bins.
        # To achieve that, for each feature with a less-than-max number of bins,
        # its bins are padded with additional phantom bins with infinite edges.
        max_n_edges = max(len(x) for x in bins)
        padding = torch.full(
            (max_n_edges,),
            math.inf,
            dtype=bins[0].dtype,
            device=bins[0].device,
        )
        edges = torch.row_stack([torch.cat([x, padding])[:max_n_edges] for x in bins])

        # The rightmost edge is needed only to compute the width of the rightmost bin.
        self.register_buffer('edges', edges[:, :-1])
        # mask is false for the padding values.
        self.register_buffer(
            'mask',
            torch.row_stack(
                [
                    torch.cat(
                        [
                            torch.ones(len(x) - 1, dtype=torch.bool, device=x.device),
                            torch.zeros(
                                max_n_edges - 1, dtype=torch.bool, device=x.device
                            ),
                        ]
                    )[: max_n_edges - 1]
                    for x in bins
                ]
            ),
        )
        self._bin_counts = tuple(len(x) - 1 for x in bins)
        self._same_bin_count = all(x == self._bin_counts[0] for x in self._bin_counts)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim < 2:
            raise ValueError(
                f'The input must have at least two dimensions, however: {x.ndim=}'
            )

        # Compute which bin each value falls into
        x = (x[..., None] - self.edges).sign().cumsum(dim=-1)

        # Ensure values are within [0, 1] range for unary encoding
        x = x.clamp(0, 1)

        return x


class UnaryEncoding(nn.Module):
    """Unary encoding.

    **Shape**

    - Input: ``(*, n_features)``
    - Output: ``(*, n_features, total_n_bins)``,
      where ``total_n_bins`` is the total number of bins for all features:
      ``total_n_bins = sum(len(b) - 1 for b in bins)``.
    """

    def __init__(self, bins: List[Tensor]) -> None:
        """
        Args:
            bins: the bins computed by `compute_bins`.
        """
        _check_bins(bins)

        super().__init__()
        self.impl = _UnaryEncodingImpl(bins)

    def forward(self, x: Tensor) -> Tensor:
        x = self.impl(x)
        return x.flatten(-2) if self.impl._same_bin_count else x[:, self.impl.mask]

class _JohnsonEncodingImpl(nn.Module):
    edges: Tensor
    mask: Tensor

    def __init__(self, bins: List[Tensor]) -> None:
        _check_bins(bins)

        super().__init__()
        # To stack bins to a tensor, all features must have the same number of bins.
        # To achieve that, for each feature with a less-than-max number of bins,
        # its bins are padded with additional phantom bins with infinite edges.
        max_n_edges = max(len(x) for x in bins)
        padding = torch.full(
            (max_n_edges,),
            math.inf,
            dtype=bins[0].dtype,
            device=bins[0].device,
        )
        edges = torch.row_stack([torch.cat([x, padding])[:max_n_edges] for x in bins])

        # The rightmost edge is needed only to compute the width of the rightmost bin.
        self.register_buffer('edges', edges[:, :-1])
        self.register_buffer('width', edges.diff())
        # mask is false for the padding values.
        self.register_buffer(
            'mask',
            torch.row_stack(
                [
                    torch.cat(
                        [
                            torch.ones(len(x) - 1, dtype=torch.bool, device=x.device),
                            torch.zeros(
                                max_n_edges - 1, dtype=torch.bool, device=x.device
                            ),
                        ]
                    )[: max_n_edges - 1]
                    for x in bins
                ]
            ),
        )
        self._bin_counts = tuple(len(x) - 1 for x in bins)
        self._same_bin_count = all(x == self._bin_counts[0] for x in self._bin_counts)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim < 2:
            raise ValueError(
                f'The input must have at least two dimensions, however: {x.ndim=}'
            )

        # Compute which bin each value falls into
        bin_indices = torch.stack([torch.bucketize(x[..., i], self.edges[i], right=True) - 1 for i in range(x.shape[-1])], dim=-1)

        # Generate Johnson code for each bin index
        max_bin = self.edges.shape[1] 
        code_length = (max_bin + 1) // 2
        johnson_code = torch.zeros(*x.shape, code_length, device=x.device, dtype=torch.float32)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                johnson_code[i, j, :] = self.temp_code(bin_indices[i, j].item(), max_bin)

        return johnson_code

    def temp_code(self, num, num_bits):
        num_bits = num_bits+1 if num_bits%2!=0 else num_bits
        bits = num_bits // 2
        a = torch.zeros([bits], dtype=torch.long)
        for i in range(bits):
            if bits - i - 1 < num <= num_bits - i - 1:
                a[i] = 1
        return a


class JohnsonEncoding(nn.Module):
    """Johnson encoding.

    **Shape**

    - Input: ``(*, n_features)``
    - Output: ``(*, n_features, total_n_bits)``,
      where ``total_n_bits`` is the total number of bits for all features:
      ``total_n_bits = sum((len(b) - 1) // 2 for b in bins)``.
    """

    def __init__(self, bins: List[Tensor]) -> None:
        """
        Args:
            bins: the bins computed by `compute_bins`.
        """
        _check_bins(bins)

        super().__init__()
        self.impl = _JohnsonEncodingImpl(bins)

    def forward(self, x: Tensor) -> Tensor:
        x = self.impl(x)
        return x.flatten(-2) # if self.impl._same_bin_count else x[:, self.impl.mask]
    
class _BinsEncodingImpl(nn.Module):
    edges: Tensor
    mask: Tensor

    def __init__(self, bins: List[Tensor]) -> None:
        _check_bins(bins)

        super().__init__()
        # To stack bins to a tensor, all features must have the same number of bins.
        # To achieve that, for each feature with a less-than-max number of bins,
        # its bins are padded with additional phantom bins with infinite edges.
        max_n_edges = max(len(x) for x in bins)
        padding = torch.full(
            (max_n_edges,),
            math.inf,
            dtype=bins[0].dtype,
            device=bins[0].device,
        )
        edges = torch.row_stack([torch.cat([x, padding])[:max_n_edges] for x in bins])

        # The rightmost edge is needed only to compute the width of the rightmost bin.
        self.register_buffer('edges', edges[:, :-1])
        # mask is false for the padding values.
        self.register_buffer(
            'mask',
            torch.row_stack(
                [
                    torch.cat(
                        [
                            torch.ones(len(x) - 1, dtype=torch.bool, device=x.device),
                            torch.zeros(
                                max_n_edges - 1, dtype=torch.bool, device=x.device
                            ),
                        ]
                    )[: max_n_edges - 1]
                    for x in bins
                ]
            ),
        )
        self._bin_counts = tuple(len(x) - 1 for x in bins)
        self._same_bin_count = all(x == self._bin_counts[0] for x in self._bin_counts)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim < 2:
            raise ValueError(
                f'The input must have at least two dimensions, however: {x.ndim=}'
            )

        # Compute which bin each value falls into
        bin_indices = torch.stack([torch.bucketize(x[..., i], self.edges[i], right=True) - 1 for i in range(x.shape[-1])], dim=-1)

        return bin_indices

class BinsEncoding(nn.Module):
    """Bins encoding.

    **Shape**

    - Input: ``(*, n_features)``
    - Output: ``(*, n_features, total_n_bins)``,
      where ``total_n_bins`` is the total number of bins for all features.
    """

    def __init__(self, bins: List[Tensor]) -> None:
        """
        Args:
            bins: the bins computed by `compute_bins`.
        """
        _check_bins(bins)

        super().__init__()
        self.impl = _BinsEncodingImpl(bins)

    def forward(self, x: Tensor) -> Tensor:
        return self.impl(x)