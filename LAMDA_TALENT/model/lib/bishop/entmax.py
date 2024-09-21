"""
An implementation of entmax (Peters et al., 2019). See
https://arxiv.org/pdf/1905.05702 for detailed description.

This builds on previous work with sparsemax (Martins & Astudillo, 2016).
See https://arxiv.org/pdf/1602.02068.

The code is from https://github.com/deep-spin/entmax/blob/master/entmax/activations.py.
"""

import torch

import torch
import random
import numpy as np

def _make_ix_like(X, dim):
  d = X.size(dim)
  rho = torch.arange(1, d + 1, device=X.device, dtype=X.dtype)
  view = [1] * X.dim()
  view[0] = -1
  return rho.view(view).transpose(0, dim)


def _roll_last(X, dim):
  if dim == -1:
      return X
  elif dim < 0:
      dim = X.dim() - dim

  perm = [i for i in range(X.dim()) if i != dim] + [dim]
  return X.permute(perm)


def _sparsemax_threshold_and_support(X, dim=-1, k=None):
  """Core computation for sparsemax: optimal threshold and support size.

  Parameters
  ----------
  X : torch.Tensor
      The input tensor to compute thresholds over.

  dim : int
      The dimension along which to apply sparsemax.

  k : int or None
      number of largest elements to partial-sort over. For optimal
      performance, should be slightly bigger than the expected number of
      nonzeros in the solution. If the solution is more than k-sparse,
      this function is recursively called with a 2*k schedule.
      If `None`, full sorting is performed from the beginning.

  Returns
  -------
  tau : torch.Tensor like `X`, with all but the `dim` dimension intact
      the threshold value for each vector
  support_size : torch LongTensor, shape like `tau`
      the number of nonzeros in each vector.
  """

  if k is None or k >= X.shape[dim]:  # do full sort
    topk, _ = torch.sort(X, dim=dim, descending=True)
  else:
    topk, _ = torch.topk(X, k=k, dim=dim)

  topk_cumsum = topk.cumsum(dim) - 1
  rhos = _make_ix_like(topk, dim)
  support = rhos * topk > topk_cumsum

  support_size = support.sum(dim=dim).unsqueeze(dim)
  tau = topk_cumsum.gather(dim, support_size - 1)
  tau /= support_size.to(X.dtype)

  if k is not None and k < X.shape[dim]:
    unsolved = (support_size == k).squeeze(dim)

    if torch.any(unsolved):
      in_ = _roll_last(X, dim)[unsolved]
      tau_, ss_ = _sparsemax_threshold_and_support(in_, dim=-1, k=2 * k)
      _roll_last(tau, dim)[unsolved] = tau_
      _roll_last(support_size, dim)[unsolved] = ss_

  return tau, support_size


def _entmax_threshold_and_support(X, dim=-1, k=None):
  """Core computation for 1.5-entmax: optimal threshold and support size.

  Parameters
  ----------
  X : torch.Tensor
      The input tensor to compute thresholds over.

  dim : int
      The dimension along which to apply 1.5-entmax.

  k : int or None
      number of largest elements to partial-sort over. For optimal
      performance, should be slightly bigger than the expected number of
      nonzeros in the solution. If the solution is more than k-sparse,
      this function is recursively called with a 2*k schedule.
      If `None`, full sorting is performed from the beginning.

  Returns
  -------
  tau : torch.Tensor like `X`, with all but the `dim` dimension intact
      the threshold value for each vector
  support_size : torch LongTensor, shape like `tau`
      the number of nonzeros in each vector.
  """

  if k is None or k >= X.shape[dim]:  # do full sort
    Xsrt, _ = torch.sort(X, dim=dim, descending=True)
  else:
    Xsrt, _ = torch.topk(X, k=k, dim=dim)

  rho = _make_ix_like(Xsrt, dim)
  mean = Xsrt.cumsum(dim) / rho
  mean_sq = (Xsrt ** 2).cumsum(dim) / rho
  ss = rho * (mean_sq - mean ** 2)
  delta = (1 - ss) / rho

  # NOTE this is not exactly the same as in reference algo
  # Fortunately it seems the clamped values never wrongly
  # get selected by tau <= sorted_z. Prove this!
  delta_nz = torch.clamp(delta, 0)
  tau = mean - torch.sqrt(delta_nz)

  support_size = (tau <= Xsrt).sum(dim).unsqueeze(dim)
  tau_star = tau.gather(dim, support_size - 1)

  if k is not None and k < X.shape[dim]:
    unsolved = (support_size == k).squeeze(dim)

    if torch.any(unsolved):
      X_ = _roll_last(X, dim)[unsolved]
      tau_, ss_ = _entmax_threshold_and_support(X_, dim=-1, k=2 * k)
      _roll_last(tau_star, dim)[unsolved] = tau_
      _roll_last(support_size, dim)[unsolved] = ss_

  return tau_star, support_size


class SparsemaxFunction(torch.autograd.Function):
  @classmethod
  def forward(cls, ctx, X, dim=-1, k=None):
    ctx.dim = dim
    max_val, _ = X.max(dim=dim, keepdim=True)
    X = X - max_val  # same numerical stability trick as softmax
    tau, supp_size = _sparsemax_threshold_and_support(X, dim=dim, k=k)
    output = torch.clamp(X - tau, min=0)
    ctx.save_for_backward(supp_size, output)
    return output

  @classmethod
  def backward(cls, ctx, grad_output):
    supp_size, output = ctx.saved_tensors
    dim = ctx.dim
    grad_input = grad_output.clone()
    grad_input[output == 0] = 0

    v_hat = grad_input.sum(dim=dim) / supp_size.to(output.dtype).squeeze(dim)
    v_hat = v_hat.unsqueeze(dim)
    grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
    return grad_input, None, None


class Entmax15Function(torch.autograd.Function):
  @classmethod
  def forward(cls, ctx, X, dim=0, k=None):
    ctx.dim = dim

    max_val, _ = X.max(dim=dim, keepdim=True)
    X = X - max_val  # same numerical stability trick as for softmax
    X = X / 2  # divide by 2 to solve actual Entmax

    tau_star, _ = _entmax_threshold_and_support(X, dim=dim, k=k)

    Y = torch.clamp(X - tau_star, min=0) ** 2
    ctx.save_for_backward(Y)
    return Y

  @classmethod
  def backward(cls, ctx, dY):
    Y, = ctx.saved_tensors
    gppr = Y.sqrt()  # = 1 / g'' (Y)
    dX = dY * gppr
    q = dX.sum(ctx.dim) / gppr.sum(ctx.dim)
    q = q.unsqueeze(ctx.dim)
    dX -= q * gppr
    return dX, None, None


def sparsemax(X, dim=-1, k=None):
  """sparsemax: normalizing sparse transform (a la softmax).

  Solves the projection:

      min_p ||x - p||_2   s.t.    p >= 0, sum(p) == 1.

  Parameters
  ----------
  X : torch.Tensor
      The input tensor.

  dim : int
      The dimension along which to apply sparsemax.

  k : int or None
      number of largest elements to partial-sort over. For optimal
      performance, should be slightly bigger than the expected number of
      nonzeros in the solution. If the solution is more than k-sparse,
      this function is recursively called with a 2*k schedule.
      If `None`, full sorting is performed from the beginning.

  Returns
  -------
  P : torch tensor, same shape as X
      The projection result, such that P.sum(dim=dim) == 1 elementwise.
  """

  return SparsemaxFunction.apply(X, dim, k)


def entmax15(X, dim=-1, k=None):
  """1.5-entmax: normalizing sparse transform (a la softmax).

  Solves the optimization problem:

      max_p <x, p> - H_1.5(p)    s.t.    p >= 0, sum(p) == 1.

  where H_1.5(p) is the Tsallis alpha-entropy with alpha=1.5.

  Parameters
  ----------
  X : torch.Tensor
      The input tensor.

  dim : int
      The dimension along which to apply 1.5-entmax.

  k : int or None
      number of largest elements to partial-sort over. For optimal
      performance, should be slightly bigger than the expected number of
      nonzeros in the solution. If the solution is more than k-sparse,
      this function is recursively called with a 2*k schedule.
      If `None`, full sorting is performed from the beginning.

  Returns
  -------
  P : torch tensor, same shape as X
      The projection result, such that P.sum(dim=dim) == 1 elementwise.
  """

  return Entmax15Function.apply(X, dim, k)


class Sparsemax(torch.nn.Module):
  def __init__(self, dim=-1, k=None):
    """sparsemax: normalizing sparse transform (a la softmax).

    Solves the projection:

        min_p ||x - p||_2   s.t.    p >= 0, sum(p) == 1.

    Parameters
    ----------
    dim : int
        The dimension along which to apply sparsemax.

    k : int or None
        number of largest elements to partial-sort over. For optimal
        performance, should be slightly bigger than the expected number of
        nonzeros in the solution. If the solution is more than k-sparse,
        this function is recursively called with a 2*k schedule.
        If `None`, full sorting is performed from the beginning.
    """
    self.dim = dim
    self.k = k
    super(Sparsemax, self).__init__()

  def forward(self, X):
    return sparsemax(X, dim=self.dim, k=self.k)


class Entmax15(torch.nn.Module):
  def __init__(self, dim=-1, k=None):
    """1.5-entmax: normalizing sparse transform (a la softmax).

    Solves the optimization problem:

        max_p <x, p> - H_1.5(p)    s.t.    p >= 0, sum(p) == 1.

    where H_1.5(p) is the Tsallis alpha-entropy with alpha=1.5.

    Parameters
    ----------
    dim : int
        The dimension along which to apply 1.5-entmax.

    k : int or None
        number of largest elements to partial-sort over. For optimal
        performance, should be slightly bigger than the expected number of
        nonzeros in the solution. If the solution is more than k-sparse,
        this function is recursively called with a 2*k schedule.
        If `None`, full sorting is performed from the beginning.
    """
    self.dim = dim
    self.k = k
    super(Entmax15, self).__init__()

  def forward(self, X):
    return entmax15(X, dim=self.dim, k=self.k)

## Implementation of Entmax has been adapted from https://github.com/deep-spin/entmax/
from pathlib import Path

import torch

home = str(Path.home())


class AlphaChooser(torch.nn.Module):
    def __init__(self, head_count):
        super(AlphaChooser, self).__init__()
        self.pre_alpha = torch.nn.Parameter(torch.randn(head_count))
        self.head_count = head_count

    def forward(self):
        alpha = 1 + torch.sigmoid(self.pre_alpha)
        return torch.clamp(alpha, min=1+1e-8, max=2)


class EntmaxAlpha(torch.nn.Module):
    def __init__(self, head_count=1, dim=-1):
        super(EntmaxAlpha, self).__init__()
        self.dim = dim
        self.alpha_chooser = torch.nn.Parameter(AlphaChooser(head_count)())
        self.alpha = self.alpha_chooser

    def forward(self, att_scores):
        batch_size, head_count, query_len, key_len = att_scores.size()
        expanded_alpha = (
            self.alpha.unsqueeze(0).unsqueeze(-1)
        )  # [1,nb_heads,1,1]
        expanded_alpha = expanded_alpha.repeat(1, head_count, 1, 1)

        expanded_alpha = expanded_alpha.expand(
            (batch_size, -1, query_len, 1)
        )  # [bs, nb_heads, query_len,1]
        p_star = entmax_bisect(att_scores, expanded_alpha)
        return p_star


class EntmaxBisectFunction(torch.autograd.Function):
    @classmethod
    def _gp(cls, x, alpha):
        return x ** (alpha - 1)

    @classmethod
    def _gp_inv(cls, y, alpha):
        return y ** (1 / (alpha - 1))

    @classmethod
    def _p(cls, X, alpha):
        return cls._gp_inv(torch.clamp(X, min=0), alpha)

    @classmethod
    def forward(cls, ctx, X, alpha=1.5, dim=-1, n_iter=50, ensure_sum_one=True):

        if not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha, dtype=X.dtype, device=X.device)

        alpha_shape = list(X.shape)
        alpha_shape[dim] = 1
        alpha = alpha.expand(*alpha_shape)

        ctx.alpha = alpha
        ctx.dim = dim
        d = X.shape[dim]

        X = X * (alpha - 1)

        max_val, _ = X.max(dim=dim, keepdim=True)

        tau_lo = max_val - cls._gp(1, alpha)
        tau_hi = max_val - cls._gp(1 / d, alpha)

        f_lo = cls._p(X - tau_lo, alpha).sum(dim) - 1

        dm = tau_hi - tau_lo

        for it in range(n_iter):

            dm /= 2
            tau_m = tau_lo + dm
            p_m = cls._p(X - tau_m, alpha)
            f_m = p_m.sum(dim) - 1

            mask = (f_m * f_lo >= 0).unsqueeze(dim)
            tau_lo = torch.where(mask, tau_m, tau_lo)

        if ensure_sum_one:
            p_m /= p_m.sum(dim=dim).unsqueeze(dim=dim)

        ctx.save_for_backward(p_m)

        return p_m

    @classmethod
    def backward(cls, ctx, dY):
        (Y,) = ctx.saved_tensors

        gppr = torch.where(Y > 0, Y ** (2 - ctx.alpha), Y.new_zeros(1))

        dX = dY * gppr
        q = dX.sum(ctx.dim) / gppr.sum(ctx.dim)
        q = q.unsqueeze(ctx.dim)
        dX -= q * gppr

        d_alpha = None
        if ctx.needs_input_grad[1]:

            # alpha gradient computation
            # d_alpha = (partial_y / partial_alpha) * dY
            # NOTE: ensure alpha is not close to 1
            # since there is an indetermination
            # batch_size, _ = dY.shape

            # shannon terms
            S = torch.where(Y > 0, Y * torch.log(Y), Y.new_zeros(1))
            # shannon entropy
            ent = S.sum(ctx.dim).unsqueeze(ctx.dim)
            Y_skewed = gppr / gppr.sum(ctx.dim).unsqueeze(ctx.dim)

            d_alpha = dY * (Y - Y_skewed) / ((ctx.alpha - 1) ** 2)
            d_alpha -= dY * (S - Y_skewed * ent) / (ctx.alpha - 1)
            d_alpha = d_alpha.sum(ctx.dim).unsqueeze(ctx.dim)

        return dX, d_alpha, None, None, None


def entmax_bisect(X, alpha=1.5, dim=-1, n_iter=50, ensure_sum_one=True):
    """alpha-entmax: normalizing sparse transform (a la softmax).
    Solves the optimization problem:
        max_p <x, p> - H_a(p)    s.t.    p >= 0, sum(p) == 1.
    where H_a(p) is the Tsallis alpha-entropy with custom alpha >= 1,
    using a bisection (root finding, binary search) algorithm.
    This function is differentiable with respect to both X and alpha.
    Parameters
    ----------
    X : torch.Tensor
        The input tensor.
    alpha : float or torch.Tensor
        Tensor of alpha parameters (> 1) to use. If scalar
        or python float, the same value is used for all rows, otherwise,
        it must have shape (or be expandable to)
        alpha.shape[j] == (X.shape[j] if j != dim else 1)
        A value of alpha=2 corresponds to sparsemax, and alpha=1 corresponds to
        softmax (but computing it this way is likely unstable).
    dim : int
        The dimension along which to apply alpha-entmax.
    n_iter : int
        Number of bisection iterations. For float32, 24 iterations should
        suffice for machine precision.
    ensure_sum_one : bool,
        Whether to divide the result by its sum. If false, the result might
        sum to close but not exactly 1, which might cause downstream problems.
    Returns
    -------
    P : torch tensor, same shape as X
        The projection result, such that P.sum(dim=dim) == 1 elementwise.
    """
    return EntmaxBisectFunction.apply(X, alpha, dim, n_iter, ensure_sum_one)