from typing import List

import torch
import numpy as np


def get_available_device_names() -> List['str']:
    device_names = ['cpu'] + [f'cuda:{i}' for i in range(torch.cuda.device_count())]
    if torch.backends.mps.is_available():
        device_names.append('mps')
    return device_names


def seeded_randperm(n, device, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    # todo: can this not be generated directly on the device?
    return torch.randperm(n, generator=generator).to(device)


def permute_idxs(idxs, seed):
    return idxs[seeded_randperm(idxs.shape[0], idxs.device, seed)]


def batch_randperm(n_batch, n, device='cpu'):
    # batched randperm:
    # https://discuss.pytorch.org/t/batched-shuffling-of-feature-vectors/30188/4
    # https://github.com/pytorch/pytorch/issues/42502
    return torch.stack([torch.randperm(n, device=device) for i in range(n_batch)], dim=0)


# from https://github.com/runopti/stg/blob/9f630968c4f14cff6da4e54421c497f24ac1e08e/python/stg/layers.py#L10
def gauss_cdf(x):
    return 0.5 * (1 + torch.erf(x / np.sqrt(2)))


class ClampWithIdentityGradientFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, low: torch.Tensor, high: torch.Tensor):
        return torch.minimum(torch.maximum(input, low), high)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None, None


def clamp_with_identity_gradient_func(x, low, high):
    return ClampWithIdentityGradientFunc.apply(x, low, high)


def cat_if_necessary(tensors: List[torch.Tensor], dim: int):
    """
    Implements torch.cat() but doesn't copy if only one tensor is provided.
    This can make it faster if no copying behavior is needed.
    :param tensors: Tensors to be concatenated.
    :param dim: Dimension in which the tensor should be concatenated.
    :return: The concatendated tensor.
    """
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim=dim)


def hash_tensor(tensor: torch.Tensor) -> int:
    # for debugging purposes, to print two tensor's hashes to see if they are equal
    # from https://discuss.pytorch.org/t/defining-hash-function-for-multi-dimensional-tensor/107531
    import pickle
    # the .numpy() appears to be necessary for equal tensors to have equal hashes
    return hash(pickle.dumps(tensor.detach().cpu().numpy()))


def torch_np_quantile(tensor: torch.Tensor, q: float, dim: int, keepdim: bool = False) -> torch.Tensor:
    """
    Alternative implementation for torch.quantile() using np.quantile()
    since the implementation of torch.quantile() uses too much RAM (extreme for Airlines_DepDelay_10M)
    and can fail for too large tensors.
    See also https://github.com/pytorch/pytorch/issues/64947
    :param tensor: tensor
    :param q: Quantile value.
    :param dim: As in torch.quantile()
    :param keepdim: As in torch.quantile()
    :return:
    """
    x_np = tensor.detach().cpu().numpy()
    q_np = np.quantile(x_np, q=q, axis=dim, keepdims=keepdim)
    return torch.as_tensor(q_np, device=tensor.device, dtype=tensor.dtype)

