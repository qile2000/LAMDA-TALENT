import torch
import torch.nn.functional as F
from typing import Dict

# ------ from fastai2
from torch.jit import script

from TALENT.model.lib.realmlp.data.data import TensorInfo, DictDataset
from TALENT.model.lib.realmlp.nn_models.base import Variable, Fitter, FitterFactory, FunctionFitter, Layer


@script
def _swish_jit_fwd(x): return x.mul(torch.sigmoid(x))


@script
def _swish_jit_bwd(x, grad_output):
    x_sigmoid = torch.sigmoid(x)
    return grad_output * (x_sigmoid * (1 + x * (1 - x_sigmoid)))


class _SwishJitAutoFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return _swish_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        return _swish_jit_bwd(x, grad_output)


# don't use the optimized version since this seems to behave slighty differently for Pytorch Lightning
# def swish(x): return _SwishJitAutoFn.apply(x)
def swish(x): return x * torch.sigmoid(x)


@script
def _mish_jit_fwd(x): return x.mul(torch.tanh(F.softplus(x)))


@script
def _mish_jit_bwd(x, grad_output):
    x_sigmoid = torch.sigmoid(x)
    x_tanh_sp = F.softplus(x).tanh()
    return grad_output.mul(x_tanh_sp + x * x_sigmoid * (1 - x_tanh_sp * x_tanh_sp))


class MishJitAutoFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return _mish_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return _mish_jit_bwd(x, grad_output)


# don't use the optimized version since this seems to behave slighty differently for Pytorch Lightning
# def mish(x): return MishJitAutoFn.apply(x)
def mish(x): return x.mul(torch.tanh(F.softplus(x)))

# ----- end fastai2


class ParametricActivationLayer(Layer):
    def __init__(self, f, weight):
        super().__init__()
        self.f = f
        self.weight = weight

    def forward_cont(self, x):
        # print(f'{self.weight.mean().item()=:g}')
        return x + (self.f(x) - x) * self.weight

    def _stack(self, layers):
        return ParametricActivationLayer(self.f, Variable.stack([l.weight for l in layers]))


class ParametricActivationFitter(Fitter):
    def __init__(self, f, **config):
        super().__init__(needs_tensors=False, is_individual=True, modified_tensors=['x_cont'])
        self.f = f
        self.act_lr_factor = config.get('act_lr_factor', 1.0)
        self.act_wd_factor = config.get('act_wd_factor', 1.0)

    def get_n_params(self, tensor_infos: Dict[str, TensorInfo]) -> int:
        return self._get_n_values(tensor_infos, ['x_cont'])

    def _fit(self, ds: DictDataset) -> Layer:
        n_cont = ds.tensor_infos['x_cont'].get_n_features()
        return ParametricActivationLayer(self.f, Variable(torch.ones(1, n_cont, device=ds.device),
                                                          trainable=True, hyper_factors={'lr': self.act_lr_factor,
                                                                                         'wd': self.act_wd_factor}))


class ActivationFactory(FitterFactory):
    def __init__(self, **config):
        super().__init__()
        self.config = config

    def _create(self, tensor_infos) -> Fitter:
        # todo: implement more activations, also parametric ones
        act_name = self.config.get('act_name', self.config.get('act', 'relu'))
        if act_name == 'relu':
            f = torch.relu
        elif act_name == 'selu':
            f = torch.selu
        elif act_name == 'swish' or act_name == 'silu':
            f = swish
        elif act_name == 'sswish':  # normalized by output variance
            f = lambda x: 1.6765 * swish(x)
        elif act_name == 'mish':
            f = mish
        elif act_name == 'smish':   # normalized by output variance
            f = lambda x: 1.6 * mish(x)
        elif act_name == 'gelu':
            f = F.gelu
        else:
            raise ValueError(f'Activation {act_name} unknown')

        if self.config.get('use_parametric_act', False):
            return ParametricActivationFitter(f, **self.config)
        else:
            return FunctionFitter(f)






