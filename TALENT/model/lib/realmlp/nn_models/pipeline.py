from typing import List, Dict, Union

import sklearn
import torch
from sklearn.base import BaseEstimator

from TALENT.model.lib.realmlp import utils
from TALENT.model.lib.realmlp.data.data import TensorInfo, DictDataset
from TALENT.model.lib.realmlp.nn_models.base import Layer, Variable, Fitter, FitterFactory, IdentityLayer, BiasLayer, ScaleLayer
from TALENT.model.lib.realmlp.torch_utils import torch_np_quantile


# todo: add factories


class ReplaceMissingContLayer(Layer):
    def __init__(self, means: Variable):
        super().__init__()
        if not isinstance(means, Variable):
           raise ValueError('means is not a Variable')
        self.means = means

    def forward_cont(self, x):
        return torch.where(torch.isnan(x), self.means, x)

    def _stack(self, layers: List['ReplaceMissingContLayer']):
        return ReplaceMissingContLayer(Variable.stack([layer.means for layer in layers]))


class MeanReplaceMissingContFactory(Fitter, FitterFactory):
    def __init__(self, trainable=False, **config):
        super().__init__(is_individual=trainable, modified_tensors=['x_cont'])
        self.trainable = trainable

    def get_n_params(self, tensor_infos: Dict[str, TensorInfo]) -> int:
        return self._get_n_values(tensor_infos, ['x_cont'])

    def _fit(self, ds: DictDataset) -> Layer:
        if ds.tensor_infos['x_cont'].is_empty():
            return IdentityLayer()
        x_cont = ds.tensors['x_cont']
        is_nan = torch.isnan(x_cont)
        x_cont_replaced = torch.where(is_nan, torch.zeros_like(x_cont), x_cont)
        means = x_cont_replaced.sum(dim=-2, keepdim=True) \
                / (x_cont.shape[-2] - is_nan.float().sum(dim=-2, keepdim=True) + 1e-30)
        return ReplaceMissingContLayer(Variable(means, trainable=self.trainable))


class MeanCenterFactory(Fitter, FitterFactory):
    def __init__(self, trainable=False, **config):
        super().__init__(is_individual=trainable, modified_tensors=['x_cont'])
        self.trainable = trainable

    def get_n_params(self, tensor_infos: Dict[str, TensorInfo]) -> int:
        return self._get_n_values(tensor_infos, ['x_cont'])

    def _fit(self, ds: DictDataset) -> Layer:
        if ds.tensor_infos['x_cont'].is_empty():
            return IdentityLayer()
        return BiasLayer(Variable(-ds.tensors['x_cont'].mean(dim=-2, keepdim=True),
                         trainable=self.trainable))


class MedianCenterFactory(Fitter, FitterFactory):
    def __init__(self, median_center_trainable=False, **config):
        super().__init__(is_individual=median_center_trainable, modified_tensors=['x_cont'])
        self.trainable = median_center_trainable

    def get_n_params(self, tensor_infos: Dict[str, TensorInfo]) -> int:
        return self._get_n_values(tensor_infos, ['x_cont'])

    def _fit(self, ds: DictDataset) -> Layer:
        # quantile requires PyTorch >= 1.7.0
        if ds.tensor_infos['x_cont'].is_empty():
            return IdentityLayer()

        # use quantile function from numpy since the torch one can use large amounts of RAM for some reason
        return BiasLayer(Variable(-torch_np_quantile(ds.tensors['x_cont'], 0.5, dim=-2, keepdim=True),
                         trainable=self.trainable))


class L2NormalizeFactory(Fitter, FitterFactory):
    def __init__(self, trainable=False, l2_normalize_eps=1e-8, **config):
        super().__init__(is_individual=trainable, modified_tensors=['x_cont'])
        self.trainable = trainable
        self.eps = l2_normalize_eps

    def get_n_params(self, tensor_infos: Dict[str, TensorInfo]) -> int:
        return self._get_n_values(tensor_infos, ['x_cont'])

    def _fit(self, ds: DictDataset) -> Layer:
        if ds.tensor_infos['x_cont'].is_empty():
            return IdentityLayer()
        scale = 1.0 / (ds.tensors['x_cont'] ** 2 + self.eps).mean(dim=-2, keepdim=True).sqrt()
        scale[:, (ds.tensors['x_cont']**2).mean(dim=-2) == 0.0] = 0.0
        return ScaleLayer(Variable(scale, trainable=self.trainable))


class L1NormalizeFactory(Fitter, FitterFactory):
    def __init__(self, trainable=False, eps=1e-8, **config):
        super().__init__(is_individual=trainable, modified_tensors=['x_cont'])
        self.trainable = trainable
        self.eps = eps

    def get_n_params(self, tensor_infos: Dict[str, TensorInfo]) -> int:
        return self._get_n_values(tensor_infos, ['x_cont'])

    def _fit(self, ds: DictDataset) -> Layer:
        if ds.tensor_infos['x_cont'].is_empty():
            return IdentityLayer()
        scale = 1.0 / (ds.tensors['x_cont'].abs() + self.eps).mean(dim=-2, keepdim=True)
        return ScaleLayer(Variable(scale, trainable=self.trainable))


class RobustScaleFactory(Fitter, FitterFactory):
    def __init__(self, robust_scale_trainable=False, robust_scale_eps=1e-30, **config):
        super().__init__(is_individual=robust_scale_trainable, modified_tensors=['x_cont'])
        self.trainable = robust_scale_trainable
        self.robust_scale_eps = robust_scale_eps

    def get_n_params(self, tensor_infos: Dict[str, TensorInfo]) -> int:
        return self._get_n_values(tensor_infos, ['x_cont'])

    def _fit(self, ds: DictDataset) -> Layer:
        if ds.tensor_infos['x_cont'].is_empty():
            return IdentityLayer()
        x_cont = ds.tensors['x_cont']
        quant_diff = torch_np_quantile(x_cont, 0.75, dim=-2) - torch_np_quantile(x_cont, 0.25, dim=-2)
        max, _ = x_cont.max(dim=-2)
        min, _ = x_cont.min(dim=-2)
        idxs = quant_diff == 0.0
        quant_diff[idxs] = 0.5 * (max[idxs] - min[idxs])
        factors = 1.0 / (quant_diff + self.robust_scale_eps)
        factors[quant_diff == 0.0] = 0.0
        return ScaleLayer(Variable(factors[None, :], trainable=self.trainable))


class RobustScaleV2Factory(Fitter, FitterFactory):
    def __init__(self, robust_scale_trainable=False, **config):
        super().__init__(is_individual=robust_scale_trainable, modified_tensors=['x_cont'])
        self.trainable = robust_scale_trainable

    def get_n_params(self, tensor_infos: Dict[str, TensorInfo]) -> int:
        return self._get_n_values(tensor_infos, ['x_cont'])

    def _fit(self, ds: DictDataset) -> Layer:
        if ds.tensor_infos['x_cont'].is_empty():
            return IdentityLayer()
        x_cont = ds.tensors['x_cont']
        x_cont_sorted, _ = torch.sort(x_cont, dim=-2)
        quantiles = torch.linspace(0.0, 1.0, x_cont.shape[-2], device=x_cont.device)
        opposite_dists = x_cont_sorted.flip(dims=[-2]) - x_cont_sorted
        opposite_quantile_dists = quantiles.flip(dims=[0]) - quantiles
        quarter_idx = x_cont.shape[-2] // 4 + 1
        possible_factors = 2.0 * opposite_quantile_dists[:quarter_idx, None] / \
                           (1e-30 + opposite_dists[..., :quarter_idx, :])
        factors = possible_factors.min(dim=-2, keepdim=True)[0]
        return ScaleLayer(Variable(factors, trainable=self.trainable))


class GlobalScaleNormalizeFactory(Fitter, FitterFactory):
    def __init__(self, global_scale_factor=1.0, **config):
        super().__init__(is_individual=False, modified_tensors=['x_cont'])
        self.global_scale_factor = global_scale_factor

    def get_n_params(self, tensor_infos: Dict[str, TensorInfo]) -> int:
        return self._get_n_values(tensor_infos, ['x_cont'])

    def _fit(self, ds: DictDataset) -> Layer:
        if ds.tensor_infos['x_cont'].is_empty():
            return IdentityLayer()
        x_cont = ds.tensors['x_cont']
        scale = self.global_scale_factor / (x_cont**2 + 1e-30).mean().sqrt().item()
        return ScaleLayer(Variable(scale * torch.ones(1, 1, device=x_cont.device), trainable=False))


class ThermometerCodingLayer(Layer):
    def __init__(self, centers: Variable, scale: float, fitter: Fitter):
        super().__init__(fitter=fitter)
        self.centers = centers
        self.scale = scale  # todo: could make scale a variable and allow for different scales per center

    def forward_cont(self, x):
        shifted = self.scale * (x.unsqueeze(-1) - self.centers)
        return torch.tanh(shifted.reshape(list(x.shape[:-1]) + [-1]))

    def _stack(self, layers):
        return ThermometerCodingLayer(Variable.stack([l.centers for l in layers]), layers[0].scale, layers[0].fitter)


class ThermometerCodingFactory(Fitter, FitterFactory):
    def __init__(self, tc_low=-1.0, tc_high=1.0, tc_num=3, tc_scale=1.0, **config):
        super().__init__(needs_tensors=False, is_individual=False, modified_tensors=['x_cont'])
        self.tc_low = tc_low
        self.tc_high = tc_high
        self.tc_num = tc_num
        self.tc_scale = tc_scale

    def get_n_params(self, tensor_infos: Dict[str, TensorInfo]) -> int:
        return self.tc_num

    def forward_tensor_infos(self, tensor_infos):
        n_cont = tensor_infos['x_cont'].get_n_features()
        return utils.join_dicts(tensor_infos, {'x_cont': TensorInfo(feat_shape=[n_cont * self.tc_num])})

    def _fit(self, ds: DictDataset) -> Layer:
        if ds.tensor_infos['x_cont'].is_empty():
            return IdentityLayer()
        centers = torch.linspace(self.tc_low, self.tc_high, self.tc_num, device=ds.device)[None, None, :]
        centers = Variable(centers, trainable=False)
        return ThermometerCodingLayer(centers=centers, scale=self.tc_scale, fitter=self)


class CircleCodingLayer(Layer):
    def __init__(self, scale: float, fitter: Fitter):
        super().__init__(fitter=fitter)
        self.scale = scale

    def forward_cont(self, x):
        x = (1.0 / self.scale) * x
        factor = 1.0 / torch.sqrt(1.0 + x**2)
        return torch.cat([x * factor, torch.ones_like(x) * factor], dim=-1)

    def _stack(self, layers):
        return CircleCodingLayer(layers[0].scale, layers[0].fitter)


class CircleCodingFactory(Fitter, FitterFactory):
    def __init__(self, circle_coding_scale=1.0, **config):
        super().__init__(needs_tensors=False, is_individual=False, modified_tensors=['x_cont'])
        self.scale = circle_coding_scale

    def forward_tensor_infos(self, tensor_infos):
        n_cont = tensor_infos['x_cont'].get_n_features()
        return utils.join_dicts(tensor_infos, {'x_cont': TensorInfo(feat_shape=[n_cont * 2])})

    def _fit(self, ds: DictDataset) -> Layer:
        if ds.tensor_infos['x_cont'].is_empty():
            return IdentityLayer()
        return CircleCodingLayer(scale=self.scale, fitter=self)


def apply_tfms_rec(tfms: Union[BaseEstimator, List], x: torch.Tensor):
    if isinstance(tfms, list):
        return torch.stack([apply_tfms_rec(tfm, x[i]) for i, tfm in enumerate(tfms)], dim=0)
    else:
        return torch.as_tensor(tfms.transform(x.detach().cpu().numpy()), dtype=x.dtype, device=x.device)


class SklearnTransformLayer(Layer):
    def __init__(self, tfms: Union[BaseEstimator, List], fitter: Fitter):
        super().__init__(fitter=fitter)
        self.tfms = tfms

    def forward_cont(self, x):
        return apply_tfms_rec(self.tfms, x)

    def _stack(self, layers):
        return SklearnTransformLayer(tfms=[l.tfms for l in layers], fitter=layers[0].fitter)


class SklearnTransformFactory(Fitter, FitterFactory):
    def __init__(self, tfm: BaseEstimator, **config):
        super().__init__(needs_tensors=True, is_individual=False, modified_tensors=['x_cont'])
        self.tfm = tfm

    def _fit(self, ds: DictDataset) -> Layer:
        if ds.tensor_infos['x_cont'].is_empty():
            return IdentityLayer()
        tfm = sklearn.base.clone(self.tfm)
        tfm.fit(ds.tensors['x_cont'].detach().cpu().numpy())
        return SklearnTransformLayer(tfm, fitter=self)





