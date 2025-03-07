import copy
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from TALENT.model.lib.realmlp import utils
from TALENT.model.lib.realmlp.data.data import TensorInfo, DictDataset
from TALENT.model.lib.realmlp.nn_models.base import Fitter, Variable, WeightLayer, BiasLayer, ScaleLayer, FitterFactory, Layer, \
    TrainContext, sub_scope_context, SequentialFitter, SequentialLayer, FunctionLayer
from TALENT.model.lib.realmlp.torch_utils import gauss_cdf


class WeightFitter(Fitter):
    def __init__(self, out_features, **config):
        super().__init__(modified_tensors=['x_cont'])
        self.out_features = out_features
        self.weight_init_mode = config.get('weight_init_mode', 'normal')
        self.weight_init_gain = config.get('weight_init_gain', 1.0)
        self.weight_lr_factor = config.get('weight_lr_factor', 1.0)
        self.weight_l2_factor = config.get('weight_l2_factor', 1.0)
        self.weight_l1_factor = config.get('weight_l1_factor', 1.0)
        self.weight_wd_factor = config.get('weight_wd_factor', 1.0)
        # use abc parameterization here?
        # todo: ntk param can imply different learning rate factors for different optimizers
        #  also, the influence of Adam's epsilon can be different
        #  maybe this can be resolved using abc-style parameterization
        self.use_ntk_param = config.get('use_ntk_param', False)
        self.use_ntk_param_v2 = config.get('use_ntk_param_v2', False)
        self.use_ntk_param_v3 = config.get('use_ntk_param_v3', False)
        self.weight_param = config.get('weight_param', 'standard')
        if self.use_ntk_param:
            raise ValueError(f'use_ntk_param is discontinued, use weight_param="ntk" instead')
        if self.use_ntk_param_v2:
            raise ValueError(f'use_ntk_param_v2 is discontinued, use weight_param="ntk-v2" instead')
        if self.use_ntk_param_v3:
            raise ValueError(f'use_ntk_param_v3 is discontinued, use weight_param="ntk-v3" instead')
        self.use_norm_weight = config.get('use_norm_weight', False)
        self.norm_weight_transpose = config.get('norm_weight_transpose', False)
        self.layer_position = config.get('layer_position', None)
        self.weight_gain = config.get('weight_gain', 1.0)
        super().__init__(needs_tensors=self.weight_init_mode in ['std'])  # todo: adjust for some weight init modes

    def get_n_params(self, tensor_infos: Dict[str, TensorInfo]) -> int:
        return self.out_features * self._get_n_values(tensor_infos, ['x_cont'])

    def forward_tensor_infos(self, tensor_infos):
        return utils.update_dict(tensor_infos, {'x_cont': TensorInfo(feat_shape=[self.out_features])})

    def _fit(self, ds: DictDataset):
        in_features = ds.tensor_infos['x_cont'].get_n_features()
        init_factor = self.weight_init_gain * np.sqrt(1.0 / in_features)
        lr_factor = self.weight_lr_factor
        wd_factor = self.weight_wd_factor
        weight_gain = self.weight_gain
        l2_factor = self.weight_l2_factor
        l1_factor = self.weight_l1_factor
        if self.weight_param == 'xavier':
            # todo: this is not a parametrization, use weight_init_mode instead
            init_factor = self.weight_init_gain * np.sqrt(2.0 / (in_features + self.out_features))
        elif self.weight_param == 'ntk' or self.weight_param == 'ntk-v3':
            weight_gain = self.weight_gain * np.sqrt(1.0 / in_features)
            init_factor = self.weight_init_gain
        elif self.weight_param == 'ntk-old':
            lr_factor *= weight_gain * np.sqrt(1.0 / in_features)
            init_factor *= weight_gain
            weight_gain = 1.0
        elif self.weight_param == 'ntk-v2':
            lr_factor = self.weight_lr_factor * weight_gain * np.sqrt(1.0 / in_features)
            init_factor *= weight_gain
            weight_gain = 1.0
            # this is chosen because wd is multiplied by lr when performing weight decay,
            #  and the effective wd step size should not scale with in_features
            wd_factor = self.weight_wd_factor * np.sqrt(in_features) / weight_gain
            # print(f'{self.weight_gain=}, {lr_factor=}, {wd_factor=}')
        elif self.weight_param == 'ntk-adam':
            init_factor = self.weight_init_gain * self.weight_gain * np.sqrt(1.0 / in_features)
            lr_factor = self.weight_lr_factor * self.weight_gain / in_features / np.sqrt(self.out_features)
            wd_factor = self.weight_wd_factor
            l2_factor = self.weight_l2_factor * np.sqrt(1.0 / self.out_features)
            l1_factor = self.weight_l1_factor * np.sqrt(1.0 / self.out_features)
            weight_gain = 1.0
        elif self.weight_param == 'mup-adam':
            # following Table 3 in "Tuning Large Neural Networks via zero-Shot Hyperparameter Transfer"
            if self.layer_position == 'first':
                lr_factor = self.weight_lr_factor
            elif self.layer_position == 'middle':
                lr_factor = self.weight_lr_factor / in_features
            elif self.layer_position == 'last':
                init_factor = self.weight_init_gain / in_features
                lr_factor = self.weight_lr_factor / in_features
            else:
                raise ValueError(f'Unknown layer_position for mup-adam: {self.layer_position}')
        elif self.weight_param == 'mup-sgd':
            # following Table 3 in "Tuning Large Neural Networks via zero-Shot Hyperparameter Transfer"
            if self.layer_position == 'first':
                lr_factor = self.weight_lr_factor * self.out_features
            elif self.layer_position == 'middle':
                lr_factor = self.weight_lr_factor
            elif self.layer_position == 'last':
                init_factor = self.weight_init_gain / in_features
                lr_factor = self.weight_lr_factor / in_features
            else:
                raise ValueError(f'Unknown layer_position for mup-adam: {self.layer_position}')
        elif self.weight_param == 'mup-adam-custom':
            # following Table 3 in "Tuning Large Neural Networks via zero-Shot Hyperparameter Transfer"
            if self.layer_position == 'first':
                lr_factor = self.weight_lr_factor / in_features
            elif self.layer_position == 'middle':
                lr_factor = self.weight_lr_factor / in_features
            elif self.layer_position == 'last':
                init_factor = self.weight_init_gain / in_features
                lr_factor = self.weight_lr_factor / in_features
            else:
                raise ValueError(f'Unknown layer_position for mup-adam-custom: {self.layer_position}')
        elif self.weight_param == 'mup-adam-custom-2':
            # following Table 3 in "Tuning Large Neural Networks via zero-Shot Hyperparameter Transfer"
            # with custom weight decay factors
            if self.layer_position == 'first':
                lr_factor = self.weight_lr_factor / in_features
                wd_factor = self.weight_wd_factor * np.sqrt(in_features)
            elif self.layer_position == 'middle':
                lr_factor = self.weight_lr_factor / in_features
                wd_factor = self.weight_wd_factor * np.sqrt(in_features)
            elif self.layer_position == 'last':
                init_factor = self.weight_init_gain / in_features
                lr_factor = self.weight_lr_factor / in_features
                # unclear if this wd is the right one,
                # but here the lr_factor is already on the scale of the initialization
                wd_factor = self.weight_wd_factor
            else:
                raise ValueError(f'Unknown layer_position for mup-adam-custom-2: {self.layer_position}')
        elif self.weight_param == 'mup-sgd-custom':
            # following Table 3 in "Tuning Large Neural Networks via zero-Shot Hyperparameter Transfer"
            if self.layer_position == 'first':
                lr_factor = self.weight_lr_factor
            elif self.layer_position == 'middle':
                lr_factor = self.weight_lr_factor
            elif self.layer_position == 'last':
                init_factor = self.weight_init_gain / in_features
                lr_factor = self.weight_lr_factor / in_features
            else:
                raise ValueError(f'Unknown layer_position for mup-sgd-custom: {self.layer_position}')
        elif self.weight_param == 'standard':
            pass  # standard parameterization
        else:
            raise ValueError(f'Unknown weight_param "{self.weight_param}"')

        # pytorch default is
        # for weights:
        # kaiming_uniform from unif[-bound, bound]
        # bound = sqrt(3) * gain / sqrt(in_features)
        # gain = sqrt(2 / (1 + sqrt(5)^2)) = sqrt(1/3)
        # therefore bound = 1 / sqrt(in_features)
        # for biases it's also unif[-1/sqrt(in_features), 1/sqrt(in_features)]

        if self.weight_init_mode == 'normal':
            weight = torch.randn(in_features, self.out_features, device=ds.device)
        elif self.weight_init_mode == 'uniform':
            # include np.sqrt(3) to ensure variance = 1
            weight = np.sqrt(3) * (2 * torch.rand(in_features, self.out_features, device=ds.device) - 1)
        elif self.weight_init_mode == 'zeros' or self.weight_init_mode == 'zero':
            weight = torch.zeros(in_features, self.out_features, device=ds.device)
        elif self.weight_init_mode == 'std':
            weight = torch.randn(in_features, self.out_features, device=ds.device)
            x = ds.tensors['x_cont']
            weight = weight / x.matmul(weight_gain * init_factor * weight).std(dim=-2, correction=0, keepdim=True)
        elif self.weight_init_mode == 'sqmom':
            weight = torch.randn(in_features, self.out_features, device=ds.device)
            x = ds.tensors['x_cont']
            weight = weight / x.matmul(weight_gain * init_factor * weight).square().sum(dim=-2, keepdim=True).sqrt()
        else:
            raise ValueError(f'Unknown weight_init_mode: {self.weight_init_mode}')

        # print(f'{repr(weight)=}')
        # print(f'{hash_tensor(weight)=}')

        if self.use_norm_weight:
            factor = np.sqrt(self.out_features / in_features) if self.norm_weight_transpose else 1.0
            return NormWeightLayer(Variable(init_factor * weight, trainable=True,
                                            hyper_factors={'lr': lr_factor, 'wd': wd_factor,
                                                           'l2': l2_factor, 'l1': l1_factor}),
                                   factor=weight_gain * factor, fitter=self, transpose=self.norm_weight_transpose)
        else:
            return WeightLayer(Variable(init_factor * weight, trainable=True,
                                        hyper_factors={'lr': lr_factor, 'wd': wd_factor,
                                                       'l2': l2_factor, 'l1': l1_factor}), factor=weight_gain)


class BiasFitter(Fitter):
    def __init__(self, **config):
        super().__init__(modified_tensors=['x_cont'])
        self.in_features = config.get('in_features', None)
        self.bias_init_mode = config.get('bias_init_mode', 'zeros')
        self.bias_init_gain = config.get('bias_init_gain', 1.0)
        self.bias_lr_factor = config.get('bias_lr_factor', 1.0)
        self.bias_l1_reg_factor = config.get('bias_l1_reg_factor', 1.0)
        self.bias_l2_reg_factor = config.get('bias_l2_reg_factor', 1.0)
        self.bias_wd_factor = config.get('bias_wd_factor', 1.0)
        self.bias_param = config.get('bias_param', 'standard')
        self.layer_position = config.get('layer_position', None)
        self.bias_gain = config.get('bias_gain', 1.0)
        # todo: adjust for some bias init modes
        super().__init__(
            needs_tensors=self.bias_init_mode in ['he+5', 'mean', 'neg-uniform-dynamic', 'neg-uniform-dynamic-2',
                                                  'normal-dynamic'])

    def get_n_params(self, tensor_infos: Dict[str, TensorInfo]) -> int:
        return self._get_n_values(tensor_infos, ['x_cont'])

    def heplus_bias(self, x, n_simplex):
        idxs = torch.randint(0, x.shape[0], size=(x.shape[1], n_simplex), device=x.device)
        simplex_weights = torch.distributions.Exponential(1.0).sample((x.shape[1], n_simplex))
        simplex_weights = simplex_weights.to(x.device)
        simplex_weights /= simplex_weights.sum(dim=1)[:, None]
        out_selected = torch.stack([x[idxs[:, i], torch.arange(x.shape[1], device=x.device)]
                                    for i in range(n_simplex)], dim=1)
        return -(out_selected * simplex_weights).sum(dim=1)

    def _fit(self, ds: DictDataset):
        n_features = ds.tensor_infos['x_cont'].get_n_features()

        lr_factor = self.bias_lr_factor
        bias_gain = self.bias_gain
        l2_factor = self.bias_l2_reg_factor
        l1_factor = self.bias_l1_reg_factor
        wd_factor = self.bias_wd_factor

        if self.bias_param == 'mup-sgd' and self.layer_position == 'first':
            # corresponds to fan_out in Table 3 of "Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer"
            lr_factor *= self.in_features
        elif self.bias_param == 'ntk-adam':
            lr_factor = self.bias_lr_factor / np.sqrt(n_features)
            l1_factor = self.bias_l1_reg_factor / np.sqrt(n_features)
            l2_factor = self.bias_l2_reg_factor / np.sqrt(n_features)

        if self.bias_init_mode == 'zeros' or self.bias_init_mode == 'zero':
            bias = torch.zeros(n_features, device=ds.device)
        elif self.bias_init_mode == 'normal':
            bias = torch.randn(n_features, device=ds.device)
        elif self.bias_init_mode == 'uniform':
            # include np.sqrt(3) to ensure variance = 1
            bias = np.sqrt(3) * (2 * torch.rand(n_features, device=ds.device) - 1)
        elif self.bias_init_mode == 'neg-uniform':
            bias = np.sqrt(3) * (-torch.rand(n_features, device=ds.device))
        elif self.bias_init_mode == 'neg-uniform-dynamic':
            mean = ds.tensors['x_cont'].mean(dim=-2)
            std = ds.tensors['x_cont'].std(dim=-2, correction=0)
            bias = -std * (mean + np.sqrt(3) * torch.rand(n_features, device=ds.device))
        elif self.bias_init_mode == 'neg-uniform-dynamic-2':
            mean = ds.tensors['x_cont'].mean(dim=-2)
            std = ds.tensors['x_cont'].std(dim=-2, correction=0)
            bias = -mean - std * np.sqrt(3) * torch.rand(n_features, device=ds.device)
        elif self.bias_init_mode == 'normal-dynamic':
            mean = ds.tensors['x_cont'].mean(dim=-2)
            std = ds.tensors['x_cont'].std(dim=-2, correction=0)
            bias = -mean + std * torch.randn(n_features, device=ds.device)
        elif self.bias_init_mode == 'he+5':
            bias = self.heplus_bias(ds.tensors['x_cont'], 5)
        elif self.bias_init_mode == 'mean':
            bias = -ds.tensors['x_cont'].mean(dim=-2)
        elif self.bias_init_mode == 'pytorch-default':
            bias = np.sqrt(1.0 / self.in_features) * (2 * torch.rand(n_features, device=ds.device) - 1)
        else:
            raise ValueError(f'Unknown bias_init_mode: {self.bias_init_mode}')

        # print(f'{repr(bias)=}')
        # print(f'{hash_tensor(bias)=}')

        return BiasLayer(Variable(self.bias_init_gain * bias[None, :] / bias_gain, trainable=True,
                                  hyper_factors={'lr': lr_factor, 'wd': wd_factor,
                                                 'l1_reg': l1_factor,
                                                 'l2_reg': l2_factor}),
                         factor=bias_gain)


class ScaleFitter(Fitter):
    def __init__(self, **config):
        super().__init__(needs_tensors=False, modified_tensors=['x_cont'])
        self.scale_init_gain = config.get('scale_init_gain', 1.0)
        self.scale_lr_factor = config.get('scale_lr_factor', 1.0)
        self.scale_wd_factor = config.get('scale_wd_factor', 1.0)
        self.scale_l2_reg_factor = config.get('scale_l2_reg_factor', 1.0)
        self.scale_l1_reg_factor = config.get('scale_l1_reg_factor', 1.0)
        self.scale_trainable = config.get('scale_trainable', True)
        self.scale_param = config.get('scale_param', 'standard')

    def get_n_params(self, tensor_infos: Dict[str, TensorInfo]) -> int:
        return self._get_n_values(tensor_infos, ['x_cont'])

    def _fit(self, ds: DictDataset):
        in_features = ds.tensor_infos['x_cont'].get_n_features()
        lr_factor = self.scale_lr_factor
        init_gain = self.scale_init_gain
        wd_factor = self.scale_wd_factor
        l2_reg_factor = self.scale_l2_reg_factor
        l1_reg_factor = self.scale_l1_reg_factor

        if self.scale_param == 'mup-adam-custom':
            lr_factor = self.scale_lr_factor / in_features
        elif self.scale_param == 'ntk-v2':
            lr_factor = self.scale_lr_factor / np.sqrt(in_features)
        elif self.scale_param == 'ntk-adam':
            lr_factor = self.scale_lr_factor / np.sqrt(in_features)
        elif self.scale_param == 'ntk-adam-v2':
            lr_factor = self.scale_lr_factor / np.sqrt(in_features)
            l2_reg_factor = self.scale_l2_reg_factor / np.sqrt(in_features)
            l1_reg_factor = self.scale_l1_reg_factor / np.sqrt(in_features)
        n_features = ds.tensor_infos['x_cont'].get_n_features()
        scale = init_gain * torch.ones(n_features, device=ds.device)
        return ScaleLayer(Variable(scale[None, :], trainable=self.scale_trainable,
                                   hyper_factors={'lr': lr_factor, 'wd': wd_factor,
                                                  'l2_reg': l2_reg_factor, 'l1_reg': l1_reg_factor}))


class ScaleFactory(FitterFactory):
    def __init__(self, **config):
        super().__init__()
        self.config = config

    def _create(self, tensor_infos: Dict[str, TensorInfo]) -> Fitter:
        return ScaleFitter(**self.config)


class DropoutLayer(Layer):
    def __init__(self):
        super().__init__()
        self.hyper_getter = self.context.hp_manager.register_hyper('p_drop', self.context.scope)

    def forward_cont(self, x):
        p_drop = self.hyper_getter()
        if p_drop == 0.0:
            return x
        return F.dropout(x, p_drop, training=self.training)


class DropoutFitter(Fitter):
    def __init__(self):
        super().__init__(needs_tensors=False, modified_tensors=['x_cont'])

    def _fit(self, ds: DictDataset) -> Layer:
        return DropoutLayer()


class NoiseLayer(Layer):
    def __init__(self):
        super().__init__()
        self.sigma_getter = self.context.hp_manager.register_hyper('layer_noise_sigma', self.context.scope)

    def forward_cont(self, x):
        sigma = self.sigma_getter()
        if sigma == 0.0 or not self.training:
            return x
        return x + sigma * torch.randn_like(x)


class NoiseFitter(Fitter):
    def __init__(self, **config):
        super().__init__(needs_tensors=False, modified_tensors=['x_cont'])

    def _fit(self, ds: DictDataset) -> Layer:
        return NoiseLayer()


# ------ Regression output rescaling / clamping -------


class ClampLayer(Layer):
    def __init__(self, low: Variable, high: Variable):
        super().__init__()
        self.low = low
        self.high = high

    def forward_cont(self, x):
        if self.training:
            return x
        else:
            return torch.min(torch.max(x, self.low), self.high)

    def _stack(self, layers):
        return ClampLayer(Variable.stack([l.low for l in layers]),
                          Variable.stack([l.high for l in layers]))


class ClampOutputFactory(Fitter, FitterFactory):
    def __init__(self, **config):
        super().__init__(needs_tensors=False, modified_tensors=['x_cont'])
        self.config = config

    def get_n_params(self, tensor_infos: Dict[str, TensorInfo]) -> int:
        return 2 * self._get_n_values(tensor_infos, ['x_cont'])

    def _fit(self, ds: DictDataset) -> Layer:
        y = TrainContext.get_global_context().hp_manager.get_more_info_dict()['trainval_ds'].tensors['y']
        return ClampLayer(low=Variable(y.min(dim=-2, keepdim=True)[0], trainable=False),
                          high=Variable(y.max(dim=-2, keepdim=True)[0], trainable=False))


class NormalizeOutputLayer(Layer):
    def __init__(self, mean: Variable, std: Variable):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward_tensors(self, tensors):
        tensors = copy.copy(tensors)  # shallow copy
        if self.training:
            assert 'y' in tensors
            tensors['y'] = (tensors['y'] - self.mean) / (self.std + 1e-30)
        else:
            tensors['x_cont'] = tensors['x_cont'] * self.std + self.mean
        return tensors

    def _stack(self, layers):
        return NormalizeOutputLayer(mean=Variable.stack([l.mean for l in layers]),
                                    std=Variable.stack([l.std for l in layers]))


class NormalizeOutputFactory(Fitter, FitterFactory):
    def __init__(self, **config):
        super().__init__(needs_tensors=False, modified_tensors=['x_cont'])

    def get_n_params(self, tensor_infos: Dict[str, TensorInfo]) -> int:
        return 2 * self._get_n_values(tensor_infos, ['x_cont'])

    def _fit(self, ds: DictDataset) -> Layer:
        y = TrainContext.get_global_context().hp_manager.get_more_info_dict()['trainval_ds'].tensors['y']
        return NormalizeOutputLayer(mean=Variable(y.mean(dim=-2, keepdim=True), trainable=False),
                                    std=Variable(y.std(dim=-2, correction=0, keepdim=True), trainable=False))


class NormWeightLayer(Layer):
    def __init__(self, weight: Variable, factor: float, fitter: Fitter, transpose=False):
        super().__init__(fitter=fitter)
        self.weight = weight
        self.factor = factor
        self.transpose = transpose

    def forward_cont(self, x):
        return x.matmul(self.factor * self.weight / self.weight.norm(dim=-1 if self.transpose else -2, keepdim=True))

    def _stack(self, layers):
        return NormWeightLayer(weight=Variable.stack([l.weight for l in layers]),
                               factor=layers[0].factor,
                               fitter=layers[0].fitter,
                               transpose=layers[0].transpose)


class FixedScaleFactory(Fitter, FitterFactory):
    def __init__(self, scale: torch.Tensor):
        super().__init__(needs_tensors=False, is_individual=True, modified_tensors=['x_cont'])
        self.scale = scale

    def _fit(self, ds: DictDataset) -> Layer:
        return ScaleLayer(Variable(self.scale, trainable=False))


class FeatureImportanceFactory(Fitter, FitterFactory):
    def __init__(self):
        super().__init__(needs_tensors=False, is_individual=True, modified_tensors=['x_cont'])

    def _fit(self, ds: DictDataset) -> Layer:
        scale = TrainContext.get_global_context().hp_manager.get_more_info_dict()['feature_importances'][None, :]
        return ScaleLayer(Variable(scale.to(ds.device), trainable=False))


class FixedWeightFactory(Fitter, FitterFactory):
    def __init__(self):
        super().__init__(needs_tensors=False, is_individual=True, modified_tensors=['x_cont'])

    def _fit(self, ds: DictDataset) -> Layer:
        weight = TrainContext.get_global_context().hp_manager.get_more_info_dict()['fixed_weights']
        return WeightLayer(Variable(weight.to(ds.device), trainable=False))


class RFFeatureImportanceFactory(Fitter, FitterFactory):
    def __init__(self):
        super().__init__(needs_tensors=True, is_individual=True, modified_tensors=['x_cont'])

    def _fit(self, ds: DictDataset) -> Layer:
        x = ds.tensors['x_cont'].cpu().numpy()
        y = ds.tensors['y'].cpu().numpy()
        n_estimators = 50
        if ds.tensor_infos['y'].is_cont():
            # assume it's regression
            model = RandomForestRegressor(n_estimators=n_estimators, n_jobs=1)
        else:
            # assume it's classification
            model = RandomForestClassifier(n_estimators=n_estimators, n_jobs=1)
        model.fit(x, y)
        scale = torch.as_tensor(model.feature_importances_, dtype=torch.float32, device=ds.device)
        # print(f'RF feature importances: {scale}')
        scale *= np.sqrt(scale.shape[0]) / scale.norm(dim=-1)
        return ScaleLayer(Variable(scale[None, :], trainable=False))


# ------ Mixup and Label smoothing ------

class PLREmbeddingsFactory(Fitter, FitterFactory):
    # an implementation of https://github.com/yandex-research/tabular-dl-num-embeddings
    def __init__(self, plr_sigma: float = 1.0, plr_hidden_1: int = 8, plr_hidden_2: int = 8,
                 plr_lr_factor: float = 1.0, plr_lr_factor_1: float = 1.0, plr_lr_factor_2: float = 1.0,
                 plr_wd_factor: float = 1.0, plr_act_name: str = 'relu',
                 plr_use_densenet: bool = False, plr_use_cos_bias: bool = False, **config):
        super().__init__(needs_tensors=False, is_individual=True, modified_tensors=['x_cont'])
        self.plr_sigma = plr_sigma
        self.plr_hidden_1 = plr_hidden_1
        self.plr_hidden_2 = plr_hidden_2
        self.plr_lr_factor = plr_lr_factor
        self.plr_lr_factor_1 = plr_lr_factor_1
        self.plr_lr_factor_2 = plr_lr_factor_2
        self.plr_wd_factor = plr_wd_factor
        self.plr_act_name = plr_act_name
        self.plr_use_densenet = plr_use_densenet
        self.plr_use_cos_bias = plr_use_cos_bias
        if not plr_use_cos_bias:
            assert plr_hidden_1 % 2 == 0

    def get_n_params(self, tensor_infos: Dict[str, TensorInfo]) -> int:
        n_cont = self._get_n_values(tensor_infos, ['x_cont'])
        hidden_2 = self.plr_hidden_2
        if self.plr_use_densenet:
            hidden_2 -= 1  # don't count densenet output for parameters
        if self.plr_use_cos_bias:
            return n_cont * (2 * self.plr_hidden_1 + (self.plr_hidden_1 + 1) * hidden_2)
        else:
            return n_cont * (self.plr_hidden_1 // 2 + (self.plr_hidden_1 + 1) * hidden_2)

    def get_n_forward(self, tensor_infos: Dict[str, TensorInfo]) -> int:
        hidden_2 = self.plr_hidden_2
        if self.plr_use_densenet:
            # for before the torch.cat() and after the torch.cat()
            hidden_2 = 2 * hidden_2 - 1
        if self.plr_act_name != 'linear':
            hidden_2 += self.plr_hidden_2
        n_cont = self._get_n_values(tensor_infos, ['x_cont'])
        if self.plr_use_cos_bias:
            # 3 for wx, wx+b, cos(wx+b)
            return n_cont * (3 * self.plr_hidden_1 + hidden_2)
        else:
            # in first hidden layer, have wx, sin(wx), cos(wx), cat(...)
            return n_cont * (int(2.5 * self.plr_hidden_1) + hidden_2)

    def forward_tensor_infos(self, tensor_infos: Dict[str, TensorInfo]) -> Dict[str, TensorInfo]:
        return utils.update_dict(tensor_infos,
                                 {'x_cont': TensorInfo(
                                     feat_shape=[tensor_infos['x_cont'].get_n_features() * self.plr_hidden_2])})

    def _fit(self, ds: DictDataset) -> Layer:
        n_cont = ds.tensor_infos['x_cont'].get_n_features()  # assuming that the shape is rank 1
        hyper_factors_1 = {'lr': self.plr_lr_factor * self.plr_lr_factor_1, 'wd': self.plr_wd_factor}
        hyper_factors_2 = {'lr': self.plr_lr_factor * self.plr_lr_factor_2, 'wd': self.plr_wd_factor}

        if self.plr_use_cos_bias:
            with sub_scope_context('weight_1'):
                weight_1 = Variable(self.plr_sigma * torch.randn(n_cont, 1, self.plr_hidden_1, device=ds.device),
                                    hyper_factors=hyper_factors_1)
            with sub_scope_context('bias_1'):
                # use uniform [-pi, pi] instead of uniform [0, 2pi] for smaller values in case of weight decay
                bias_1 = Variable(np.pi * (-1 + 2 * torch.rand(n_cont, 1, self.plr_hidden_1, device=ds.device)),
                                  hyper_factors=hyper_factors_1)
        else:
            # normal initialization as in the paper
            with sub_scope_context('weight_1'):
                weight_1 = Variable(self.plr_sigma * torch.randn(n_cont, 1, self.plr_hidden_1 // 2, device=ds.device),
                                    hyper_factors=hyper_factors_1)

        # kaiming init from nn.Linear
        in_features = self.plr_hidden_1
        hidden_2 = self.plr_hidden_2
        if self.plr_use_densenet:
            hidden_2 -= 1
        with sub_scope_context('weight_2'):
            weight_2 = Variable(
                (-1 + 2 * torch.rand(n_cont, self.plr_hidden_1, hidden_2, device=ds.device))
                / np.sqrt(in_features),
                hyper_factors=hyper_factors_2)
        with sub_scope_context('bias_2'):
            bias_2 = Variable(
                (-1 + 2 * torch.rand(n_cont, 1, hidden_2, device=ds.device)) / np.sqrt(in_features),
                hyper_factors=hyper_factors_2)

        if self.plr_use_cos_bias:
            return PLREmbeddingsLayerCosBias(fitter=self, weight_1=weight_1, weight_2=weight_2, bias_1=bias_1,
                                             bias_2=bias_2, plr_act_name=self.plr_act_name,
                                             plr_use_densenet=self.plr_use_densenet)
        else:
            return PLREmbeddingsLayer(fitter=self, weight_1=weight_1, weight_2=weight_2, bias_2=bias_2,
                                      plr_act_name=self.plr_act_name, plr_use_densenet=self.plr_use_densenet)


class PLREmbeddingsLayer(Layer):
    # an implementation of https://github.com/yandex-research/tabular-dl-num-embeddings
    # see https://github.com/yandex-research/rtdl-num-embeddings/tree/main/package
    def __init__(self, fitter: Fitter, weight_1: Variable, weight_2: Variable, bias_2: Variable, plr_act_name: str,
                 plr_use_densenet: bool = False):
        super().__init__(fitter=fitter)
        self.weight_1 = weight_1
        self.weight_2 = weight_2
        self.bias_2 = bias_2
        self.plr_act_name = plr_act_name
        self.plr_use_densenet = plr_use_densenet

    def forward_cont(self, x):
        # transpose to treat the continuous feature dimension like a batched dimension
        # then add a new channel dimension
        # shape will be (vectorized..., n_cont, batch, 1)
        x_orig = x
        x = x.transpose(-1, -2).unsqueeze(-1)
        x = 2 * torch.pi * x.matmul(self.weight_1)  # matmul is automatically batched
        x = torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
        x = x.matmul(self.weight_2)  # matmul is automatically batched
        x = x + self.bias_2
        if self.plr_act_name == 'relu':
            x = torch.relu(x)
        elif self.plr_act_name == 'linear':
            pass
        else:
            raise ValueError(f'Unknown plr_act_name "{self.plr_act_name}"')
        # bring back n_cont dimension after n_batch
        # then flatten the last two dimensions
        x = x.transpose(-2, -3)
        x = x.reshape(*x.shape[:-2], x.shape[-2] * x.shape[-1])
        if self.plr_use_densenet:
            x = torch.cat([x, x_orig], dim=-1)
        return x

    def _stack(self, layers):
        return PLREmbeddingsLayer(fitter=layers[0].fitter,
                                  weight_1=Variable.stack([l.weight_1 for l in layers]),
                                  weight_2=Variable.stack([l.weight_2 for l in layers]),
                                  bias_2=Variable.stack([l.bias_2 for l in layers]),
                                  plr_act_name=layers[0].plr_act_name,
                                  plr_use_densenet=layers[0].plr_use_densenet)


class PLREmbeddingsLayerCosBias(Layer):
    # an implementation of https://github.com/yandex-research/tabular-dl-num-embeddings
    # see https://github.com/yandex-research/rtdl-num-embeddings/tree/main/package
    def __init__(self, fitter: Fitter, weight_1: Variable, bias_1: Variable,
                 weight_2: Variable, bias_2: Variable, plr_act_name: str,
                 plr_use_densenet: bool = False):
        super().__init__(fitter=fitter)
        self.weight_1 = weight_1
        self.weight_2 = weight_2
        self.bias_1 = bias_1
        self.bias_2 = bias_2
        self.plr_act_name = plr_act_name
        self.plr_use_densenet = plr_use_densenet

    def forward_cont(self, x):
        # transpose to treat the continuous feature dimension like a batched dimension
        # then add a new channel dimension
        # shape will be (vectorized..., n_cont, batch, 1)
        x_orig = x
        x = x.transpose(-1, -2).unsqueeze(-1)
        x = 2 * torch.pi * x.matmul(self.weight_1)  # matmul is automatically batched
        x = x + self.bias_1
        # x = torch.sin(x)
        x = torch.cos(x)
        x = x.matmul(self.weight_2)  # matmul is automatically batched
        x = x + self.bias_2
        if self.plr_act_name == 'relu':
            x = torch.relu(x)
        elif self.plr_act_name == 'linear':
            pass
        else:
            raise ValueError(f'Unknown plr_act_name "{self.plr_act_name}"')
        # bring back n_cont dimension after n_batch
        # then flatten the last two dimensions
        x = x.transpose(-2, -3)
        x = x.reshape(*x.shape[:-2], x.shape[-2] * x.shape[-1])
        if self.plr_use_densenet:
            x = torch.cat([x, x_orig], dim=-1)
        return x

    def _stack(self, layers):
        return PLREmbeddingsLayerCosBias(fitter=layers[0].fitter,
                                         weight_1=Variable.stack([l.weight_1 for l in layers]),
                                         weight_2=Variable.stack([l.weight_2 for l in layers]),
                                         bias_1=Variable.stack([l.bias_1 for l in layers]),
                                         bias_2=Variable.stack([l.bias_2 for l in layers]),
                                         plr_act_name=layers[0].plr_act_name,
                                         plr_use_densenet=layers[0].plr_use_densenet)


class PeriodicEmbeddingsFactory(Fitter, FitterFactory):
    # an implementation of https://github.com/yandex-research/tabular-dl-num-embeddings
    def __init__(self, periodic_emb_sigma: float = 1.0, periodic_emb_dim: int = 8,
                 periodic_emb_lr_factor: float = 1.0, periodic_emb_wd_factor: float = 1.0,
                 periodic_emb_only_cos: bool = False, periodic_emb_densenet: bool = False, **config):
        super().__init__(needs_tensors=False, is_individual=True, modified_tensors=['x_cont'])
        self.periodic_emb_sigma = periodic_emb_sigma
        self.periodic_emb_dim = periodic_emb_dim
        self.periodic_emb_lr_factor = periodic_emb_lr_factor
        self.periodic_emb_wd_factor = periodic_emb_wd_factor
        self.periodic_emb_only_cos = periodic_emb_only_cos
        self.periodic_emb_densenet = periodic_emb_densenet

    def get_n_params(self, tensor_infos: Dict[str, TensorInfo]) -> int:
        n_params_single = self.periodic_emb_dim
        if self.periodic_emb_densenet:
            n_params_single -= 1
        if self.periodic_emb_only_cos:
            n_params_single *= 2
        else:
            n_params_single //= 2
        return self._get_n_values(tensor_infos, ['x_cont']) * n_params_single

    def get_n_forward(self, tensor_infos: Dict[str, TensorInfo]) -> int:
        n_cont = self._get_n_values(tensor_infos, ['x_cont'])
        # factor 2 * for sin, cos, x, and concat
        return 2 * n_cont * self.periodic_emb_dim

    def forward_tensor_infos(self, tensor_infos: Dict[str, TensorInfo]) -> Dict[str, TensorInfo]:
        return utils.update_dict(tensor_infos,
                                 {'x_cont': TensorInfo(
                                     feat_shape=[tensor_infos['x_cont'].get_n_features() * self.periodic_emb_dim])})

    def _fit(self, ds: DictDataset) -> Layer:
        n_cont = ds.tensor_infos['x_cont'].get_n_features()  # assuming that the shape is rank 1
        hyper_factors = {'lr': self.periodic_emb_lr_factor, 'wd': self.periodic_emb_wd_factor}

        param_dim = self.periodic_emb_dim
        if self.periodic_emb_densenet:
            param_dim -= 1
        if self.periodic_emb_only_cos:
            # not implemented because it turned out to be not so good to omit the linear layer afterward
            raise NotImplementedError()
        else:
            if param_dim % 2 == 1:
                raise ValueError(f'Wrong parity for periodic_emb_dim, got {self.periodic_emb_dim=}')
            param_dim //= 2

            with sub_scope_context('weight'):
                weight = Variable(
                    self.periodic_emb_sigma * torch.randn(n_cont, 1, param_dim, device=ds.device),
                    hyper_factors=hyper_factors)

            return PeriodicEmbeddingsLayerSinCos(self, weight, periodic_emb_densenet=self.periodic_emb_densenet)


class PeriodicEmbeddingsLayerSinCos(Layer):
    # an implementation of https://github.com/yandex-research/tabular-dl-num-embeddings
    # see https://github.com/yandex-research/rtdl-num-embeddings/tree/main/package
    def __init__(self, fitter: Fitter, weight: Variable, periodic_emb_densenet: bool):
        super().__init__(fitter=fitter)
        self.weight = weight
        self.periodic_emb_densenet = periodic_emb_densenet

    def forward_cont(self, x):
        # transpose to treat the continuous feature dimension like a batched dimension
        # then add a new channel dimension
        # shape will be (vectorized..., n_cont, batch, 1)
        x_orig = x
        x = x.transpose(-1, -2).unsqueeze(-1)
        x = 2 * torch.pi * x.matmul(self.weight)  # matmul is automatically batched
        x = torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
        # bring back n_cont dimension after n_batch
        # then flatten the last two dimensions
        x = x.transpose(-2, -3)
        x = x.reshape(*x.shape[:-2], x.shape[-2] * x.shape[-1])
        if self.periodic_emb_densenet:
            x = torch.cat([x, x_orig], dim=-1)
        return x

    def _stack(self, layers):
        return PeriodicEmbeddingsLayerSinCos(fitter=layers[0].fitter,
                                             weight=Variable.stack([l.weight for l in layers]),
                                             periodic_emb_densenet=layers[0].periodic_emb_densenet)


class ToSoftLabelLayer(Layer):
    def __init__(self, y_tensor_info, fitter: Fitter):
        super().__init__(fitter=fitter)
        self.y_tensor_info = y_tensor_info

    def forward_tensors(self, tensors):
        if 'y' not in tensors:
            return tensors
        else:
            y = tensors['y']
            y_cs = self.y_tensor_info.get_cat_sizes().numpy()
            new_y_cols = []
            for i, cs in enumerate(y_cs):
                if cs == 0:
                    # already continuous
                    new_y_cols.append(y[[slice(None)] * (y.dim() - 1) + [slice(i, i + 1)]])
                else:
                    # make continuous
                    # todo: is there a better one-hot function without the long -> float conversion?
                    new_y_cols.append(F.one_hot(y[[slice(None)] * (y.dim() - 1) + [i]], num_classes=cs).float())
            return utils.join_dicts(tensors, {'y': torch.cat(new_y_cols, dim=-1)})


class ToSoftLabelFitter(Fitter):
    def __init__(self):
        super().__init__(needs_tensors=False, is_individual=False, modified_tensors=['y'])

    def forward_tensor_infos(self, tensor_infos):
        if 'y' not in tensor_infos:
            return tensor_infos

        new_y_shape = sum([max(1, cs) for cs in tensor_infos['y'].get_cat_sizes().numpy()])
        return utils.update_dict(tensor_infos, {'y': TensorInfo(feat_shape=[new_y_shape])})

    def _fit(self, ds: DictDataset) -> Layer:
        return ToSoftLabelLayer(y_tensor_info=ds.tensor_infos['y'], fitter=self)


class LabelSmoothingLayer(Layer):
    # assumes soft labels as inputs
    def __init__(self, ls_dist: Variable):
        super().__init__()
        self.hyper_getter = self.context.hp_manager.register_hyper('ls_eps', self.context.scope)
        self.ls_dist = ls_dist

    def forward_tensors(self, tensors):
        # print(f'{self.training=}, {list(tensors.keys())=}')
        # if not self.training or 'y' not in tensors:
        if 'y' not in tensors:
            return tensors

        ls_eps = self.hyper_getter()
        # print(f'{ls_eps=:g}')
        y = tensors['y']
        y = (1.0 - ls_eps) * y + ls_eps * self.ls_dist
        return utils.update_dict(tensors, {'y': y})

    def _stack(self, layers):
        return LabelSmoothingLayer(Variable.stack([l.ls_dist for l in layers]))


class LabelSmoothingFitter(Fitter):
    def __init__(self, use_ls_prior=False, **config):
        # todo: we set needs_tensors=True and is_individual=True here
        #  because the transformation can depend on the hyperparameter ls_eps, which can be scheduled.
        #  If needs_tensors=True, this fitter is not fitted for one-time preprocessing,
        #  where the schedules are not yet available.
        #  ideally, super().__init__() would use another parameter is_dynamic or so which could be set to true instead
        # formerly, we used needs_tensors=use_ls_prior
        super().__init__(needs_tensors=True, is_individual=True, modified_tensors=['y'])
        self.use_ls_prior = use_ls_prior

    def _fit(self, ds: DictDataset) -> Layer:
        # consistency check since y must be soft labels and not hard labels
        assert ds.tensor_infos['y'].is_cont()
        # y is assumed to already be converted to one-hot
        if self.use_ls_prior:
            y = ds.tensors['y']
            ls_dist = y.mean(dim=-2, keepdim=True)
        else:
            n_classes = ds.tensor_infos['y'].get_n_features()
            ls_dist = torch.ones(1, n_classes, device=ds.device) / n_classes
        return LabelSmoothingLayer(Variable(ls_dist, trainable=False))


class LabelSmoothingFactory(FitterFactory):
    def __init__(self, **config):
        super().__init__()
        self.config = config

    def _create(self, tensor_infos) -> Fitter:
        if tensor_infos['y'].get_cat_sizes()[0].item() > 0:
            # labels are still in categorical form
            return SequentialFitter([ToSoftLabelFitter(), LabelSmoothingFitter(**self.config)])

        return LabelSmoothingFitter(**self.config)


class StochasticLabelNoiseLayer(Layer):
    def __init__(self):
        super().__init__()
        self.sigma_getter = self.context.hp_manager.register_hyper('sln_sigma', self.context.scope)

    def forward_tensors(self, tensors):
        if 'y' not in tensors:
            return tensors

        y = tensors['y']
        return utils.join_dicts(tensors, {'y': y + self.sigma_getter() * torch.randn_like(y)})


class StochasticLabelNoiseFitter(Fitter):
    def __init__(self):
        super().__init__(needs_tensors=False, is_individual=False, modified_tensors=['x_cont'])

    def _fit(self, ds: DictDataset) -> Layer:
        # todo: could do a consistency check since y must be soft labels and not hard labels
        return StochasticLabelNoiseLayer()


class StochasticLabelNoiseFactory(FitterFactory):
    def _create(self, tensor_infos) -> Fitter:
        if tensor_infos['y'].get_cat_sizes()[0].item() > 0:
            # labels are still in categorical form
            return SequentialFitter([ToSoftLabelFitter(), StochasticLabelNoiseFitter()])

        return StochasticLabelNoiseFitter()


# implementing "Feature Selection using Stochastic Gates"
class StochasticGateLayer(Layer):
    def __init__(self, mu: Variable):
        super().__init__()
        self.sigma_getter = self.context.hp_manager.register_hyper('sg_sigma', self.context.scope)
        self.lambda_getter = self.context.hp_manager.register_hyper('sg_lambda', self.context.scope)
        self.mu = mu

    def forward_cont(self, x):
        mu = self.mu
        if self.training:
            sigma = self.sigma_getter()
            mu = mu + sigma * torch.randn_like(x)
            reg = gauss_cdf(self.mu / sigma).mean(dim=-1).mean(dim=-1).sum()
            self.context.hp_manager.add_reg_term(self.lambda_getter() * reg)
        z = mu.clamp(0.0, 1.0)
        # z = z / (z.mean(dim=-1, keepdim=True) + 1e-8)
        return x * z

    def _stack(self, layers):
        return StochasticGateLayer(Variable.stack([l.mu for l in layers]))


class StochasticGateFactory(Fitter, FitterFactory):
    def __init__(self):
        super().__init__(needs_tensors=False, modified_tensors=['x_cont'])

    def get_n_params(self, tensor_infos: Dict[str, TensorInfo]) -> int:
        return self._get_n_values(tensor_infos, ['x_cont'])

    def get_n_forward(self, tensor_infos: Dict[str, TensorInfo]) -> int:
        # rough upper bound
        return 15 * self._get_n_values(tensor_infos, ['x_cont'])

    def _fit(self, ds: DictDataset) -> Layer:
        # see https://github.com/runopti/stg/blob/9f630968c4f14cff6da4e54421c497f24ac1e08e/python/stg/layers.py#L10
        n_cont = ds.tensor_infos['x_cont'].get_n_features()
        return StochasticGateLayer(Variable(0.5 * torch.ones(1, n_cont, device=ds.device), hyper_factors={'wd': 0.0}))


class AntisymmetricInitializationFactory(FitterFactory):
    def __init__(self, factory, **config):
        super().__init__()
        self.factory = factory
        self.config = config

    def _create(self, tensor_infos) -> Fitter:
        fitter = self.factory.create(tensor_infos)
        # return AntisymmetricInitializationFitter(fitter, **self.config)

        # only duplicate the part of the fitter that is actually learnable
        common, individual = fitter.split_off_individual()
        return SequentialFitter([common, AntisymmetricInitializationFitter(individual, **self.config)])


class AntisymmetricInitializationFitter(Fitter):
    """
    Implements the antisymmetric initialization trick from http://proceedings.mlr.press/v107/zhang20a/zhang20a.pdf
    """

    def __init__(self, fitter: Fitter, **config):
        super().__init__(needs_tensors=fitter.needs_tensors, is_individual=fitter.is_individual,
                         scope_names=fitter.scope_names, modified_tensors=fitter.modified_tensors)
        self.fitter = fitter
        self.asi_factor = config.get('asi_factor', 1 / np.sqrt(2))

    def forward_tensor_infos(self, tensor_infos: Dict[str, TensorInfo]):
        return self.fitter.forward_tensor_infos(tensor_infos)

    def get_n_params(self, tensor_infos: Dict[str, TensorInfo]):
        return 2 * self.fitter.get_n_params(tensor_infos)

    def get_n_forward(self, tensor_infos: Dict[str, TensorInfo]):
        return 2 * self.fitter.get_n_forward(tensor_infos)  # maybe not entirely accurate but almost

    def _fit(self, ds: DictDataset) -> Layer:
        tfm1 = self.fitter.fit(ds)
        tfm2 = self.fitter.fit(ds)
        with torch.no_grad():
            for p1, p2 in zip(tfm1.parameters(), tfm2.parameters()):
                p2.data = p1.data
            for b1, b2 in zip(tfm1.buffers(), tfm2.buffers()):
                b2.data = b1.data
        # multiply by 1/sqrt(2) at the end to preserve the learning speed for SGD,
        # however, would need to multiply by 0.5 for adam
        return SequentialLayer([SubtractionLayer(tfm1, tfm2),
                                FunctionLayer(lambda x, a=self.asi_factor: a * x)])

    def __str__(self):
        sub_strings = ['  ' + line for line in str(self.fitter).split('\n')]
        return f'{self.__class__.__name__} (\n' + '\n'.join(sub_strings) + '\n)\n'


class SubtractionLayer(Layer):
    def __init__(self, layer1: Layer, layer2: Layer):
        super().__init__()
        self.layer1 = layer1
        self.layer2 = layer2

    def forward_tensor_infos(self, tensor_infos):
        return utils.join_dicts(self.layer1.forward_tensor_infos(tensor_infos),
                                self.layer2.forward_tensor_infos(tensor_infos))

    def forward_tensors(self, tensors):
        out1 = self.layer1.forward_tensors(tensors)
        out2 = self.layer2.forward_tensors(tensors)
        if 'x_cont' not in out2:
            return utils.join_dicts(out1, out2)

        return utils.join_dicts(out1, out2, {'x_cont': out1['x_cont'] - out2['x_cont']})

    def _stack(self, layers):
        return SubtractionLayer(layers[0].layer1.stack([l.layer1 for l in layers]),
                                layers[0].layer2.stack([l.layer2 for l in layers]))
