import copy
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.preprocessing import QuantileTransformer

from TALENT.model.lib.realmlp.nn_models.activations import ActivationFactory
from TALENT.model.lib.realmlp.nn_models.base import FitterFactory, SequentialFitter, ResidualFitter, Fitter, RenameTensorFactory, FunctionFactory, \
    SequentialFactory, FilterTensorsFactory, ConcatParallelFactory
from TALENT.model.lib.realmlp.nn_models.categorical import EncodingFactory, SingleOneHotFactory, SingleEmbeddingFactory, SingleOrdinalEncodingFactory, \
    SingleTargetEncodingFactory
from TALENT.model.lib.realmlp.nn_models.nn import DropoutFitter, WeightFitter, BiasFitter, ScaleFitter, NoiseFitter, PLREmbeddingsFactory, ScaleFactory, \
    PeriodicEmbeddingsFactory, RFFeatureImportanceFactory, LabelSmoothingFactory, StochasticLabelNoiseFactory, \
    StochasticGateFactory, FeatureImportanceFactory, FixedWeightFactory, AntisymmetricInitializationFactory, \
    NormalizeOutputFactory, ClampOutputFactory
from TALENT.model.lib.realmlp.nn_models.pipeline import MedianCenterFactory, RobustScaleFactory, MeanCenterFactory, GlobalScaleNormalizeFactory, \
    L2NormalizeFactory, L1NormalizeFactory, ThermometerCodingFactory, CircleCodingFactory, SklearnTransformFactory, \
    RobustScaleV2Factory
from TALENT.model.lib.realmlp import utils
from TALENT.model.lib.realmlp.data.data import TensorInfo
from TALENT.model.lib.realmlp.utils import TabrQuantileTransformer


class BlockFactory(FitterFactory):
    def __init__(self, out_features: int, block_str: str = 'w-b-a', **config):
        super().__init__()
        # could also make this a SequentialFactory if there were factories for all the individual fitters
        # or a LambdaFactory
        self.block_str = block_str
        self.out_features = out_features
        self.config = config

    def _create_transform(self, tensor_infos):
        in_features = tensor_infos['x_cont'].get_n_features()
        fitters = []
        for layer_str in self.block_str.split('-'):
            # todo: mixup layer?
            if layer_str in ['a', 'act', 'activation']:
                fitters.append(ActivationFactory(**self.config).create(tensor_infos).add_scope('act'))
            elif layer_str in ['d', 'drop', 'dropout']:
                fitters.append(DropoutFitter())
            elif layer_str in ['w', 'weight']:
                fitters.append(WeightFitter(self.out_features, **self.config).add_scope('weight'))
            elif layer_str in ['b', 'bias']:
                fitters.append(BiasFitter(in_features=in_features, **self.config).add_scope('bias'))
            # elif layer_str == 'D':  # alpha-dropout for self-normalizing neural networks
            #     pass  # todo
            elif layer_str in ['s', 'scale']:
                fitters.append(ScaleFitter(**self.config).add_scope('scale'))
            # elif layer_str == 'n':
            #     pass  # todo: batchnorm
            elif layer_str in ['noise']:
                fitters.append(NoiseFitter(**self.config))
            elif layer_str in ['r', 'res', 'residual']:
                out_tensor_infos = SequentialFitter(fitters).forward_tensor_infos(tensor_infos)
                if np.equal(tensor_infos['X_cont'].get_feat_shape(), out_tensor_infos['X_cont'].get_feat_shape()):
                    # can use residual connection
                    fitters = [ResidualFitter(SequentialFitter(fitters))]
            else:
                raise ValueError(f'BlockFactory: Unknown layer string {layer_str}')
            tensor_infos = fitters[-1].forward_tensor_infos(tensor_infos)
        return SequentialFitter(fitters), tensor_infos


def smooth_clip_func(x):
    return x / (1 + (1 / 9) * x ** 2).sqrt()


def tanh_clip_func(x):
    return 5 * torch.tanh(0.2 * x)


class PreprocessingFactory(FitterFactory):
    def __init__(self, **config):
        super().__init__()
        self.config = config

    def _create(self, tensor_infos: Dict[str, TensorInfo]) -> Fitter:
        tfm_factories = []

        for tfm in self.config.get('tfms', []):
            if tfm == 'one_hot':
                tfm_factories.append(EncodingFactory(SingleOneHotFactory(**self.config), enc_output_name='x_one_hot'))
                tfm_factories.append(RenameTensorFactory(old_name='x_one_hot', new_name='x_cont'))
            elif tfm == 'median_center':
                tfm_factories.append(MedianCenterFactory(**self.config))
            elif tfm == 'robust_scale':
                tfm_factories.append(RobustScaleFactory(**self.config))
            elif tfm == 'smooth_clip':
                tfm_factories.append(FunctionFactory(smooth_clip_func))
            elif tfm == 'tanh_5_clip':
                tfm_factories.append(FunctionFactory(tanh_clip_func))
            elif tfm == 'mean_center':
                tfm_factories.append(MeanCenterFactory(**self.config))
            elif tfm == 'embedding':
                tfm_factories.append(EncodingFactory(SingleEmbeddingFactory(**self.config)).add_scope('emb'))
            elif tfm == 'global_scale_normalize':
                tfm_factories.append(GlobalScaleNormalizeFactory(**self.config))
            elif tfm == 'l2_normalize':
                tfm_factories.append(L2NormalizeFactory(**self.config))
            elif tfm == 'l1_normalize':
                tfm_factories.append(L1NormalizeFactory(**self.config))
            elif tfm == 'thermometer_coding':
                tfm_factories.append(ThermometerCodingFactory(**self.config))
            elif tfm == 'circle_coding':
                tfm_factories.append(CircleCodingFactory(**self.config))
            elif tfm == 'ordinal_encoding':
                tfm_factories.append(EncodingFactory(SingleOrdinalEncodingFactory(**self.config)))
            elif tfm == 'target_encoding':
                tfm_factories.append(EncodingFactory(SingleTargetEncodingFactory(**self.config)))
            elif tfm == 'kdi':
                from kditransform import KDITransformer
                tfm = KDITransformer(alpha=self.config.get('kdi_alpha', 1.0),
                                     output_distribution=self.config.get('kdi_output_distribution', 'normal'))
                tfm_factories.append(SklearnTransformFactory(tfm))
            elif tfm == 'quantile':
                tfm = QuantileTransformer(output_distribution=self.config.get('quantile_output_distribution', 'normal'))
                tfm_factories.append(SklearnTransformFactory(tfm))
            elif tfm == "quantile_tabr":
                tfm = TabrQuantileTransformer()
                tfm_factories.append(SklearnTransformFactory(tfm))

        # old interface, using 'tfms' is preferred
        if self.config.get('use_one_hot', False):
            tfm_factories.append(EncodingFactory(SingleOneHotFactory(**self.config)))
        if self.config.get('use_median_center', False):
            tfm_factories.append(MedianCenterFactory(**self.config))
        if self.config.get('use_robust_scale', False):
            tfm_factories.append(RobustScaleFactory(**self.config))
        if self.config.get('use_robust_scale_v2', False):
            tfm_factories.append(RobustScaleV2Factory(**self.config))
        if self.config.get('use_smooth_clip', False):
            tfm_factories.append(FunctionFactory(lambda x: x / (1 + (1 / 9) * x ** 2).sqrt()))
        if self.config.get('use_mean_center', False):
            tfm_factories.append(MeanCenterFactory(**self.config))
        if self.config.get('use_embedding', False):
            tfm_factories.append(EncodingFactory(SingleEmbeddingFactory(**self.config)).add_scope('emb'))
        if self.config.get('use_global_scale_normalize', False):
            tfm_factories.append(GlobalScaleNormalizeFactory(**self.config))

        return SequentialFactory(tfm_factories).add_scope('tfms').create(tensor_infos=tensor_infos)


class NNFactory(FitterFactory):
    def __init__(self, **config):
        super().__init__()
        self.config = config

        if 'use_embedding' not in config:
            # dirty fix to not miss out on categorical values here,
            # but do no use this as a default in PreprocessingFactory since that is also used for GBDTs
            # that can have native categorical processing capabilities
            self.config['use_embedding'] = True

    def _create_transform(self, tensor_infos: Dict[str, TensorInfo]) -> Tuple[Fitter, Dict[str, TensorInfo]]:
        y_cat_sizes = tensor_infos['y'].get_cat_sizes().numpy()
        n_classes = y_cat_sizes[0]

        factories = []
        net_factories = []

        if 'one_hot' in self.config.get('tfms', []) or self.config.get('use_one_hot', False):
            # do it already here so it can get done once instead of per batch
            factories.append(EncodingFactory(SingleOneHotFactory(**self.config), enc_output_name='x_one_hot'))

        prep_factory = PreprocessingFactory(**self.config)

        num_emb_type = self.config.get('num_emb_type', None)

        num_emb_config = copy.copy(self.config)

        if num_emb_type is None or num_emb_type == 'ignore':
            pass  # don't modify the other configuration parameters
        elif num_emb_type == 'none':
            num_emb_config['use_plr_embeddings'] = False
            num_emb_config['use_periodic_emb'] = False
        elif num_emb_type == 'pl':
            num_emb_config['use_plr_embeddings'] = True
            num_emb_config['plr_use_densenet'] = False
            num_emb_config['plr_use_cos_bias'] = False
            num_emb_config['plr_act_name'] = 'linear'
        elif num_emb_type == 'plr':
            num_emb_config['use_plr_embeddings'] = True
            num_emb_config['plr_use_densenet'] = False
            num_emb_config['plr_use_cos_bias'] = False
            num_emb_config['plr_act_name'] = 'relu'
        elif num_emb_type == 'pbld':
            num_emb_config['use_plr_embeddings'] = True
            num_emb_config['plr_use_densenet'] = True
            num_emb_config['plr_use_cos_bias'] = True
            num_emb_config['plr_act_name'] = 'linear'
        elif num_emb_type == 'pblrd':
            num_emb_config['use_plr_embeddings'] = True
            num_emb_config['plr_use_densenet'] = True
            num_emb_config['plr_use_cos_bias'] = True
            num_emb_config['plr_act_name'] = 'relu'
        else:
            raise ValueError(f'Unknown numerical embedding type: {num_emb_type=}')

        if num_emb_config.get('use_plr_embeddings', False):
            plr_factory = PLREmbeddingsFactory(**num_emb_config).add_scope('plr')
            if num_emb_config.get('use_plr_scale', False):
                plr_factory = SequentialFactory([ScaleFactory(**num_emb_config), plr_factory])
            num_factory = SequentialFactory([
                FilterTensorsFactory(include_keys=['x_cont']),
                prep_factory,
                plr_factory
            ])
            cat_factory = SequentialFactory([
                FilterTensorsFactory(exclude_keys=['x_cont']),
                # EncodingFactory(SingleOneHotFactory(**self.config)),
                prep_factory,
                # EncodingFactory(SingleEmbeddingFactory(**self.config)).add_scope('emb')
            ])
            factories.append(ConcatParallelFactory([num_factory, cat_factory]))
        elif num_emb_config.get('use_periodic_emb', False):
            periodic_emb_factory = PeriodicEmbeddingsFactory(**num_emb_config).add_scope('periodic_emb')
            num_factory = SequentialFactory([
                FilterTensorsFactory(include_keys=['x_cont']),
                prep_factory,
                periodic_emb_factory
            ])
            cat_factory = SequentialFactory([
                FilterTensorsFactory(exclude_keys=['x_cont']),
                # EncodingFactory(SingleOneHotFactory(**self.config)),
                prep_factory,
                # EncodingFactory(SingleEmbeddingFactory(**self.config)).add_scope('emb')
            ])
            factories.append(ConcatParallelFactory([num_factory, cat_factory]))
        else:
            factories.append(prep_factory)

        if self.config.get('use_rf_importances', False):
            factories.append(RFFeatureImportanceFactory())

        if self.config.get('use_ls', False) and n_classes > 0:
            factories.append(LabelSmoothingFactory(**self.config))
        if self.config.get('use_sln', False) and n_classes > 0:
            factories.append(StochasticLabelNoiseFactory())
        if self.config.get('use_sg', False) and n_classes > 0:
            factories.append(StochasticGateFactory())

        if self.config.get('add_importance_layer', False):
            factories.append(FeatureImportanceFactory())
        if self.config.get('add_fixed_weight_layer', False):
            factories.append(FixedWeightFactory())

        out_sizes = self.config.get('hidden_sizes', [256]*3) + [len(y_cat_sizes) if n_classes == 0 else n_classes]
        for i in range(len(out_sizes)):
            layer_position = 'middle'
            block_scope_2 = f'layer-{i}'
            config = self.config
            if i+1 == len(out_sizes):
                config = utils.join_dicts(config, {'block_str': 'w-b'}, config.get('last_layer_config', {}))
                layer_position = 'last'
            elif i == 0:
                config = utils.join_dicts(config, config.get('first_layer_config', {}))
                if config.get('add_front_scale', False):
                    config['block_str'] = 's-' + config.get('block_str', 'w-b-a-d')
                layer_position = 'first'
            block_scope = layer_position + '_layer'
            net_factories.append(
                BlockFactory(out_features=out_sizes[i], layer_position=layer_position,
                             **config).add_scope(block_scope).add_scope(block_scope_2))

        factories.append(SequentialFactory(net_factories).add_scope('net'))

        if self.config.get('use_antisymmetric_initialization', False):
            factories = [AntisymmetricInitializationFactory(SequentialFactory(factories), **self.config)]

        if self.config.get('output_factor', 1.0) != 1.0:
            factories.append(FunctionFactory(lambda x, c=self.config['output_factor']: c*x))
        if self.config.get('normalize_output', False):
            factories.append(NormalizeOutputFactory(**self.config))
        if self.config.get('clamp_output', False):
            # use clamp after normalization!
            factories.append(ClampOutputFactory(**self.config))

        factory = SequentialFactory(factories)

        return factory.create_transform(tensor_infos)
