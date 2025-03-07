import pathlib
from typing import Optional, Any, Union, List, Dict

import numpy as np

# Implementation of "Better by Default: Strong Pre-Tuned MLPs and Boosted Trees on Tabular Data"
# Source: https://github.com/dholzmueller/pytabkit/

from TALENT.model.lib.realmlp.sklearn.default_params import DefaultParams
from TALENT.model.lib.realmlp.sklearn.sklearn_base import AlgInterfaceRegressor, AlgInterfaceClassifier
from TALENT.model.lib.realmlp.alg_interfaces.alg_interfaces import AlgInterface
from TALENT.model.lib.realmlp.alg_interfaces.nn_interfaces import NNAlgInterface

class RealMLPConstructorMixin:
    def __init__(self, device: Optional[str] = None, random_state: Optional[Union[int, np.random.RandomState]] = None,
                 n_cv: int = 1, n_refit: int = 0, val_fraction: float = 0.2, n_threads: Optional[int] = None,
                 tmp_folder: Optional[Union[str, pathlib.Path]] = None, verbosity: int = 0,
                 train_metric_name: Optional[str] = None, val_metric_name: Optional[str] = None,
                 n_epochs: Optional[int] = None,
                 batch_size: Optional[int] = None, predict_batch_size: Optional[int] = None,
                 hidden_sizes: Optional[List[int]] = None,
                 tfms: Optional[List[str]] = None,
                 num_emb_type: Optional[str] = None,
                 use_plr_embeddings: Optional[bool] = None, plr_sigma: Optional[float] = None,
                 plr_hidden_1: Optional[int] = None, plr_hidden_2: Optional[int] = None,
                 plr_act_name: Optional[str] = None, plr_use_densenet: Optional[bool] = None,
                 plr_use_cos_bias: Optional[bool] = None, plr_lr_factor: Optional[float] = None,
                 max_one_hot_cat_size: Optional[int] = None, embedding_size: Optional[int] = None,
                 act: Optional[str] = None,
                 use_parametric_act: Optional[bool] = None, act_lr_factor: Optional[float] = None,
                 weight_param: Optional[str] = None, weight_init_mode: Optional[str] = None,
                 weight_init_gain: Optional[str] = None,
                 weight_lr_factor: Optional[float] = None,
                 bias_init_mode: Optional[str] = None, bias_lr_factor: Optional[float] = None,
                 bias_wd_factor: Optional[float] = None,
                 add_front_scale: Optional[bool] = None,
                 scale_lr_factor: Optional[float] = None,
                 block_str: Optional[str] = None,
                 first_layer_config: Optional[Dict[str, Any]] = None,
                 last_layer_config: Optional[Dict[str, Any]] = None,
                 middle_layer_config: Optional[Dict[str, Any]] = None,
                 p_drop: Optional[float] = None, p_drop_sched: Optional[str] = None,
                 wd: Optional[float] = None, wd_sched: Optional[str] = None,
                 opt: Optional[str] = None,
                 lr: Optional[Union[float, Dict[str, float]]] = None, lr_sched: Optional[str] = None,
                 mom: Optional[float] = None, mom_sched: Optional[str] = None,
                 sq_mom: Optional[float] = None, sq_mom_sched: Optional[str] = None,
                 opt_eps: Optional[float] = None, opt_eps_sched: Optional[str] = None,
                 normalize_output: Optional[bool] = None, clamp_output: Optional[bool] = None,
                 use_ls: Optional[bool] = None, ls_eps: Optional[float] = None, ls_eps_sched: Optional[float] = None,
                 use_early_stopping: Optional[bool] = None,
                 early_stopping_additive_patience: Optional[int] = None,
                 early_stopping_multiplicative_patience: Optional[float] = None,
                 ):
        """
        Constructor for RealMLP, using the default parameters from RealMLP-TD.
        For lists of default parameters, we refer to sklearn.default_params.DefaultParams.
        RealMLP-TD does automatic preprocessing,
        so no manual preprocessing is necessary except for imputing missing numerical values.

        Tips for modifications:

        * For faster training: For large datasets (say >50K samples), especially on GPUs, increase batch_size.
          It can also help to decrease n_epochs, set use_plr_embeddings=False (in case of many numerical features),
          increase max_one_hot_cat_size (in case of large-cardinality categories), or set use_parametric_act=False
        * For more accuracy: You can try increasing n_epochs or hidden_sizes while also decreasing lr.
        * For classification, if you care about metrics like cross-entropy or AUC instead of accuracy,
          we recommend setting val_metric_name='cross_entropy' and use_ls=False.

        :param device: PyTorch device name like 'cpu', 'cuda', 'cuda:0', 'mps' (default=None).
            If None, 'cuda' will be used if available, otherwise 'cpu'.
        :param random_state: Random state to use for random number generation
            (splitting, initialization, batch shuffling). If None, the behavior is not deterministic.
        :param n_cv: Number of cross-validation splits to use (default=1).
            If validation set indices are given in fit(), `n_cv` models will be fitted using different random seeds.
            Otherwise, `n_cv`-fold cross-validation will be used (stratified for classification). If `n_refit=0` is set,
            the prediction will use the average of the models fitted during cross-validation.
            (Averaging is over probabilities for classification, and over outputs for regression.)
            Otherwise, refitted models will be used.
        :param n_refit: Number of models that should be refitted on the training+validation dataset (default=0).
            If zero, only the models from the cross-validation stage are used.
            If positive, `n_refit` models will be fitted on the training+validation dataset (all data given in fit())
            and their predictions will be averaged during predict().
        :param val_fraction: Fraction of samples used for validation (default=0.2). Has to be in [0, 1).
            Only used if `n_cv==1` and no validation split is provided in fit().
        :param n_threads: Number of threads that the method is allowed to use (default=number of physical cores).
        :param tmp_folder: Temporary folder in which data can be stored during fit().
            (Currently unused for MLP-TD and variants.) If None, methods generally try to not store intermediate data.
        :param verbosity: Verbosity level (default=0, higher means more verbose).
            Set to 2 to see logs from intermediate epochs.
        :param train_metric_name: Name of the training metric
            (default='cross_entropy' for clasification and 'mse' for regression).
            Currently most other metrics are not available for training.
        :param val_metric_name: Name of the validation metric (used for selecting the best epoch).
            Defaults are 'class_error' for classification and 'rmse' for regression.
            Main available classification metrics (all to be minimized): 'class_error', 'cross_entropy', '1-auc_ovo',
            '1-auc_ovr', '1-auc_mu', 'brier', '1-balanced_accuracy', '1-mcc', 'ece'.
            Main available regression metrics: 'rmse', 'mae', 'max_error',
            'pinball(0.95)' (also works with other quantiles specified directly in the string).
            For more metrics, we refer to `models.training.metrics.Metrics.apply()`.
        :param n_epochs: Number of epochs to train the model for (default=256)
        :param batch_size: Batch size to be used for fit(), default=256.
        :param predict_batch_size: Batch size to be used for predict(), default=1024.
        :param hidden_sizes: List of numbers of neurons for each hidden layer, default=[256, 256, 256].
        :param tfms: List of preprocessing transformations,
            default=`['one_hot', 'median_center', 'robust_scale', 'smooth_clip', 'embedding']`.
            Other possible transformations include: 'median_center', 'l2_normalize', 'l1_normalize', 'quantile', 'kdi'.
        :param num_emb_type: Type of numerical embeddings used (default='pbld'). If not set to 'ignore',
            it overrides the parameters `use_plr_embeddings`, `plr_act_name`, `plr_use_densenet`, `plr_use_cos_bias`.
            Possible values: 'ignore', 'none' (no numerical embeddings), 'pl', 'plr', 'pbld', 'pblrd'.
        :param use_plr_embeddings: Whether PLR (or PL) numerical embeddings should be used (default=True).
        :param plr_sigma: Initialization standard deviation for first PLR embedding layer (default=0.1).
        :param plr_hidden_1: (Half of the) number of hidden neurons in the first PLR hidden layer (default=8).
            This number will be doubled since there are sin() and cos() versions for each hidden neuron.
        :param plr_hidden_2: Number of output neurons of the PLR hidden layer,
            excluding the optional densenet connection (default=7).
        :param plr_act_name: Name of PLR activation function (default='linear').
            Use 'relu' for the PLR version and 'linear' for the PL version.
        :param plr_use_densenet: Whether to append the original feature to the numerical embeddings (default=True).
        :param plr_use_cos_bias: Whether to use the cos(wx+b)
            version for the periodic embeddings instead of the (sin(wx), cos(wx)) version (default=True).
        :param plr_lr_factor: Learning rate factor for PLR embeddings (default=0.1).
            Gets multiplied with lr and with the value of the schedule.
        :param max_one_hot_cat_size: Maximum category size that one-hot encoding should be applied to,
            including the category for missing/unknown values (default=9).
        :param embedding_size: Number of output features of categorical embedding layers (default=8).
        :param act: Activation function (default='selu' for classification and 'mish' for regression).
            Can also be 'relu' or 'silu'.
        :param use_parametric_act: Whether to use a parametric activation as described in the paper (default=True).
        :param act_lr_factor: Learning rate factor for parametric activation (default=0.1).
        :param weight_param: Weight parametrization (default='ntk'). See models.nn.WeightFitter() for more options.
        :param weight_init_mode: Weight initialization mode (default='std').
            See models.nn.WeightFitter() for more options.
        :param weight_init_gain: Multiplier for the weight initialization standard deviation.
            (Does not apply to 'std' initialization mode.)
        :param weight_lr_factor: Learning rate factor for weights.
        :param bias_init_mode: Bias initialization mode (default='he+5'). See models.nn.BiasFitter() for more options.
        :param bias_lr_factor: Bias learning rate factor.
        :param bias_wd_factor: Bias weight decay factor.
        :param add_front_scale: Whether to add a scaling layer (diagonal weight matrix)
            before the linear layers (default=True). If set to true and a scaling layer is already configured
            in the block_str, this will create an additional scaling layer.
        :param scale_lr_factor: Scaling layer learning rate factor
            (default=1.0 but will be overridden by default for the first layer in first_layer_config).
        :param block_str: String describing the default hidden layer components.
            The default is 'w-b-a-d' for weight, bias, activation, dropout.
            By default, the last layer config will override it with 'w-b'
            and the first layer config will override it with 's-w-b-a-d', where the 's' stands for the scaling layer.
        :param first_layer_config: Dictionary with more options
            that can override the other options for the construction of the first MLP layer specifically.
            The default is dict(block_str='s-w-b-a-d', scale_lr_factor=6.0),
            using a scaling layer at the beginning of the first layer with lr factor 6.0.
        :param last_layer_config: Dictionary with more options
            that can override the other options for the construction of the last MLP layer specifically.
            The default is an empty dict, in which case the block_str will still be overridden by 'w-b'.
        :param middle_layer_config: Dictionary with more options
            that can override the other options for the construction of the layers except first and last MLP layer.
            The default is an empty dict.
        :param p_drop: Dropout probability (default=0.15). Needs to be in [0, 1).
        :param p_drop_sched: Dropout schedule (default='flat_cos').
        :param wd: Weight decay implemented as in the PyTorch AdamW but works with all optimizers
            (default=0.0 for regression and 1e-2 for classification).
            Weight decay is implemented as
            param -= current_lr_value * current_wd_value * param
            where the current lr and wd values are determined using the base values (lr and wd),
            factors for the given parameter if available, and the respective schedule.
            Note that this is not identical to the original AdamW paper,
            where the lr base value is not included in the update equation.
        :param wd_sched: Weight decay schedule.
        :param opt: Optimizer (default='adam'). See optim.optimizers.get_opt_class().
        :param lr: Learning rate base value (default=0.04 for classification and 0.14 for regression).
        :param lr_sched: Learning rate schedule (default='coslog4'). See training.scheduling.get_schedule().
        :param mom: Momentum parameter, aka :math:`\\beta_1` for Adam (default=0.9).
        :param mom_sched: Momentum schedule (default='constant').
        :param sq_mom: Momentum of squared gradients, aka :math:`\\beta_2` for Adam (default=0.95).
        :param sq_mom_sched: Schedule for sq_mom (default='constant').
        :param opt_eps: Epsilon parameter of the optimizer (default=1e-8 for Adam).
        :param opt_eps_sched: Schedule for opt_eps (default='constant').
        :param normalize_output: Whether to standardize the target for regression (default=True for regression).
        :param clamp_output: Whether to clamp the output for predict() for regression
            to the min/max range seen during training (default=True for regression).
        :param use_ls: Whether to use label smoothing for classification (default=True for classification).
        :param ls_eps: Epsilon parameter for label smoothing (default=0.1 for classification)
        :param ls_eps_sched: Schedule for ls_eps (default='constant').
        :param use_early_stopping: Whether to use early stopping (default=False).
            Note that even without early stopping,
            the best epoch on the validation set is selected if there is a validation set.
            Training is stopped if the epoch exceeds
            early_stopping_multiplicative_patience * best_epoch + early_stopping_additive_patience.
        :param early_stopping_additive_patience: See use_early_stopping (default=20).
        :param early_stopping_multiplicative_patience: See use_early_stopping (default=2).
            We recommend to set it to 1 for monotone learning rate schedules
            but to keep it at 2 for the default schedule.
        """
        super().__init__()  # call the constructor of the other superclass for multiple inheritance
        self.device = device
        self.random_state = random_state
        self.n_cv = n_cv
        self.n_refit = n_refit
        self.val_fraction = val_fraction
        self.n_threads = n_threads
        self.tmp_folder = tmp_folder
        self.verbosity = verbosity
        self.train_metric_name = train_metric_name
        self.val_metric_name = val_metric_name
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.predict_batch_size = predict_batch_size
        self.hidden_sizes = hidden_sizes
        self.tfms = tfms
        self.max_one_hot_cat_size = max_one_hot_cat_size
        self.embedding_size = embedding_size
        self.num_emb_type = num_emb_type
        self.use_plr_embeddings = use_plr_embeddings
        self.plr_sigma = plr_sigma
        self.plr_hidden_1 = plr_hidden_1
        self.plr_hidden_2 = plr_hidden_2
        self.plr_act_name = plr_act_name
        self.plr_use_densenet = plr_use_densenet
        self.plr_use_cos_bias = plr_use_cos_bias
        self.plr_lr_factor = plr_lr_factor
        self.act = act
        self.use_parametric_act = use_parametric_act
        self.act_lr_factor = act_lr_factor
        self.weight_param = weight_param
        self.weight_init_mode = weight_init_mode
        self.weight_init_gain = weight_init_gain
        self.weight_lr_factor = weight_lr_factor
        self.bias_init_mode = bias_init_mode
        self.bias_lr_factor = bias_lr_factor
        self.bias_wd_factor = bias_wd_factor
        self.add_front_scale = add_front_scale
        self.scale_lr_factor = scale_lr_factor
        self.block_str = block_str
        self.first_layer_config = first_layer_config
        self.last_layer_config = last_layer_config
        self.middle_layer_config = middle_layer_config
        self.p_drop = p_drop
        self.p_drop_sched = p_drop_sched
        self.wd = wd
        self.wd_sched = wd_sched
        self.opt = opt
        self.lr = lr
        self.lr_sched = lr_sched
        self.mom = mom
        self.mom_sched = mom_sched
        self.sq_mom = sq_mom
        self.sq_mom_sched = sq_mom_sched
        self.opt_eps = opt_eps
        self.opt_eps_sched = opt_eps_sched
        self.normalize_output = normalize_output
        self.clamp_output = clamp_output
        self.use_ls = use_ls
        self.ls_eps = ls_eps
        self.ls_eps_sched = ls_eps_sched
        self.use_early_stopping = use_early_stopping
        self.early_stopping_additive_patience = early_stopping_additive_patience
        self.early_stopping_multiplicative_patience = early_stopping_multiplicative_patience


class RealMLP_TD_Classifier(RealMLPConstructorMixin, AlgInterfaceClassifier):
    """
    MLP-TD classifier. For constructor parameters, see `MLPConstructorMixin`.
    """

    def _get_default_params(self):
        return DefaultParams.RealMLP_TD_CLASS

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        return NNAlgInterface(**self.get_config())

    def _allowed_device_names(self) -> List[str]:
        return ['cpu', 'cuda', 'mps']


class RealMLP_TD_Regressor(RealMLPConstructorMixin, AlgInterfaceRegressor):
    """
    MLP-TD regressor. For constructor parameters, see `MLPConstructorMixin`.
    """

    def _get_default_params(self):
        return DefaultParams.RealMLP_TD_REG

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        return NNAlgInterface(**self.get_config())

    def _allowed_device_names(self) -> List[str]:
        return ['cpu', 'cuda', 'mps']