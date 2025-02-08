import optunahub
import torch
from sklearn.metrics import balanced_accuracy_score, r2_score

from experiment.scikit_talent.talent_classifier import DeepClassifier
from experiment.scikit_talent.talent_regressor import DeepRegressor
from model.utils import merge_sampled_parameters, sample_parameters, get_method


def tune_hyper_parameters(
    self: "DeepClassifier",
    opt_space,
    x_train,
    y_train,
    x_val,
    y_val,
    categorical_indicator,
):
    """
    Tune hyper-parameters.

    :args: argparse.Namespace, arguments
    :opt_space: dict, search space
    :train_val_data: tuple, training and validation data
    :info: dict, information about the dataset
    :return: argparse.Namespace, arguments
    """
    import optuna.trial

    is_regression = isinstance(self, DeepRegressor)

    list_variables = set()
    for k, v in opt_space[self.model_type]["model"].items():
        if isinstance(v, list) and v[0] == "categorical" and isinstance(v[1][0], list):
            values = v[1]
            values = [str(e) for e in values]
            opt_space[self.model_type]["model"][k][1] = values
            list_variables.add(k)

    def objective(trial):
        config = {}
        try:
            opt_space[self.model_type]["training"]["n_bins"] = ["int", 2, 256]
        except:
            opt_space[self.model_type]["fit"]["n_bins"] = ["int", 2, 256]
        merge_sampled_parameters(
            config, sample_parameters(trial, opt_space[self.model_type], config)
        )
        for k in config["model"]:
            if k in list_variables:
                config["model"][k] = eval(config["model"][k])

        if self.model_type == "xgboost" and torch.cuda.is_available():
            config["model"]["tree_method"] = "gpu_hist"
            config["model"]["gpu_id"] = self.gpu
            config["fit"]["verbose"] = False
        elif self.model_type == "catboost" and torch.cuda.is_available():
            config["fit"]["logging_level"] = "Silent"

        elif self.model_type == "RandomForest":
            config["model"]["max_depth"] = 12

        if self.model_type in ["resnet"]:
            config["model"]["activation"] = "relu"
            config["model"]["normalization"] = "batchnorm"

        if self.model_type in ["ftt"]:
            config["model"].setdefault("prenormalization", False)
            config["model"].setdefault("initialization", "xavier")
            config["model"].setdefault("activation", "reglu")
            config["model"].setdefault("n_heads", 8)
            config["model"].setdefault("d_token", 64)
            config["model"].setdefault("token_bias", True)
            config["model"].setdefault("kv_compression", None)
            config["model"].setdefault("kv_compression_sharing", None)

        if self.model_type in ["excelformer"]:
            config["model"].setdefault("prenormalization", False)
            config["model"].setdefault("kv_compression", None)
            config["model"].setdefault("kv_compression_sharing", None)
            config["model"].setdefault("token_bias", True)
            config["model"].setdefault("init_scale", 0.01)
            config["model"].setdefault("n_heads", 8)

        if self.model_type in ["node"]:
            config["model"].setdefault("choice_function", "sparsemax")
            config["model"].setdefault("bin_function", "sparsemoid")

        if self.model_type in ["tabr"]:
            config["model"]["num_embeddings"].setdefault("type", "PLREmbeddings")
            config["model"]["num_embeddings"].setdefault("lite", True)
            config["model"].setdefault("d_multiplier", 2.0)
            config["model"].setdefault("mixer_normalization", "auto")
            config["model"].setdefault("dropout1", 0.0)
            config["model"].setdefault("normalization", "LayerNorm")
            config["model"].setdefault("activation", "ReLU")

        if self.model_type in ["mlp_plr"]:
            config["model"]["num_embeddings"].setdefault("type", "PLREmbeddings")
            config["model"]["num_embeddings"].setdefault("lite", True)

        if self.model_type in ["ptarl"]:
            config["model"]["n_clusters"] = 20
            config["model"]["regularize"] = "True"
            config["general"]["diversity"] = "True"
            config["general"]["ot_weight"] = 0.25
            config["general"]["diversity_weight"] = 0.25
            config["general"]["r_weight"] = 0.25

        if self.model_type in ["modernNCA", "tabm"]:
            config["model"]["num_embeddings"].setdefault("type", "PLREmbeddings")
            config["model"]["num_embeddings"].setdefault("lite", True)

        if self.model_type in ["tabm"]:
            config["model"]["backbone"].setdefault("type", "MLP")
            config["model"].setdefault("arch_type", "tabm")
            config["model"].setdefault("k", 32)

        if self.model_type in ["danets"]:
            config["general"]["k"] = 5
            config["general"]["virtual_batch_size"] = 256

        if self.model_type in ["dcn2"]:
            config["model"]["stacked"] = False

        if self.model_type in ["grownet"]:
            config["ensemble_model"]["lr"] = 1.0
            config["model"]["sparse"] = False
            config["training"]["lr_scaler"] = 3

        if self.model_type in ["autoint"]:
            config["model"].setdefault("prenormalization", False)
            config["model"].setdefault("initialization", "xavier")
            config["model"].setdefault("activation", "relu")
            config["model"].setdefault("n_heads", 8)
            config["model"].setdefault("d_token", 64)
            config["model"].setdefault("kv_compression", None)
            config["model"].setdefault("kv_compression_sharing", None)

        if self.model_type in ["protogate"]:
            config["training"].setdefault("lam", 1e-3)
            config["training"].setdefault("pred_coef", 1)
            config["training"].setdefault("sorting_tau", 16)
            config["training"].setdefault("feature_selection", True)
            config["model"].setdefault("a", 1)
            config["model"].setdefault("sigma", 0.5)

        if self.model_type in ["grande"]:
            config["model"].setdefault("from_logits", True)
            config["model"].setdefault("use_class_weights", True)
            config["model"].setdefault("bootstrap", False)

        if self.model_type in ["amformer"]:
            config["model"].setdefault("heads", 8)
            config["model"].setdefault("groups", [54, 54, 54, 54])
            config["model"].setdefault("sum_num_per_group", [32, 16, 8, 4])
            config["model"].setdefault("prod_num_per_group", [6, 6, 6, 6])
            config["model"].setdefault("cluster", True)
            config["model"].setdefault("target_mode", "mix")
            config["model"].setdefault("token_descent", False)

        if config.get("config_type") == "trv4":
            if config["model"]["activation"].endswith("glu"):
                # This adjustment is needed to keep the number of parameters roughly in the
                # same range as for non-glu activations
                config["model"]["d_ffn_factor"] *= 2 / 3

        trial_configs.append(config)
        # method.fit(train_val_data, info, train=True, config=config)
        # run with this config
        try:
            self.config = config
            self.fit(x_train, y_train, categorical_indicator=categorical_indicator)
            if is_regression:
                val_pred = self.predict(x_val)
                result = r2_score(y_val, val_pred)
            else:
                val_pred = self.predict(x_val)
                result = balanced_accuracy_score(y_val, val_pred)
            return result
        except Exception as e:
            print(e)
            # worst solution (R2/Accuracy)
            return -1e9

    # get data property
    if is_regression:
        direction = "maximize"
        for key in opt_space[self.model_type]["model"].keys():
            if (
                "dropout" in key
                and "?" not in opt_space[self.model_type]["model"][key][0]
            ):
                opt_space[self.model_type]["model"][key][0] = (
                    "?" + opt_space[self.model_type]["model"][key][0]
                )
                opt_space[self.model_type]["model"][key].insert(1, 0.0)
    else:
        direction = "maximize"

    method = get_method(self.model_type)(self, is_regression)

    trial_configs = []
    # Load HEBO Sampler from OptunaHub
    module = optunahub.load_module("samplers/hebo")
    sampler = module.HEBOSampler()
    study = optuna.create_study(
        direction=direction,
        sampler=sampler,
    )
    study.optimize(
        objective,
        **{"n_trials": self.n_trials},
    )
    # get best configs
    best_trial_id = study.best_trial.number
    # update config files
    print("Best Hyper-Parameters")
    print(trial_configs[best_trial_id])
    # update config
    self.config = trial_configs[best_trial_id]

    return self