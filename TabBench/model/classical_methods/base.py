import abc
import numpy as np
import sklearn.metrics as skm
from model.utils import (
    set_seeds,
    get_device
)
# from sklearn.externals import joblib
from model.lib.data import (
    Dataset,
    data_nan_process,
    data_enc_process,
    num_enc_process,
    data_norm_process,
    data_label_process,
)

def check_softmax(logits):
    # Check if any values are outside the [0, 1] range and Ensure they sum to 1
    if np.any((logits < 0) | (logits > 1)) or (not np.allclose(logits.sum(axis=-1), 1, atol=1e-5)):
        exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # stabilize by subtracting max
        return exps / np.sum(exps, axis=1, keepdims=True)
    else:
        return logits
    
class classical_methods(object, metaclass=abc.ABCMeta):
    def __init__(self, args, is_regression):
        self.args = args
        print(args.config)
        self.is_regression = is_regression
        self.D = None
        self.args.device = get_device()
        self.trlog = {}
        assert args.cat_policy != 'indices'

    def data_format(self, is_train = True, N = None, C = None, y = None):
        if is_train:
            self.N, self.C, self.num_new_value, self.imputer, self.cat_new_value = data_nan_process(self.N, self.C, self.args.num_nan_policy, self.args.cat_nan_policy)
            self.y, self.y_info, self.label_encoder = data_label_process(self.y, self.is_regression)
            self.n_bins = self.args.config['fit']['n_bins']
            self.N,self.num_encoder = num_enc_process(self.N,num_policy = self.args.num_policy, n_bins = self.n_bins,y_train=self.y['train'],is_regression=self.is_regression)
            self.N, self.C, self.ord_encoder, self.mode_values, self.cat_encoder = data_enc_process(self.N, self.C, self.args.cat_policy, self.y['train'])
            self.N, self.normalizer = data_norm_process(self.N, self.args.normalization, self.args.seed)
            
            if self.is_regression:
                self.d_out = 1
            else:
                self.d_out = len(np.unique(self.y['train']))
            self.n_num_features = self.N['train'].shape[1] if self.N is not None else 0
            self.n_cat_features = self.C['train'].shape[1] if self.C is not None else 0
            self.d_in = 0 if self.N is None else self.N['train'].shape[1]
        else:
            N_test, C_test, _, _, _ = data_nan_process(N, C, self.args.num_nan_policy, self.args.cat_nan_policy, self.num_new_value, self.imputer, self.cat_new_value)
            y_test, _, _ = data_label_process(y, self.is_regression, self.y_info, self.label_encoder)
            N_test,_ = num_enc_process(N_test,num_policy=self.args.num_policy,n_bins = self.n_bins,y_train=None,encoder = self.num_encoder)
            N_test, C_test, _, _, _ = data_enc_process(N_test, C_test, self.args.cat_policy, None, self.ord_encoder, self.mode_values, self.cat_encoder)
            N_test, _ = data_norm_process(N_test, self.args.normalization, self.args.seed, self.normalizer)
            if N_test is not None and C_test is not None:
                self.N_test,self.C_test = N_test['test'],C_test['test']
            elif N_test is None and C_test is not None:
                self.N_test,self.C_test = None,C_test['test']
            else:
                self.N_test,self.C_test = N_test['test'],None
            self.y_test = y_test['test']
            
    def construct_model(self, model_config = None):
        raise NotImplementedError

    def fit(self, data, info, train = True, config = None):
        N, C, y = data
        # if self.D is None:
        self.D = Dataset(N, C, y, info)
        self.N, self.C, self.y = self.D.N, self.D.C, self.D.y
        self.is_binclass, self.is_multiclass, self.is_regression = self.D.is_binclass, self.D.is_multiclass, self.D.is_regression
          
        if config is not None:
            self.reset_stats_withconfig(config)
        self.data_format(is_train = True)
        self.construct_model()

        # if not train, skip the training process. such as load the checkpoint and directly predict the results
        if not train:
            return
        
    def reset_stats_withconfig(self, config):
        set_seeds(self.args.seed)
        self.config = self.args.config = config

    def metric(self, predictions, labels, y_info):
        if not isinstance(labels, np.ndarray):
            labels = labels.cpu().numpy()
        if not isinstance(predictions, np.ndarray):
            predictions = predictions.cpu().numpy()
        if self.is_regression:
            mae = skm.mean_absolute_error(labels, predictions)
            rmse = skm.mean_squared_error(labels, predictions) ** 0.5
            r2 = skm.r2_score(labels, predictions)
            if y_info['policy'] == 'mean_std':
                mae *= y_info['std']
                rmse *= y_info['std']
            return (mae,r2,rmse), ("MAE", "R2", "RMSE")
        elif self.is_binclass:
            # if not softmax, convert to probabilities
            predictions = check_softmax(predictions)
            accuracy = skm.accuracy_score(labels, predictions.argmax(axis=-1))
            avg_recall = skm.balanced_accuracy_score(labels, predictions.argmax(axis=-1))
            avg_precision = skm.precision_score(labels, predictions.argmax(axis=-1), average='macro')
            f1_score = skm.f1_score(labels, predictions.argmax(axis=-1), average='binary')
            log_loss = skm.log_loss(labels, predictions)
            auc = skm.roc_auc_score(labels, predictions[:, 1])
            return (accuracy, avg_recall, avg_precision, f1_score, log_loss, auc), ("Accuracy", "Avg_Recall", "Avg_Precision", "F1", "LogLoss", "AUC")
        elif self.is_multiclass:
            # if not softmax, convert to probabilities
            predictions = check_softmax(predictions)
            accuracy = skm.accuracy_score(labels, predictions.argmax(axis=-1))
            avg_recall = skm.balanced_accuracy_score(labels, predictions.argmax(axis=-1))
            avg_precision = skm.precision_score(labels, predictions.argmax(axis=-1), average='macro')
            f1_score = skm.f1_score(labels, predictions.argmax(axis=-1), average='macro')
            log_loss = skm.log_loss(labels, predictions)
            auc = skm.roc_auc_score(labels, predictions, average='macro', multi_class='ovr')
            return (accuracy, avg_recall, avg_precision, f1_score, log_loss, auc), ("Accuracy", "Avg_Recall", "Avg_Precision", "F1", "LogLoss", "AUC")
        else:
            raise ValueError("Unknown tabular task type")
        
