import numpy as np
import random

import torch
from torch.utils.checkpoint import checkpoint

import pickle
import io
import os
import pathlib
from pathlib import Path
from functools import partial

from sklearn.preprocessing import PowerTransformer, QuantileTransformer, RobustScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils import column_or_1d

from model.lib.tabpfn.utils import CustomUnpickler
from model.lib.tabpfn.utils import load_model_workflow, get_params_from_config, transformer_predict
# Source: https://github.com/automl/TabPFN


class TabPFNClassifier(BaseEstimator, ClassifierMixin):

    models_in_memory = {}

    def __init__(self, device='cpu', base_path=pathlib.Path(__file__).parent.resolve(), model_string='',
                 N_ensemble_configurations=3, no_preprocess_mode=False, multiclass_decoder='permutation',
                 feature_shift_decoder=True, only_inference=True, seed=0, no_grad=True, batch_size_inference=32,
                 subsample_features=False):
        """
        Initializes the classifier and loads the model. 
        Depending on the arguments, the model is either loaded from memory, from a file, or downloaded from the 
        repository if no model is found.
        
        Can also be used to compute gradients with respect to the inputs X_train and X_test. Therefore no_grad has to be 
        set to False and no_preprocessing_mode must be True. Furthermore, X_train and X_test need to be given as 
        torch.Tensors and their requires_grad parameter must be set to True.
        
        
        :param device: If the model should run on cuda or cpu.
        :param base_path: Base path of the directory, from which the folders like models_diff can be accessed.
        :param model_string: Name of the model. Used first to check if the model is already in memory, and if not, 
               tries to load a model with that name from the models_diff directory. It looks for files named as 
               follows: "prior_diff_real_checkpoint" + model_string + "_n_0_epoch_e.cpkt", where e can be a number 
               between 100 and 0, and is checked in a descending order. 
        :param N_ensemble_configurations: The number of ensemble configurations used for the prediction. Thereby the 
               accuracy, but also the running time, increases with this number. 
        :param no_preprocess_mode: Specifies whether preprocessing is to be performed.
        :param multiclass_decoder: If set to permutation, randomly shifts the classes for each ensemble configuration. 
        :param feature_shift_decoder: If set to true shifts the features for each ensemble configuration according to a 
               random permutation.
        :param only_inference: Indicates if the model should be loaded to only restore inference capabilities or also 
               training capabilities. Note that the training capabilities are currently not being fully restored.
        :param seed: Seed that is used for the prediction. Allows for a deterministic behavior of the predictions.
        :param batch_size_inference: This parameter is a trade-off between performance and memory consumption.
               The computation done with different values for batch_size_inference is the same,
               but it is split into smaller/larger batches.
        :param no_grad: If set to false, allows for the computation of gradients with respect to X_train and X_test. 
               For this to correctly function no_preprocessing_mode must be set to true.
        :param subsample_features: If set to true and the number of features in the dataset exceeds self.max_features (100),
                the features are subsampled to self.max_features.
        """

        # Model file specification (Model name, Epoch)
        i = 0
        model_key = model_string+'|'+str(device)
        if model_key in self.models_in_memory:
            model, c, results_file = self.models_in_memory[model_key]
        else:
            model, c, results_file = load_model_workflow(i, 42, add_name=model_string, base_path=base_path, device=device,
                                                         eval_addition='', only_inference=only_inference)
            self.models_in_memory[model_key] = (model, c, results_file)
            if len(self.models_in_memory) == 2:
                print('Multiple models in memory. This might lead to memory issues. Consider calling remove_models_from_memory()')
        #style, temperature = self.load_result_minimal(style_file, i, e)

        self.device = device
        self.model = model
        self.c = c
        self.style = None
        self.temperature = None
        self.N_ensemble_configurations = N_ensemble_configurations
        self.base__path = base_path
        self.base_path = base_path
        self.i = i
        self.model_string = model_string

        self.max_num_features = self.c['num_features']
        self.max_num_classes = self.c['max_num_classes']
        self.differentiable_hps_as_style = self.c['differentiable_hps_as_style']

        self.no_preprocess_mode = no_preprocess_mode
        self.feature_shift_decoder = feature_shift_decoder
        self.multiclass_decoder = multiclass_decoder
        self.only_inference = only_inference
        self.seed = seed
        self.no_grad = no_grad
        self.subsample_features = subsample_features

        assert self.no_preprocess_mode if not self.no_grad else True, \
            "If no_grad is false, no_preprocess_mode must be true, because otherwise no gradient can be computed."

        self.batch_size_inference = batch_size_inference

    def remove_models_from_memory(self):
        self.models_in_memory = {}

    def load_result_minimal(self, path, i, e):
        with open(path, 'rb') as output:
            _, _, _, style, temperature, optimization_route = CustomUnpickler(output).load()

            return style, temperature

    def _validate_targets(self, y):
        y_ = column_or_1d(y, warn=True)
        check_classification_targets(y)
        cls, y = np.unique(y_, return_inverse=True)
        if len(cls) < 2:
            raise ValueError(
                "The number of classes has to be greater than one; got %d class"
                % len(cls)
            )

        self.classes_ = cls

        return np.asarray(y, dtype=np.float64, order="C")

    def fit(self, X, y, overwrite_warning=False):
        """
        Validates the training set and stores it.

        If clf.no_grad (default is True):
        X, y should be of type np.array
        else:
        X should be of type torch.Tensors (y can be np.array or torch.Tensor)
        """
        if self.no_grad:
            # Check that X and y have correct shape
            X, y = check_X_y(X, y, force_all_finite=False)
        # Store the classes seen during fit
        y = self._validate_targets(y)
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y)

        self.X_ = X
        self.y_ = y

        if (X.shape[1] > self.max_num_features):
            if self.subsample_features:
                print('WARNING: The number of features for this classifier is restricted to ', self.max_num_features, ' and will be subsampled.')
            else:
                raise ValueError("The number of features for this classifier is restricted to ", self.max_num_features)
        if len(np.unique(y)) > self.max_num_classes:
            raise ValueError("The number of classes for this classifier is restricted to ", self.max_num_classes)
        if X.shape[0] > 1024 and not overwrite_warning:
            raise ValueError("⚠️ WARNING: TabPFN is not made for datasets with a trainingsize > 1024. Prediction might take a while, be less reliable. We advise not to run datasets > 10k samples, which might lead to your machine crashing (due to quadratic memory scaling of TabPFN). Please confirm you want to run by passing overwrite_warning=True to the fit function.")


        # Return the classifier
        return self

    def predict_proba(self, X, normalize_with_test=False, return_logits=False):
        """
        Predict the probabilities for the input X depending on the training set previously passed in the method fit.

        If no_grad is true in the classifier the function takes X as a numpy.ndarray. If no_grad is false X must be a
        torch tensor and is not fully checked.
        """
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        if self.no_grad:
            X = check_array(X, force_all_finite=False)
            X_full = np.concatenate([self.X_, X], axis=0)
            X_full = torch.tensor(X_full, device=self.device).float().unsqueeze(1)
        else:
            assert (torch.is_tensor(self.X_) & torch.is_tensor(X)), "If no_grad is false, this function expects X as " \
                                                                    "a tensor to calculate a gradient"
            X_full = torch.cat((self.X_, X), dim=0).float().unsqueeze(1).to(self.device)

            if int(torch.isnan(X_full).sum()):
                print('X contains nans and the gradient implementation is not designed to handel nans.')

        y_full = np.concatenate([self.y_, np.zeros(shape=X.shape[0])], axis=0)
        y_full = torch.tensor(y_full, device=self.device).float().unsqueeze(1)

        eval_pos = self.X_.shape[0]

        prediction = transformer_predict(self.model[2], X_full, y_full, eval_pos,
                                         device=self.device,
                                         style=self.style,
                                         inference_mode=True,
                                         preprocess_transform='none' if self.no_preprocess_mode else 'mix',
                                         normalize_with_test=normalize_with_test,
                                         N_ensemble_configurations=self.N_ensemble_configurations,
                                         softmax_temperature=self.temperature,
                                         multiclass_decoder=self.multiclass_decoder,
                                         feature_shift_decoder=self.feature_shift_decoder,
                                         differentiable_hps_as_style=self.differentiable_hps_as_style,
                                         seed=self.seed,
                                         return_logits=return_logits,
                                         no_grad=self.no_grad,
                                         batch_size_inference=self.batch_size_inference,
                                         **get_params_from_config(self.c))
        prediction_, y_ = prediction.squeeze(0), y_full.squeeze(1).long()[eval_pos:]

        return prediction_.detach().cpu().numpy() if self.no_grad else prediction_

    def predict(self, X, return_winning_probability=False, normalize_with_test=False):
        p = self.predict_proba(X, normalize_with_test=normalize_with_test)
        y = np.argmax(p, axis=-1)
        y = self.classes_.take(np.asarray(y, dtype=np.intp))
        if return_winning_probability:
            return y, p.max(axis=-1)
        return y
    