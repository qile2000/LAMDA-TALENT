from typing import Optional, List

import numpy as np
import openml
from sklearn.base import RegressorMixin
from sklearn.model_selection import train_test_split

from data_loader import (
    split_train_val,
    convert_test,
    generate_info,
)
from talent_classifier import DeepClassifier, classical_models
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.utils import get_method, set_gpu, set_seeds, mkdir


class DeepRegressor(DeepClassifier, RegressorMixin):
    def __init__(
        self,
        dataset: Optional[str] = None,
        model_type: Optional[str] = None,
        max_epoch: Optional[int] = None,
        batch_size: Optional[int] = None,
        normalization: Optional[str] = None,
        num_nan_policy: Optional[str] = None,
        cat_nan_policy: Optional[str] = None,
        cat_policy: Optional[str] = None,
        num_policy: Optional[str] = None,
        n_bins: Optional[int] = None,
        cat_min_frequency: Optional[float] = None,
        n_trials: Optional[int] = None,
        seed_num: Optional[int] = None,
        workers: Optional[int] = None,
        gpu: Optional[int] = None,
        tune: bool = False,
        retune: bool = False,
        evaluate_option: Optional[str] = None,
        dataset_path: Optional[str] = None,
        model_path: Optional[str] = None,
        talent_path: str = "..",
    ):
        """
        Initialize the DeepRegressorEstimator with given parameters.
        Parameters not provided will be loaded from configuration files.
        """
        super().__init__(
            dataset=dataset,
            model_type=model_type,
            max_epoch=max_epoch,
            batch_size=batch_size,
            normalization=normalization,
            num_nan_policy=num_nan_policy,
            cat_nan_policy=cat_nan_policy,
            cat_policy=cat_policy,
            num_policy=num_policy,
            n_bins=n_bins,
            cat_min_frequency=cat_min_frequency,
            n_trials=n_trials,
            seed_num=seed_num,
            workers=workers,
            gpu=gpu,
            tune=tune,
            retune=retune,
            evaluate_option=evaluate_option,
            dataset_path=dataset_path,
            model_path=model_path,
            talent_path=talent_path,
        )

    def fit(self, X, y, categorical_indicator: List[bool]):
        """
        Fit the deep learning regression model.
        Parameters:
            X: Features (pre-split training data)
            y: Continuous target variable
            categorical_indicator: List indicating which features are categorical
        Returns:
            self
        """
        # Split the training data into train and validation
        self.categorical_indicator = categorical_indicator
        train_val_data = split_train_val(
            X_train=X,
            y_train=y,
            categorical_features=categorical_indicator,
            task_type="regression",
            val_size=0.2,
            random_state=0,
        )

        # Generate info
        info = generate_info(
            categorical_features=categorical_indicator, task_type="regression"
        )
        self.info = info

        # Retrieve the appropriate method based on model_type
        method = get_method(self.model_type)(self, is_regression=True)

        # Fit the model using the training data
        time_cost = method.fit(train_val_data, info, train=True)
        self.method = method

        return self

    def predict(self, X):
        """
        Make predictions with the trained regression model.
        Parameters:
            X: Features (pre-split test data)
            categorical_indicator: List indicating which features are categorical
        Returns:
            Predictions as a 1D NumPy array
        """
        # Convert test data
        test_data = convert_test(
            X_test=X, categorical_features=self.categorical_indicator
        )

        # Make predictions
        if self.model_type in classical_models:
            _, _, predict_logits = self.method.predict(
                test_data, self.info, model_name=self.evaluate_option
            )
        else:
            _, _, _, predict_logits = self.method.predict(
                test_data, self.info, model_name=self.evaluate_option
            )
        return predict_logits.flatten()


if __name__ == "__main__":
    results_list, time_list = [], []
    for model in [
        "modernNCA",  # ICLR 2025
        "xgboost",  # KDD 2014
    ]:
        e = DeepRegressor(model_type=model)
        dataset = openml.datasets.get_dataset(
            549,
            download_data=True,
            download_qualities=True,
            download_features_meta_data=True,
        )
        qualities = dataset.qualities
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            target=dataset.default_target_attribute, dataset_format="dataframe"
        )
        X, y = np.array(X), np.array(y)
        X_train_pre, X_test, y_train_pre, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0, shuffle=True
        )
        e.fit(X_train_pre, y_train_pre, categorical_indicator)
        print(e.predict(X_test))