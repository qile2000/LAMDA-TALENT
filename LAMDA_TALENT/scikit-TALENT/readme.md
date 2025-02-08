# Scikit-TALENT

> We would like to extend our thanks to [hengzhe-zhang](https://github.com/hengzhe-zhang) for providing the initial scikit-learn interface code.

🚀 **Scikit-TALENT** 🚀 is a scikit-compatible wrapper that enables the use of state-of-the-art deep learning models for
tabular data with minimal effort and no learning curve.

## ✨ Features

- Seamlessly integrates with scikit-learn
- Supports deep learning-based classifiers for tabular data
- Enables quick experimentation with modern deep learning models
- Built-in hyperparameter tuning using Optuna

## 🛠️ Usage

### Basic Example

```python
import numpy as np
import openml
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from scikit_talent.talent_classifier import DeepClassifier

# Define the deep learning model type
model = "modernNCA"
e = DeepClassifier(model_type=model)

# Load dataset from OpenML
dataset = openml.datasets.get_dataset(3, download_data=True)
X, y, categorical_indicator, _ = dataset.get_data(
    target=dataset.default_target_attribute, dataset_format="dataframe"
)
X, y = np.array(X), np.array(y)

# Split dataset
dataset_size = 100  # Define training size
X_train_pre, X_test, y_train_pre, y_test = train_test_split(
    X, y, train_size=dataset_size, random_state=0, shuffle=True, stratify=y
)

# Train and evaluate the model
e.fit(X_train_pre, y_train_pre, categorical_indicator)
predictions = e.predict(X_test)
score = balanced_accuracy_score(y_test, predictions)
print(f"{model}: Balanced Accuracy Score = {score}")
```

## 🎯 Hyperparameter Tuning

Hyperparameter tuning is supported via Optuna, allowing automatic optimization of model parameters.

Example usage (refer to `optuna_demo.py` for details):

```python
tune_hyper_parameters(
    model,  # Model instance
    opt_space,  # Hyperparameter search space
    x_train_sub,  # Training subset
    y_train_sub,  # Training labels
    x_val,  # Validation subset
    y_val,  # Validation labels
    categorical_indicator,  # Categorical feature indicators
)
```