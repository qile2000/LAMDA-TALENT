====================================
LAMDA-TALENT Base Method Module
====================================

This module defines the base method class for implementing different machine learning models within the TALENT framework. It handles the entire training and evaluation pipeline, including data processing, model training, validation, and prediction.

.. automodule:: LAMDA_TALENT.model.methods.base
   :members:
   :undoc-members:
   :show-inheritance:

==========================
Classes
==========================

.. autoclass:: LAMDA_TALENT.model.methods.base.Method
   :members:
   :undoc-members:
   :show-inheritance:

The **Method** class serves as the base class for all models implemented in TALENT. It is designed to handle various tasks, including binary classification, multiclass classification, and regression. 

**Key Methods:**

- **__init__(self, args, is_regression)**: Initializes the method with given arguments and task type (regression or classification).
- **reset_stats_withconfig(self, config)**: Resets training statistics and sets new configuration.
- **data_format(self, is_train=True, N=None, C=None, y=None)**: Formats the data, processes missing values, encodes numerical and categorical features, and normalizes the data.
- **fit(self, data, info, train=True, config=None)**: Trains the model with the provided data and configuration. It handles the entire training loop including validation.
- **predict(self, data, info, model_name)**: Loads a pre-trained model and performs predictions on the given data.
- **train_epoch(self, epoch)**: Executes one training epoch, including forward pass, loss computation, and backpropagation.
- **validate(self, epoch)**: Validates the model on the validation set after each training epoch.
- **metric(self, predictions, labels, y_info)**: Computes various performance metrics based on the task type (e.g., regression, binary classification, multiclass classification).

==========================
Functions
==========================

.. autofunction:: LAMDA_TALENT.model.methods.base.check_softmax
   Checks and converts logits into softmax probabilities if needed.


==========================
Training and Evaluation
==========================

The **Method** class provides a robust pipeline for training and evaluating machine learning models, handling the following steps:

1. **Data Preprocessing**: It processes missing values, encodes categorical features, normalizes numerical features, and prepares data for training.
2. **Training Loop**: The model is trained using the AdamW optimizer with configurable learning rate and weight decay.
3. **Validation**: After each epoch, the model is validated on a separate validation set. Early stopping is employed if validation performance does not improve.
4. **Evaluation**: The model can be evaluated using various metrics depending on the task type (e.g., accuracy, F1 score, R2 score, etc.).

==========================
Performance Metrics
==========================

The **metric** function in the **Method** class computes various metrics based on the task type:

- **For regression**:
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - R2 Score

- **For binary classification**:
  - Accuracy
  - Average Precision
  - F1 Score
  - Log Loss
  - Area Under the Curve (AUC)

- **For multiclass classification**:
  - Accuracy
  - Average Precision
  - F1 Score
  - Log Loss
  - AUC

This ensures that the performance of the models is measured comprehensively across different tasks.
