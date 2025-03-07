====================================
LAMDA-TALENT Data Module
====================================

The **data** module provides functionalities for handling datasets, including data loading, preprocessing, encoding, and normalization. It also provides helper functions for handling missing data and loading datasets from disk.

.. automodule:: TALENT.model.lib.data
   :members:
   :undoc-members:
   :show-inheritance:

==========================
Classes
==========================

.. autoclass:: TALENT.model.lib.Dataset
   :members:
   :undoc-members:
   :show-inheritance:

The **Dataset** class encapsulates the numerical, categorical features, and labels of the dataset, and provides properties to determine the type of task (binary classification, multiclass classification, or regression) and the number of features.

**Properties:**

- **is_binclass**: Returns `True` if the task is binary classification.
- **is_multiclass**: Returns `True` if the task is multiclass classification.
- **is_regression**: Returns `True` if the task is regression.
- **n_num_features**: Number of numerical features.
- **n_cat_features**: Number of categorical features.
- **n_features**: Total number of features (numerical + categorical).
- **size(part)**: Returns the size of a particular part of the dataset (e.g., 'train', 'val', 'test').

==========================
Functions
==========================

.. autofunction:: TALENT.model.lib.raise_unknown
   Raises a ValueError when an unknown value is encountered during data processing.

.. autofunction:: TALENT.model.lib.load_json
   Loads a JSON file from the given path.

.. autofunction:: TALENT.model.lib.dataname_to_numpy
   Converts dataset names to NumPy arrays by loading data files from disk.

.. autofunction:: TALENT.model.lib.get_dataset
   Returns the train/validation/test sets along with the dataset information for the given dataset.

.. autofunction:: TALENT.model.lib.data_nan_process
   Processes missing values (NaN) in both numerical and categorical data according to specified policies.

.. autofunction:: TALENT.model.lib.num_enc_process
   Processes and encodes numerical data based on different encoding policies (e.g., Piecewise Linear Encoding, Unary Encoding).

.. autofunction:: TALENT.model.lib.data_enc_process
   Encodes categorical data using various encoding techniques (e.g., ordinal, one-hot, binary, hashing, target, CatBoost).

.. autofunction:: TALENT.model.lib.data_norm_process
   Normalizes numerical data according to specified normalization techniques (e.g., standard scaling, min-max scaling, quantile transformation).

.. autofunction:: TALENT.model.lib.data_label_process
   Processes the labels (target values) for both regression and classification tasks. Normalizes labels for regression tasks.

.. autofunction:: TALENT.model.lib.data_loader_process
   Converts data into PyTorch tensors and prepares DataLoaders for training, validation, and testing.

.. autofunction:: TALENT.model.lib.to_tensors
   Converts a dictionary of NumPy arrays to a dictionary of PyTorch tensors.

.. autofunction:: TALENT.model.lib.get_categories
   Returns the number of categories for each categorical feature in the dataset.


