====================================
LAMDA-TALENT Utils Module
====================================

The **utils** module provides utility functions and helper classes used across the TALENT project. These functions include GPU management, configuration loading, random seed setting, and logging.

.. automodule:: utils
   :members:
   :undoc-members:
   :show-inheritance:

==========================
Classes
==========================

.. autoclass:: Averager
   :members:
   :undoc-members:
   :show-inheritance:

This class helps compute the running average of values, useful for logging and evaluation.

**Methods:**

- **add(x)**: Add a new value `x` to the average.
- **item()**: Return the current average.

.. autoclass:: Timer
   :members:
   :undoc-members:
   :show-inheritance:

This class provides a simple way to measure elapsed time.


==========================
Functions
==========================

.. autofunction:: LAMDA_TALENT.utils.mkdir
   Ensure that the specified directory exists, creating it if necessary.

.. autofunction:: LAMDA_TALENT.utils.set_gpu
   Set the visible GPUs by configuring the `CUDA_VISIBLE_DEVICES` environment variable.

.. autofunction:: LAMDA_TALENT.utils.ensure_path
   Ensure that the specified path exists. If the path already exists and the `remove` flag is set, the path will be deleted and recreated.

.. autofunction:: LAMDA_TALENT.utils.pprint
   A pretty-printing wrapper around Python's `pprint` function for easy logging.

.. autofunction:: LAMDA_TALENT.utils.set_seeds
   Set random seeds for reproducibility across different libraries, including Python's `random`, `numpy`, and PyTorch.

.. autofunction:: LAMDA_TALENT.utils.get_device
   Return the available device (`cuda:0` if available, else `cpu`).

.. autofunction:: LAMDA_TALENT.utils.rmse
   Compute the Root Mean Square Error (RMSE) of predictions.

.. autofunction:: LAMDA_TALENT.utils.load_config
   Load the configuration from a specified JSON file and store it in the `args` object.

.. autofunction:: LAMDA_TALENT.utils.sample_parameters
   Helper function to sample hyperparameters from a search space using a trial object (e.g., from Optuna).

.. autofunction:: LAMDA_TALENT.utils.merge_sampled_parameters
   Merge the sampled parameters into the configuration.

.. autofunction:: LAMDA_TALENT.utils.get_classical_args
   Parse and return arguments for classical machine learning models.

.. autofunction:: LAMDA_TALENT.utils.get_deep_args
   Parse and return arguments for deep learning models.

.. autofunction:: LAMDA_TALENT.utils.show_results_classical
   Display the evaluation results for classical models, including metrics and training time.

.. autofunction:: LAMDA_TALENT.utils.show_results
   Display the evaluation results for deep learning models, including loss, metrics, and training time.

.. autofunction:: LAMDA_TALENT.utils.tune_hyper_parameters
   Tune hyperparameters using a specified search space and objective function (e.g., with Optuna).

.. autofunction:: LAMDA_TALENT.utils.get_method
   Return the appropriate method (model) class based on the provided model name.


