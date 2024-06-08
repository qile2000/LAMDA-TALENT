**TabBench** is a benchmark with a comprehensive toolkit for tabular data learning. It offers the following advantages:

1. **Diverse Methods**: Includes various classical methods, tree-based methods, and the latest popular deep learning methods.
2. **Extensive Dataset Collection**: Equipped with 300 datasets, covering a wide range of task types, size distributions, and dataset domains.
3. **Customizability**: Easily allows the addition of datasets and methods.
4. **Versatile Support**: Supports diverse normalization, encoding, and metrics.

**Future Updates**: We will soon update with extensive dataset experiment results and analysis tool codes that aid research in tabular data learning.

# How to Place Datasets

Datasets are placed in the project's upper-level directory, corresponding to the file name specified by `args.dataset_path`. For instance, if the project is `TabularBenchmark`, the data should be placed in `TabularBenchmark/../args.dataset_path/args.dataset`.

Each dataset folder `args.dataset` consists of:

- Numeric features: `N_train/val/test.npy` (can be omitted if there are no numeric features)
- Categorical features: `C_train/val/test.npy` (can be omitted if there are no categorical features)
- Labels: `y_train/val/test.npy`
- `info.json`, which must include the following three contents (`task_type` can be "regression", "multiclass" or "binclass"):

  ```json
  {
    "task_type": "regression", 
    "n_num_features": 10,
    "n_cat_features": 10
  }
  ```

# How to Run Methods

Examples can be found in `example_cls.sh` and `example_reg.sh`. Other args adjustments refer to `train_model_deep.py/train_model_classical.py`'s `get_args()`.

There are various:

- **Encodings:** See the `data_enc_process` function in `model/lib/data.py`.
- **Normalizations:** See the `data_norm_process` function in `model/lib/data.py`.
- **Metrics:** See the `metric` function in `model/methods/base.py`. Running any method and dataset will calculate all metrics. The metric used for early stopping is accuracy/rmse.

# How to Add New Methods

For methods like the MLP class that only need to design the model, you only need to:

- Inherit from `model/methods/base.py` and override the `construct_model()` method in the new class.
- Add the model class in `model/models`.
- Add the method name in the `modeltype_to_method` function in `model/utils.py`.
- Add the parameter settings for the new method in `default_para.json` and `opt_space.json`.

For other methods that require changing the training process, partially override functions based on `model/methods/base.py`. For details, refer to the implementation of other methods in `model/methods/`.