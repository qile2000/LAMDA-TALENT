====================================
How to Use TALENT
====================================

This guide will walk you through how to set up and use the TALENT toolbox for benchmarking models on tabular data, running experiments, and adding new methods.

==========================
1. Cloning the Repository
==========================

To get started, clone the TALENT repository from GitHub:

.. code-block:: bash

   git clone https://github.com/qile2000/LAMDA-TALENT


.. code-block:: bash

   cd LAMDA-TALENT/LAMDA-TALENT


Make sure you have the required dependencies installed. Refer to the `dependencies` section for more details on how to install them.

==========================
2. Running Experiments
==========================

TALENT supports running experiments for both deep learning methods and classical machine learning models. You can easily configure and run experiments by following these steps:

1. **Configure the experiment settings**:
   
   - Edit the configuration files located in `configs/default/[MODEL_NAME].json` and `configs/opt_space/[MODEL_NAME].json` to customize global settings and hyperparameters for the model you wish to train.

2. **Run the experiment**:
   
   To run an experiment for deep learning methods, use the following command:

   .. code-block:: bash

      python train_model_deep.py --model_type [MODEL_NAME]

   
   For classical machine learning methods, use:

   .. code-block:: bash

      python train_model_classical.py --model_type [MODEL_NAME]

   Replace `[MODEL_NAME]` with the name of the model you wish to run (e.g., `MLP`, `ResNet`, `XGBoost`, etc.).

==========================
3. Adding New Methods
==========================

TALENT is designed to be easily extendable. You can add new models by following these steps:


1. **Create the model**: 
   
   - Add the model class to the `model/models/` directory. You can use one of the existing models as a template.

2. **Override the base class**:

   - Inherit from the base class located at `model/methods/base.py`, and override the `construct_model()` method in the new class to define the architecture of your model.

3. **Register the method**:

   - Add the method name to the `get_method` function in `model/utils.py`.

4. **Update configuration files**:

   - Add the parameter settings for your new method in `configs/default/[MODEL_NAME].json` and `configs/opt_space/[MODEL_NAME].json`.


==========================
4. Configuring Hyperparameters
==========================

TALENT allows you to fine-tune models through configuration files. These files are located in the `configs/default/` and `configs/opt_space/` directories.

- **`configs/default/`**: Contains global settings and default parameters for each method.
- **`configs/opt_space/`**: Defines the hyperparameter optimization space for each method.

To modify the hyperparameters:

1. Open the appropriate `.json` configuration file.
2. Edit the values for parameters such as learning rate, batch size, number of layers, etc.
3. Save the changes and run the experiment again using the `train_model_deep.py` or `train_model_classical.py` script.


You can customize the logging behavior by modifying the configuration files.

==========================
5. Troubleshooting
==========================

If you encounter any issues while using TALENT, try the following steps:

1. **Check the logs**: Review the logs in the `logs/` directory for any error messages.
2. **Verify dependencies**: Ensure that all required dependencies are installed. Refer to the `dependencies.rst` for more information.
3. **Configuration issues**: Double-check your configuration files to ensure the paths, dataset names, and hyperparameters are correct.
4. **Contact**: If you're unable to resolve the issue, feel free to open an issue on GitHub or contact the developers.

==========================
Conclusion
==========================

TALENT provides a flexible and powerful platform for experimenting with both classical and deep learning models on tabular data. By following the steps in this guide, you can quickly set up and run experiments, fine-tune models, and even add your own methods to the toolbox. For any further assistance, refer to the documentation or reach out to the development team.
