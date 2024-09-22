====================================
Dependencies
====================================

TALENT relies on the following dependencies to provide a comprehensive machine learning toolbox for tabular data analysis. Ensure that these packages are installed in your environment before using TALENT:

==========================
Python Libraries
==========================

1. `PyTorch <https://pytorch.org/>`_ : The deep learning framework used for model development and training in TALENT.
2. `scikit-learn <https://scikit-learn.org/stable/>`_ : Provides classical machine learning models and utilities for data preprocessing and evaluation.
3. `pandas <https://pandas.pydata.org/>`_ : A data manipulation and analysis library for handling tabular data.
4. `numpy <https://numpy.org/>`_ : Fundamental package for scientific computing with Python, including support for large, multi-dimensional arrays and matrices.
5. `scipy <https://scipy.org/>`_ : A library for scientific and technical computing, used for optimization, integration, and statistics.
6. `tqdm <https://tqdm.github.io/>`_ : A library for creating progress bars, used to display the progress of model training and data processing.

==========================
Optional Dependencies
==========================

Some methods in TALENT require additional dependencies for specific tasks. If you intend to use the following methods, make sure to install these optional packages:

- `faiss-gpu <https://github.com/facebookresearch/faiss>`_ : Required for **TabR** to efficiently handle nearest neighbor searches. Install via conda:
   
   .. code-block:: bash

      conda install faiss-gpu -c pytorch
   

==========================
Installation
==========================

To install the necessary dependencies for TALENT, you can use the following commands:

1. **Using pip**:

   Install the required libraries from the `requirements.txt` file:
   
   .. code-block:: bash
   
      pip install -r requirements.txt
   

2. **Using conda**:

   If you are using `conda`, you can create a new environment and install the dependencies:
   
   .. code-block:: bash
   
      conda create -n talent python=3.10
      conda activate talent
      pip install -r requirements.txt
   

==========================
Additional Notes
==========================

Ensure that your Python version is compatible with the dependencies listed above. TALENT is tested with **Python 3.10** and **PyTorch 2.0.1**. 