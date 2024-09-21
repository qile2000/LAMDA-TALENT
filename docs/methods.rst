====================================
Methods in TALENT
====================================

TALENT integrates an extensive array of 30+ deep learning architectures and classical methods for tabular data. Below is a summary of these methods, organized by type.

==========================
Deep Learning Methods
==========================

TALENT offers the following deep learning models, specifically designed to enhance performance on tabular data:

1. `MLP <https://arxiv.org/abs/2106.11959>`_ : A multi-layer neural network implemented according to RTDL.
2. `ResNet <https://arxiv.org/abs/2106.11959>`_ : A DNN that uses skip connections across many layers.
3. `SNN <https://arxiv.org/abs/1706.02515>`_ : A self-normalizing network that uses the SELU activation, enabling deeper network training.
4. `DANets <https://arxiv.org/abs/2112.02962>`_ : Groups correlated features to enhance tabular data processing while reducing computational complexity.
5. `TabCaps <https://openreview.net/pdf?id=OgbtSLESnI>`_ : A capsule network that encapsulates feature values into vectorial representations.
6. `DCNv2 <https://arxiv.org/abs/2008.13535>`_: Combines an MLP module with a feature crossing module using linear layers and multiplications.
7. `NODE <https://arxiv.org/abs/1909.06312>`_: Generalizes oblivious decision trees, blending gradient-based optimization and hierarchical representation learning.
8. `GrowNet <https://arxiv.org/abs/2002.07971>`_ : A gradient boosting method utilizing shallow neural networks as weak learners.
9. `TabNet <https://arxiv.org/abs/1908.07442>`_ : Sequential attention-based method for tabular data, enhancing feature selection and providing interpretability.
10. `TabR <https://arxiv.org/abs/2307.14338>`_ : A model integrating KNN and attention mechanisms to improve prediction accuracy.
11. `ModernNCA <https://arxiv.org/abs/2407.03257>`_ : Inspired by traditional NCA, this model makes predictions using relationships with neighbors in a learned embedding space.
12. `DNNR <https://arxiv.org/abs/2205.08434>`_ : Enhances KNN using local gradients and Taylor approximations for better predictions.
13. `AutoInt <https://arxiv.org/abs/1810.11921>`_ : Uses multi-head self-attention to automatically learn high-order feature interactions.
14. `Saint <https://arxiv.org/abs/2106.01342>`_ : A token-based model that applies row and column attention mechanisms to tabular data.
15. `TabTransformer <https://arxiv.org/abs/2012.06678>`_ : Enhances tabular data modeling by transforming categorical features into contextual embeddings.
16. `FT-Transformer <https://arxiv.org/abs/2106.11959>`_ : A feature transformation-based method using attention mechanisms on tabular data.
17. `TANGOS <https://openreview.net/pdf?id=n6H86gW8u0d>`_ : A regularization-based method encouraging neuron specialization for tabular data.
18. `SwitchTab <https://arxiv.org/abs/2401.02013>`_ : A self-supervised method improving representation learning through an encoder-decoder framework.
19. `PTaRL <https://openreview.net/pdf?id=G32oY4Vnm8>`_ : Enhances prediction by constructing a prototype-based space for regularization.
20. `TabPFN <https://arxiv.org/abs/2207.01848>`_ : A pre-trained model that generalizes across diverse tabular tasks.
21. `HyperFast <https://arxiv.org/abs/2402.14335>`_ : A meta-trained hypernetwork that generates task-specific neural networks for tabular data.
22. `TabPTM <https://arxiv.org/abs/2311.00055>`_ : Standardizes heterogeneous datasets using meta-representations for tabular data.
23. `BiSHop <https://arxiv.org/abs/2404.03830>`_ : A sparse Hopfield model for tabular learning with column-wise and row-wise modules.
24. `ProtoGate <https://arxiv.org/abs/2306.12330>`_ : A prototype-based model for feature selection in HDLSS biomedical data.
25. `RealMLP <https://arxiv.org/abs/2407.04491>`_ : An improved multilayer perceptron (MLP) with better efficiency.
26. `MLP_PLR <https://arxiv.org/abs/2203.05556>`_ : An enhanced MLP that uses periodic activations to improve performance.
27. `Excelformer <https://arxiv.org/abs/2301.02819>`_ : A model featuring semi-permeable attention modules for tabular data, addressing rotational invariance.
28. `GRANDE <https://arxiv.org/abs/2309.17130>`_ : A tree-mimic model using gradient descent for axis-aligned decision trees.
29. `AMFormer <https://arxiv.org/abs/2402.02334>`_ : A transformer-based method for tabular data, with attention mechanisms based on feature interactions.
30. `Trompt <https://arxiv.org/abs/2305.18446>`_ : A prompt-based neural network for separating intrinsic column features and sample-specific feature importance.

==========================
Classical Methods
==========================

TALENT integrates the following classical machine learning methods, which serve as strong baselines for tabular data tasks:

1. **CatBoost**: A gradient boosting algorithm that excels at handling categorical features and performing well on tabular datasets.
2. **Dummy Classifier**: A simple baseline method that outputs the most frequent class or mean value, used to benchmark against random or naïve predictions.
3. **K-Nearest Neighbors (KNN)**: A classic instance-based learning algorithm that makes predictions based on the closest training samples.
4. **LightGBM**: A highly efficient gradient boosting framework that uses decision tree algorithms to reduce memory usage and improve speed.
5. **Logistic Regression (LogReg)**: A basic classification method that models the probability of a binary outcome based on input features.
6. **Linear Regression (LR)**: A regression method that models the relationship between a dependent variable and one or more independent variables.
7. **Naive Bayes**: A probabilistic classifier based on Bayes' theorem, particularly useful for categorical data with strong independence assumptions.
8. **Nearest Class Mean (NCM)**: A classifier that assigns a sample to the class whose mean is closest to the sample, based on a distance metric.
9. **Random Forest**: An ensemble learning method that constructs multiple decision trees and merges them to improve the accuracy and robustness of predictions.
10. **Support Vector Machine (SVM)**: A powerful classification method that finds the hyperplane best separating different classes of data.
11. **XGBoost**: An advanced gradient boosting algorithm that is particularly effective for structured/tabular data and provides robust performance in competition settings.

==========================
Methodology Summary
==========================

TALENT provides a comprehensive toolkit for tabular data analysis, integrating both deep learning and classical machine learning models. These methods offer flexibility, customizability, and ease of integration into various tabular data tasks, making TALENT a powerful resource for researchers and practitioners alike.
