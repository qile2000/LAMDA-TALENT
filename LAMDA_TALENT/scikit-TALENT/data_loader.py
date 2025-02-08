from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from sklearn.model_selection import train_test_split


def separate_features(
    X_subset: np.ndarray, categorical_indices: List[int], numeric_indices: List[int]
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Helper function to separate numeric and categorical features from a feature matrix.

    :param X_subset: Feature matrix as a NumPy array.
    :param categorical_indices: List of column indices corresponding to categorical features.
    :param numeric_indices: List of column indices corresponding to numeric features.
    :return: A tuple containing:
             - Numeric features as a NumPy array or None if no numeric features.
             - Categorical features as a NumPy array or None if no categorical features.
    """
    X_num = X_subset[:, numeric_indices] if numeric_indices else None
    X_cat = X_subset[:, categorical_indices] if categorical_indices else None
    return X_num, X_cat


def split_train_val(
    X_train: np.ndarray,
    y_train: np.ndarray,
    categorical_features: List[bool],
    task_type: str = "regression",
    val_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[
    Tuple[
        Optional[Dict[str, np.ndarray]],
        Optional[Dict[str, np.ndarray]],
        Dict[str, np.ndarray],
    ]
]:
    """
    Split the pre-split training data into training and validation sets, separate features,
    and merge them into a single dictionary.

    :param X_train: Training feature matrix as a NumPy array of shape (n_samples, n_features).
    :param y_train: Training labels as a NumPy array of shape (n_samples,).
    :param categorical_features: List of booleans indicating which features are categorical.
                                 Length must be equal to the number of features in X_train.
    :param task_type: Type of the task. Must be one of "regression", "multiclass", or "binclass".
                      Defaults to "regression".
    :param val_size: Proportion of the training data to include in the validation split.
                     Must be between 0.0 and 1.0. Defaults to 0.2 (20%).
    :param random_state: Seed used by the random number generator. Defaults to 42.

    :return: A tuple containing:
             - train_val_data: Tuple[N_train_val_dict, C_train_val_dict, y_train_val_dict]
    """
    # -----------------------------
    # 1. Input Validation
    # -----------------------------
    if not isinstance(X_train, np.ndarray):
        raise ValueError("X_train must be a NumPy array.")
    if not isinstance(y_train, np.ndarray):
        raise ValueError("y_train must be a NumPy array.")
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError("Number of samples in X_train and y_train must be the same.")
    if len(categorical_features) != X_train.shape[1]:
        raise ValueError(
            "Length of categorical_features must match number of features in X_train."
        )
    if task_type not in {"regression", "multiclass", "binclass"}:
        raise ValueError(
            'task_type must be one of "regression", "multiclass", or "binclass".'
        )
    if not (0.0 < val_size < 1.0):
        raise ValueError("val_size must be between 0.0 and 1.0.")

    # -----------------------------
    # 2. Identify Feature Indices
    # -----------------------------
    categorical_indices = [i for i, is_cat in enumerate(categorical_features) if is_cat]
    numeric_indices = [i for i, is_cat in enumerate(categorical_features) if not is_cat]

    # -----------------------------
    # 3. Split into Train and Val
    # -----------------------------
    stratify_param = y_train if task_type in {"multiclass", "binclass"} else None
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_size,
        random_state=random_state,
        shuffle=True,
        stratify=stratify_param,
    )

    # -----------------------------
    # 4. Separate Features
    # -----------------------------
    N_tr, C_tr = separate_features(X_tr, categorical_indices, numeric_indices)
    N_val, C_val = separate_features(X_val, categorical_indices, numeric_indices)

    # -----------------------------
    # 5. Prepare Data Dictionaries
    # -----------------------------
    # Training Data
    y_train_dict = {"train": y_tr}
    N_train_dict = {"train": N_tr} if numeric_indices else None
    C_train_dict = {"train": C_tr} if categorical_indices else None

    # Validation Data
    y_val_dict = {"val": y_val}
    N_val_dict = {"val": N_val} if numeric_indices else None
    C_val_dict = {"val": C_val} if categorical_indices else None

    # Merge Train and Val Data into single dict
    N_train_val_dict = {"train": N_tr, "val": N_val} if numeric_indices else None
    C_train_val_dict = {"train": C_tr, "val": C_val} if categorical_indices else None
    y_train_val_dict = {"train": y_tr, "val": y_val}

    train_val_data = (N_train_val_dict, C_train_val_dict, y_train_val_dict)

    return train_val_data


def convert_test(
    X_test: np.ndarray,
    categorical_features: List[bool],
    y_test: Optional[np.ndarray] = None,
    default_y=0,
) -> Tuple[
    Optional[Dict[str, np.ndarray]],
    Optional[Dict[str, np.ndarray]],
    Dict[str, np.ndarray],
]:
    """
    Convert the pre-split test data into the desired dictionary format with separated features.
    If y_test is not provided, create a dummy placeholder.

    :param X_test: Test feature matrix as a NumPy array of shape (n_samples, n_features).
    :param y_test: Test labels as a NumPy array of shape (n_samples,). If None, dummy values are used.
    :param categorical_features: List of booleans indicating which features are categorical.
                                 Length must be equal to the number of features in X_test.

    :return: A tuple containing:
             - N_test_dict: Dictionary with key "test" pointing to numeric features or None.
             - C_test_dict: Dictionary with key "test" pointing to categorical features or None.
             - y_test_dict: Dictionary with key "test" pointing to labels.
    """
    # -----------------------------
    # 1. Input Validation
    # -----------------------------
    if not isinstance(X_test, np.ndarray):
        raise ValueError("X_test must be a NumPy array.")
    if y_test is not None and not isinstance(y_test, np.ndarray):
        raise ValueError("y_test must be a NumPy array or None.")
    if y_test is not None and X_test.shape[0] != y_test.shape[0]:
        raise ValueError("Number of samples in X_test and y_test must be the same.")
    if len(categorical_features) != X_test.shape[1]:
        raise ValueError(
            "Length of categorical_features must match number of features in X_test."
        )

    # -----------------------------
    # 2. Identify Feature Indices
    # -----------------------------
    categorical_indices = [i for i, is_cat in enumerate(categorical_features) if is_cat]
    numeric_indices = [i for i, is_cat in enumerate(categorical_features) if not is_cat]

    # -----------------------------
    # 3. Separate Features
    # -----------------------------
    N_test, C_test = separate_features(X_test, categorical_indices, numeric_indices)

    # -----------------------------
    # 4. Handle Missing y_test
    # -----------------------------
    if y_test is None:
        if isinstance(default_y, (list, np.ndarray)):
            y_test = np.random.choice(default_y, X_test.shape[0])
        else:
            y_test = np.full(X_test.shape[0], default_y)

    # -----------------------------
    # 5. Prepare Data Dictionaries
    # -----------------------------
    y_test_dict = {"test": y_test}
    N_test_dict = {"test": N_test} if numeric_indices else None
    C_test_dict = {"test": C_test} if categorical_indices else None

    return N_test_dict, C_test_dict, y_test_dict


def generate_info(
    categorical_features: List[bool], task_type: str = "regression"
) -> Dict[str, Any]:
    """
    Generate metadata information for the dataset.

    :param categorical_features: List of booleans indicating which features are categorical.
    :param task_type: Type of the task. Must be one of "regression", "multiclass", or "binclass".
                      Defaults to "regression".
    :return: Dictionary containing metadata about the dataset.
    """
    n_num_features = sum(not is_cat for is_cat in categorical_features)
    n_cat_features = sum(is_cat for is_cat in categorical_features)

    info = {
        "task_type": task_type,
        "n_num_features": n_num_features,
        "n_cat_features": n_cat_features,
    }

    return info


# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    import numpy as np

    # Generate synthetic data
    num_samples = 1000
    num_features = 20
    X = np.random.randn(num_samples, num_features)
    y = np.random.randint(0, 2, size=num_samples)  # Binary classification

    # Define categorical features (e.g., first 5 features are categorical)
    categorical_features = [True] * 5 + [False] * (num_features - 5)

    # Split the data into pre-split train and test sets
    X_train_pre, X_test, y_train_pre, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, shuffle=True, stratify=y
    )

    # Now, use the two functions to further split train into train and val, and convert test
    train_data = split_train_val(
        X_train=X_train_pre,
        y_train=y_train_pre,
        categorical_features=categorical_features,
        task_type="binclass",
        val_size=0.2,
        random_state=0,
    )

    test_data = convert_test(
        X_test=X_test, y_test=y_test, categorical_features=categorical_features
    )

    # Generate info
    info = generate_info(
        categorical_features=categorical_features, task_type="binclass"
    )

    # Accessing the data
    N_train_dict, C_train_dict, y_train_dict = train_data
    N_test_dict, C_test_dict, y_test_dict = test_data

    # Displaying info
    print("Dataset Info:")
    print(info)

    # Example output
    print("\nTrain Labels:")
    print(y_train_dict["train"])

    print("\nTest Labels:")
    print(y_test_dict["test"])

    if N_train_dict is not None:
        print("\nNumber of Numeric Features in Train:", N_train_dict["train"].shape[1])
    if C_train_dict is not None:
        print(
            "Number of Categorical Features in Train:", C_train_dict["train"].shape[1]
        )

    if N_test_dict is not None:
        print("\nNumber of Numeric Features in Test:", N_test_dict["test"].shape[1])
    if C_test_dict is not None:
        print("Number of Categorical Features in Test:", C_test_dict["test"].shape[1])