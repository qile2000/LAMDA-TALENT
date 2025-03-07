from typing import Union, List, Optional

import numpy as np
import pandas as pd
import torch
from pandas import Index
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer

from TALENT.model.lib.realmlp.data.data import DictDataset, TensorInfo


class ToDictDatasetConverter:
    def __init__(self, cat_features: Optional[Union[List[bool], np.ndarray]] = None):
        self.cat_features = cat_features if cat_features is None else np.asarray(cat_features, dtype=np.bool_)
        self.num_tf = None
        self.cat_tf = None
        self.fitted = False
        self.tensor_infos = None
        self.fitted_columns = None
        self.fitted_type = None

    def fit_transform(self, x: Union[np.ndarray, pd.DataFrame, pd.Series, DictDataset]) -> DictDataset:
        self.fitted = True
        self.fitted_type = type(x)

        if isinstance(x, DictDataset):
            return x

        x = pd.DataFrame(x)
        self.fitted_columns = set(x.columns)

        if self.cat_features is not None:
            cat_columns = list(x.columns[self.cat_features])
            num_columns = list(x.columns[~self.cat_features])
            self.num_tf = ColumnTransformer(transformers=[
                ('continuous', FunctionTransformer(), num_columns),
            ])
            self.cat_tf = ColumnTransformer(transformers=[
                ('categorical', OrdinalEncoder(dtype=np.int64, handle_unknown='use_encoded_value', unknown_value=-1,
                                               encoded_missing_value=-1), cat_columns)
            ])
        else:
            self.num_tf = ColumnTransformer(transformers=[
                ('continuous', FunctionTransformer(), make_column_selector(dtype_include='number')),
                # todo: include this if we can make skrub a dependency
                # ('datetime', DatetimeEncoder(), make_column_selector(dtype_include=['datetime', 'datetimetz']))
            ])
            self.cat_tf = ColumnTransformer(transformers=[
                ('categorical', OrdinalEncoder(dtype=np.int64, handle_unknown='use_encoded_value', unknown_value=-1,
                                               encoded_missing_value=-1),
                 make_column_selector(dtype_include=["string", "object", "category"]))
            ])

        x_cont = torch.as_tensor(self.num_tf.fit_transform(x), dtype=torch.float32)
        x_cat = torch.as_tensor(self.cat_tf.fit_transform(x) + 1, dtype=torch.long)

        # print(f'{self.num_tf.transformers_=}')
        # print(f'{self.cat_tf.transformers_=}')

        for col_tfm in [self.num_tf, self.cat_tf]:
            for name, tfm, cols in col_tfm.transformers_:
                if tfm != 'drop':
                    # todo: log this at an appropriate level instead of printing
                    # print(f'Columns classified as {name}: {list(cols)}')
                    pass

        cat_sizes = torch.max(x_cat, dim=0)[0] + 1
        self.tensor_infos = {'x_cont': TensorInfo(feat_shape=x_cont.shape[1:]),
                             'x_cat': TensorInfo(cat_sizes=cat_sizes)}

        return DictDataset(tensors={'x_cont': x_cont, 'x_cat': x_cat}, tensor_infos=self.tensor_infos)

    def transform(self, x: Union[np.ndarray, pd.DataFrame, pd.Series, DictDataset]) -> DictDataset:
        if not self.fitted:
            raise ValueError("Call fit() first to fit the converter.")
        if not isinstance(x, self.fitted_type):
            raise ValueError(f'Different input types during fit and predict: {self.fitted_type} and {type(x)}')

        if isinstance(x, DictDataset):
            # todo: could check whether cat_sizes etc. match?
            return x

        x = pd.DataFrame(x)

        # print(set(x.columns), self.fitted_columns)

        if set(x.columns) != self.fitted_columns:
            print('Raising column error')
            raise ValueError(f'Different columns during fit() and predict(): {self.fitted_columns} and {set(x.columns)}')

        x_cont = torch.as_tensor(self.num_tf.transform(x), dtype=torch.float32)
        x_cat = torch.as_tensor(self.cat_tf.transform(x) + 1, dtype=torch.long)

        return DictDataset(tensors={'x_cont': x_cont, 'x_cat': x_cat}, tensor_infos=self.tensor_infos)


if __name__ == '__main__':
    data = {'Continuous1': [1.2, 2.3, 3.4, 4.5, 5.6],
            'Continuous2': [5.6, 6.7, 7.8, 8.9, 10.0],
            'Category1': ['A', 'B', 'A', 'C', None],
            'Category2': ['X', 'Y', None, 'X', None]}
    df = pd.DataFrame(data)
    df['Category2'] = df['Category2'].astype('category')

    print(set(df.columns) == set(df.columns))

    print(ToDictDatasetConverter(cat_features=[True, False, True, True]).fit_transform(df).tensors)
    print(ToDictDatasetConverter().fit_transform(df).tensors)
