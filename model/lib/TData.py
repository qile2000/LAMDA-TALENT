import torch
from torch.utils.data import Dataset

class TData(Dataset):
    def __init__(self, is_regression, X, Y, y_info, part):
        assert(part in ['train', 'val', 'test'])
        X_num, X_cat = X
        self.X_num = X_num[part] if X_num is not None else None
        self.X_cat = X_cat[part] if X_cat is not None else None
        self.Y, self.y_info = Y[part], y_info
        
        # self.num_class = 1 if is_regression else torch.unique(Y['train']).shape[0]
        
    def get_dim_in(self):
        return 0 if self.X_num is None else self.X_num.shape[1]

    def get_categories(self):
        return (
            None
            if self.X_cat is None
            else [
                len(set(self.X_cat[:, i].cpu().tolist()))
                for i in range(self.X_cat.shape[1])
            ]
        )

    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, i):
        if self.X_num is not None and self.X_cat is not None:
            data = (self.X_num[i], self.X_cat[i])
        elif self.X_cat is not None and self.X_num is None:
            data, label = self.X_cat[i], self.Y[i]
        else:
            data, label = self.X_num[i], self.Y[i]
        label = self.Y[i]
        return data, label

