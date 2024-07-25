from typing import Union, List, Tuple, Dict

class NestedDict:
    """
    Dictionary that can be used with multiple indices. Instead of
    d = dict()
    d['first'] = dict()
    d['first']['second'] = 1.0

    we can use

    d = NestedDict()
    d['first', 'second'] = 1.0
    """
    def __init__(self, data_dict=None):
        self.data_dict = data_dict if data_dict is not None else {}

    def __getitem__(self, idxs):
        if not isinstance(idxs, tuple):
            idxs = (idxs,)
        d = self.data_dict
        for idx in idxs:
            d = d[idx]
        return d

    def __setitem__(self, idxs, value):
        if not isinstance(idxs, tuple):
            idxs = (idxs,)
        if isinstance(value, NestedDict):
            value = value.data_dict # allow to properly "hang in" value in the case that value is of type NestedDict?
        d = self.data_dict
        for i, idx in enumerate(idxs):
            if idx not in d or i+1 == len(idxs):
                v = value
                for rev_idx in idxs[:i:-1]:
                    v = {rev_idx: v}
                d[idx] = v
                return
            d = d[idx]

    def __contains__(self, item: Union[List, Tuple]):
        current_dict = self.data_dict
        for elem in item:
            if elem not in current_dict:
                return False
            current_dict = current_dict[elem]
        return True


    def get(self, idxs, default=None):
        try:
            return self[idxs]
        except KeyError:
            return default

    def _dict_update_rec(self, d1: dict, d2: dict):
        for key in d2:
            if key in d1:
                self._dict_update_rec(d1[key], d2[key])
            else:
                d1[key] = d2[key]

    def update(self, other: 'NestedDict'):
        self._dict_update_rec(self.data_dict, other.data_dict)

    def __str__(self):
        return str(self.data_dict)

    def __repr__(self):
        return f'NestedDict({str(self)})'

    def get_dict(self) -> Dict:
        return self.data_dict

    @staticmethod
    def from_kwargs(**kwargs):
        return NestedDict(
            {key: (value.data_dict if isinstance(value, NestedDict) else value) for key, value in kwargs.items()}
        )



if __name__ == '__main__':
    nd = NestedDict()
    nd['test', 'test'] = 1
    print(nd['test'])

