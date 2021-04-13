import json
import os
import pickle
import gzip

class DictObj(object):
    def __init__(self, data):
        for name, value in data.items():
            setattr(self, name, self._wrap(value))

    def __getitem__(self, key):
        return getattr(self, key)

    def _wrap(self, value):
        if isinstance(value, (tuple, list, set, frozenset)):
            return type(value)([self._wrap(v) for v in value])
        else:
            return DictObj(value) if isinstance(value, dict) else value

def load_config(conf):
    with open(os.path.join('config', '{}.json'.format(conf)), 'r') as f:
        config =json.load(f)
    return DictObj(config)

def data_loader(data_path):
    with gzip.open(data_path, "rb") as f:
        data = pickle.load(f)
    return data

def data_save(data, data_path):
    with gzip.open(data_path, 'wb') as f:
        pickle.dump(data, f)