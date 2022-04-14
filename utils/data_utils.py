"""
Data handling utilities.
"""
import torch

import pandas as pd

from sklearn.preprocessing import StandardScaler

from datasets.tabular import get_data

from aif360.datasets import StandardDataset


class TabularData(object):
    def __init__(self, config, seed, device):
        self.train, self.valid, self.test, self.priv, self.unpriv = get_data(config['dataset'], config['protected'],
                                                                             seed=seed, config=config)
        
        # priv_index is the index of the priviledged column.
        priv_index = self.train.protected_attribute_names.index(list(self.priv[0].keys())[0])

        scale_orig = StandardScaler()
        self.X_train = torch.tensor(scale_orig.fit_transform(self.train.features), dtype=torch.float32)
        self.y_train = torch.tensor(self.train.labels.ravel(), dtype=torch.float32)
        self.p_train = self.train.protected_attributes[:, priv_index]

        self.X_valid = torch.tensor(scale_orig.transform(self.valid.features), dtype=torch.float32)
        self.X_valid_gpu = self.X_valid.to(device)
        self.y_valid = torch.tensor(self.valid.labels.ravel(), dtype=torch.float32)
        self.y_valid_gpu = self.y_valid.to(device)
        self.p_valid = self.valid.protected_attributes[:, priv_index]
        self.p_valid_gpu = torch.tensor(self.p_valid).to(device)

        valid_train_indices, valid_valid_indices = torch.split(torch.randperm(self.X_valid.size(0)), int(0.7*self.X_valid.size(0)))
        self.X_valid_train, self.X_valid_valid = self.X_valid[valid_train_indices, :], self.X_valid[valid_valid_indices, :]
        self.y_valid_train, self.y_valid_valid = self.y_valid[valid_train_indices], self.y_valid[valid_valid_indices]
        self.p_valid_train, self.p_valid_valid = self.p_valid[valid_train_indices], self.p_valid[valid_valid_indices]

        self.X_test = torch.tensor(scale_orig.transform(self.test.features), dtype=torch.float32)
        self.X_test_gpu = self.X_test.to(device)
        self.y_test = torch.tensor(self.test.labels.ravel(), dtype=torch.float32)
        self.y_test_gpu = self.y_test.to(device)
        self.p_test = self.test.protected_attributes[:, priv_index]
        self.p_test_gpu = torch.tensor(self.p_test).to(device)

        self.num_features = self.X_train.size(1)


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def to_dataframe(y_true, y_pred, y_prot, prot_name):
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, prot_name: y_prot})
    dataset = StandardDataset(df, 'y_true', [1.], [prot_name], [[1.]])
    dataset.scores = y_pred.reshape(-1, 1)
    return dataset
