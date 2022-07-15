"""
Data loaders for synthetic data
"""
import numpy as np

import scipy.stats as stats

import pandas as pd

from utils.sim_utils import random_nonlin_map

from aif360.datasets import StandardDataset


def simulate_loh(n, effect_model='B1', alpha=1.0):
    """Simulation based on the paper by Loh et al. [DOI: https://doi.org/10.1002/widm.1326]"""
    X = np.zeros((n, 10))

    X[:, 0] = np.random.normal(loc=0, scale=1, size=(n, ))
    X[:, 1:3] = np.random.multivariate_normal(mean=np.array([0., 0.]), cov=np.array([[1., 0.5], [0.5, 1.]]),
                                              size=(n, ))
    X[:, 3] = np.random.exponential(scale=1., size=(n, ))
    X[:, 4] = np.random.choice(a=np.array([0., 1.]),  size=(n, ), replace=True, p=np.array([0.5, 0.5]))
    X[:, 5] = np.random.choice(a=np.arange(0, 10),  size=(n, ), replace=True, p=np.array([0.1 for _ in range(10)]))
    X[:, 6:10] = np.random.multivariate_normal(mean=np.zeros((4, )), cov=(0.5 * np.ones((4, 4)) + np.eye(4) * 0.5),
                                               size=(n, ))

    Z = np.random.choice(a=np.array([0., 1.]),  size=(n, ), replace=True, p=np.array([0.5, 0.5]))

    logits = np.zeros((n, ))
    if effect_model == 'B1':
        logits = 0.5 * (X[:, 0] + X[:, 1] - X[:, 4]) + alpha * 2 * Z * ((X[:, 5] % 2 == 1) * 1.0)
    elif effect_model == 'B2':
        logits = 0.5 * X[:, 1] + alpha * 2 * Z * ((X[:, 0] > 0) * 1.0)
    elif effect_model == 'B3':
        logits = 0.3 * (X[:, 0] + X[:, 1]) + alpha * 2 * Z * ((X[:, 0] > 0) * 1.0)
    elif effect_model == 'B4':
        logits = 0.3 * (X[:, 1] + X[:, 2] - 2) + alpha * 2 * Z * X[:, 3]
    elif effect_model == 'B5':
        logits = 0.2 * (X[:, 0] + X[:, 1] - 2) + alpha * 2 * Z * ((X[:, 0] < 1) * 1.0) * ((X[:, 5] % 2 == 1) * 1.0)
    elif effect_model == 'B6':
        logits = 0.5 * (X[:, 1] - 1) + alpha * 2 * Z * ((np.abs(X[:, 0]) < 0.8) * 1.0)
    elif effect_model == 'B7':
        logits = 0.2 * (X[:, 1] + 2 * X[:, 1] ** 2 - 6) + alpha * 2 * Z * ((X[:, 0] > 0) * 1.0)
    elif effect_model == 'B8':
        logits = 0.5 * X[:, 1] + alpha * 2 * Z * X[:, 4]
    else:
        raise NotImplementedError('ERROR: Treatment effect model not supported!')

    odds = np.exp(logits)
    probs_0 = 1 / (odds + 1)
    probs_1 = 1 - probs_0

    Y = np.array([np.random.choice(a=[0., 1.], size=(1, ), p=np.array([probs_0[i], probs_1[i]]))[0] for i in range(n)])

    return X, Z, Y


def simulate_zafar_lin(n, theta=1.0, seed=42):
    """Simulation based on the paper by Zafar et al. [https://arxiv.org/abs/1507.05259]"""
    np.random.seed(seed)

    rot_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    # Binary labels
    Y = np.random.choice(a=np.array([0., 1.]),  size=(n, ), replace=True, p=np.array([0.5, 0.5]))

    # Features
    X = np.zeros((n, 2))
    X[Y == 0, :] = np.random.multivariate_normal(mean=np.array([-2, -2]), cov=np.array([[10, 1], [1, 3]]),
                                                 size=(np.sum(Y == 0), ))
    X[Y == 1, :] = np.random.multivariate_normal(mean=np.array([2, 2]), cov=np.array([[5, 1], [1, 5]]),
                                                 size=(np.sum(Y == 1),))

    p_x_y0 = stats.multivariate_normal(mean=np.array([-2, -2]), cov=np.array([[10, 1], [1, 3]]))
    p_x_y1 = stats.multivariate_normal(mean=np.array([2, 2]), cov=np.array([[5, 1], [1, 5]]))

    X_ = np.dot(X, rot_mat)

    # Protected attribute
    probs_1 = p_x_y1.pdf(X_) / (p_x_y1.pdf(X_) + p_x_y0.pdf(X_))
    probs_0 = 1 - probs_1
    Z = np.array([np.random.choice(a=[0., 1.], size=(1,), p=np.array([probs_0[i], probs_1[i]]))[0] for i in range(n)])

    return X, Z, Y


def simulate_zafar_nlin(n, theta=1.0, seed=42):
    """A nonlinear extension of the simulation by Zafar et al. [https://arxiv.org/abs/1507.05259]"""
    X_tilde, Z, Y = simulate_zafar_lin(n=n, theta=theta, seed=seed)

    dec = random_nonlin_map(n_in=2, n_out=500, n_hidden=100, rank=50)

    X = dec(X_tilde)

    return X, Z, Y


class SyntheticZafarDataset(StandardDataset):
    """ Zafar et al. Dataset"""
    def __init__(self, label_name='Y', favorable_classes=[1],
                 protected_attribute_names=['Z'],
                 privileged_classes=[[1]],
                 instance_weights_name=None,
                 categorical_features=[],
                 features_to_keep=[],
                 features_to_drop=[],
                 na_values='', custom_preprocessing=None,
                 metadata=None, n=30000, theta=np.pi/2, seed=42):
        """See :obj:`StandardDataset` for a description of the arguments.
        """

        # Reproducibility
        np.random.seed(seed)

        X, Z, Y = simulate_zafar_nlin(n, theta=theta, seed=42)
        data_arr = np.concatenate((X, np.expand_dims(Z, 1), np.expand_dims(Y, 1)), axis=1)
        df = pd.DataFrame(data=data_arr, columns=np.concatenate((['X' + str(i) for i in range(1, X.shape[1] + 1)],
                                                                 ['Z'], ['Y'])))

        super(SyntheticZafarDataset, self).__init__(df=df, label_name=label_name, favorable_classes=favorable_classes,
                                                    protected_attribute_names=protected_attribute_names,
                                                    privileged_classes=privileged_classes,
                                                    instance_weights_name=instance_weights_name,
                                                    categorical_features=categorical_features,
                                                    features_to_keep=features_to_keep,
                                                    features_to_drop=features_to_drop, na_values=na_values,
                                                    custom_preprocessing=custom_preprocessing, metadata=metadata)


class SyntheticLohDataset(StandardDataset):
    """Loh et al. Dataset"""
    def __init__(self, label_name='Y', favorable_classes=[1],
                 protected_attribute_names=['Z'],
                 privileged_classes=[[1]],
                 instance_weights_name=None,
                 categorical_features=['X5', 'X6'],
                 features_to_keep=[],
                 features_to_drop=[],
                 na_values='', custom_preprocessing=None,
                 metadata=None, n=30000, alpha=1.0, seed=42):
        """See :obj:`StandardDataset` for a description of the arguments.
        """

        # Reproducibility
        np.random.seed(seed)

        X, Z, Y = simulate_loh(n=n, alpha=alpha)
        data_arr = np.concatenate((X, np.expand_dims(Z, 1), np.expand_dims(Y, 1)), axis=1)
        df = pd.DataFrame(data=data_arr, columns=np.concatenate((['X' + str(i) for i in range(1, X.shape[1] + 1)],
                                                                 ['Z'], ['Y'])))

        super(SyntheticLohDataset, self).__init__(df=df, label_name=label_name, favorable_classes=favorable_classes,
                                                  protected_attribute_names=protected_attribute_names,
                                                  privileged_classes=privileged_classes,
                                                  instance_weights_name=instance_weights_name,
                                                  categorical_features=categorical_features,
                                                  features_to_keep=features_to_keep,
                                                  features_to_drop=features_to_drop, na_values=na_values,
                                                  custom_preprocessing=custom_preprocessing, metadata=metadata)
