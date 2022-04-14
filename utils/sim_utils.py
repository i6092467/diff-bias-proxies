"""
Numerical simulation utility functions.
"""
import numpy as np

from sklearn.datasets import make_low_rank_matrix


def unit_vector(vector):
    """
    Returns the unit vector of the vector.
    """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """
    Returns the angle in radians between vectors 'v1' and 'v2'
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def random_nonlin_map(n_in, n_out, n_hidden, rank=1000):
    # Random MLP mapping
    W_0 = make_low_rank_matrix(n_in, n_hidden, effective_rank=rank)
    W_1 = make_low_rank_matrix(n_hidden, n_hidden, effective_rank=rank)
    W_2 = make_low_rank_matrix(n_hidden, n_out, effective_rank=rank)
    # Disabled biases for now...
    b_0 = np.random.uniform(0, 0, (1, n_hidden))
    b_1 = np.random.uniform(0, 0, (1, n_hidden))
    b_2 = np.random.uniform(0, 0, (1, n_out))

    nlin_map = lambda x: np.matmul(ReLU(np.matmul(ReLU(np.matmul(x, W_0) + np.tile(b_0, (x.shape[0], 1))),
                                                       W_1) + np.tile(b_1, (x.shape[0], 1))), W_2) + \
                         np.tile(b_2, (x.shape[0], 1))

    return nlin_map


def ReLU(x):
    return x * (x > 0)
