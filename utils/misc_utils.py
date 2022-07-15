"""
Miscellaneous utility functions
"""
import os

import random

import torch

import numpy as np

import json

from tabulate import tabulate


def tabulate_results(ex_name: str, seeds: np.ndarray, methods: np.ndarray, epsilon: float, method_names=None,
                     log_dir: str = None):
    """Prints out the experiment's results in a tabular format"""
    if method_names is not None:
        assert len(method_names) == len(methods)
    else:
        method_names = methods

    # Load .json files with the results
    results = {}
    for s in seeds:
        if log_dir is None:
            file_s = open(os.path.join('bin/results/logs', ex_name + '_' + str(s) + '.json'))
        else:
            file_s = open(os.path.join(log_dir, ex_name + '_' + str(s) + '.json'))
        res_s = json.load(file_s)
        results[str(s)] = res_s

    # Construct the table
    my_table = []
    cnt = 0
    for meth in methods:
        meth_bias = [results[str(s)][meth]['bias'] for s in seeds]
        meth_perf = [results[str(s)][meth]['performance'] for s in seeds]
        my_table.append([method_names[cnt],
                         "{:1.3f}".format(np.round(np.mean(meth_bias), 3)) + \
                         ' +/- ' + "{:1.3f}".format(np.round(np.std(meth_bias), 3)),
                         "{:1.3f}".format(np.round(np.mean(meth_perf), 3)) + \
                         ' +/- ' + "{:1.3f}".format(np.round(np.std(meth_perf), 3))])
        cnt += 1
    headers = ['Method', 'Bias', 'Performance']

    print(tabulate(my_table, headers, tablefmt="github"))


def set_seeds(seed):
    """Fixes random seeds for the experiment replicability"""
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
