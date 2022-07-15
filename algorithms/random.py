"""
Random perturbation intra-processing algorithm by Savani et al. (2020) [https://arxiv.org/abs/2006.08564]

Code adapted from https://github.com/abacusai/intraprocessing_debiasing
"""
import copy
import logging
import math

import numpy as np
import torch

from models.networks_tabular import load_model
from utils.evaluation import get_best_thresh, get_test_objective, get_valid_objective

import progressbar

logger = logging.getLogger("Debiasing")


def random_debiasing(model_state_dict, data, config, device, verbose=True):
    """Runs random perturbation intra-processing, returns a perturbed network maximising the constrained objective"""
    if verbose:
        print('Perturbing the network randomly...')
        print()
    rand_model = load_model(data.num_features, config.get('hyperparameters', {}))
    rand_model.to(device)
    rand_result = {'objective': -math.inf, 'model': rand_model.state_dict(), 'thresh': -1}
    if verbose:
        bar = progressbar.ProgressBar(maxval=config['random']['num_trials'])
        bar.start()
        bar_cnt = 0
    for iteration in range(config['random']['num_trials']):
        rand_model.load_state_dict(model_state_dict)
        for param in rand_model.parameters():
            param.data = param.data * (torch.randn_like(param) * config['random']['stddev'] + 1)

        rand_model.eval()
        with torch.no_grad():
            scores = rand_model(data.X_valid_gpu)[:, 0].reshape(-1).cpu().numpy()

        threshs = np.linspace(0, 1, 101)
        best_rand_thresh, best_obj = get_best_thresh(scores, threshs, data, config,  margin=config['random']['margin'])
        if best_obj > rand_result['objective']:
            rand_result = {'objective': best_obj, 'model': copy.deepcopy(rand_model.state_dict()),
                           'thresh': best_rand_thresh}
            rand_model.eval()
            with torch.no_grad():
                y_pred = (rand_model(data.X_test_gpu)[:, 0] > best_rand_thresh).reshape(-1).cpu().numpy()
            best_test_result = get_test_objective(y_pred, data, config)['objective']

        if verbose:
            bar.update(bar_cnt)
            bar_cnt += 1

    if verbose:
        print('\n' * 2)

    rand_model.load_state_dict(rand_result['model'])
    rand_model.eval()
    with torch.no_grad():
        y_pred = (rand_model(data.X_valid_gpu)[:, 0] > rand_result['thresh']).reshape(-1).cpu().numpy()
    results_valid = get_valid_objective(y_pred, data, config)

    rand_model.eval()
    with torch.no_grad():
        y_pred = (rand_model(data.X_test_gpu)[:, 0] > rand_result['thresh']).reshape(-1).cpu().numpy()
    results_test = get_test_objective(y_pred, data, config)
    logger.info(f'Results: {results_test}')

    return results_valid, results_test
