"""
Runs tabular experiments
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

import yaml
import copy

import os

from aif360.algorithms.postprocessing import (EqOddsPostprocessing, RejectOptionClassification)
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, roc_auc_score, average_precision_score

from models.networks_tabular import load_model, train_model

from utils.evaluation import get_valid_objective, get_test_objective
from utils.data_utils import TabularData

from algorithms.pruning import prune_fc
from algorithms.biasGrad import bias_gradient_decent

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

logger = logging.getLogger("Debiasing")
log_handler = logging.StreamHandler()
logger.addHandler(log_handler)
logger.setLevel(logging.INFO)
log_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.propagate = False


def main(config):
    seeds = [np.random.randint(0, high=10000)]
    if 'seed' in config:
        seeds = config['seed']

    for seed in seeds:
        logger.info(f'Running the experiment for seed: {seed}.')
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Setup directories to save models and results
        Path('models').mkdir(exist_ok=True)
        Path('results').mkdir(exist_ok=True)
        Path('results/figures').mkdir(exist_ok=True)
        Path('results/logs').mkdir(exist_ok=True)

        # Get data
        logger.info(f'Loading data from dataset: {config["dataset"]}.')
        data = TabularData(config, seed, device)
        logger.info(f'Privileged group: {data.priv}')

        # Get trained model
        model = load_model(data.num_features, config.get('hyperparameters', {}))
        model_path = os.path.join('models', config['modelpath'] + str('_') + str(seed) + '.pt')
        if Path(model_path).is_file():
            logger.info(f'Loading Model from {model_path}.')
            model.load_state_dict(torch.load(model_path))
        else:
            logger.info(f'Training model from scratch.')
            train_model(model, data, epochs=config.get('epochs', 1001))
            torch.save(model.state_dict(), model_path)
        model_state_dict = copy.deepcopy(model.state_dict())

        # Preliminaries
        logger.info('Setting up preliminaries.')
        model.eval()

        with torch.no_grad():

            valid_pred = data.valid.copy(deepcopy=True)
            valid_pred.scores = model(data.X_valid)[:, 0].reshape(-1, 1).numpy()
            valid_pred.labels = np.array(valid_pred.scores > 0.5)

            test_pred = data.test.copy(deepcopy=True)
            test_pred.scores = model(data.X_test)[:, 0].reshape(-1, 1).numpy()
            test_pred.labels = np.array(test_pred.scores > 0.5)

        results_valid = {}
        results_test = {}

        if config['acc_metric'] == 'balanced_accuracy':
            print('For comparison: Balanced accuracy score: ' + str(
                balanced_accuracy_score(data.y_valid, valid_pred.scores > 0.5)))
        elif config['acc_metric'] == 'accuracy':
            print('For comparison: Accuracy score: ' + str(accuracy_score(data.y_valid, valid_pred.scores > 0.5)))
        elif config['acc_metric'] == 'f1_score':
            print('For comparison: F1 score: ' + str(f1_score(data.y_valid, valid_pred.scores > 0.5)))
        else:
            print('Accuracy metric not defined')
        print('For comparison: AUROC score: ' + str(roc_auc_score(data.y_valid, valid_pred.scores)))
        print('For comparison: AP score: ' + str(average_precision_score(data.y_valid, valid_pred.scores)))

        # Evaluate the default model
        if 'default' in config['models']:
            logger.info('Finding best threshold for default model to minimize objective function')
            threshs = np.linspace(0, 1, 101)
            performances = []
            for thresh in threshs:
                if config['acc_metric'] == 'balanced_accuracy':
                    perf = balanced_accuracy_score(data.y_valid, valid_pred.scores > thresh)
                elif config['acc_metric'] == 'accuracy':
                    perf = accuracy_score(data.y_valid, valid_pred.scores > thresh)
                elif config['acc_metric'] == 'f1_score':
                    perf = f1_score(data.y_valid, valid_pred.scores > thresh)
                else:
                    print('Accuracy metric not defined')
                performances.append(perf)
            best_thresh = threshs[np.argmax(performances)]

            logger.info('Evaluating default model with best threshold.')
            results_valid['default'] = get_valid_objective(valid_pred.scores > best_thresh, data, config)
            results_test['default'] = get_test_objective(test_pred.scores > best_thresh, data, config)
            logger.info(f'Results: {results_test["default"]}')

        # Evaluate pruning
        if 'pruning' in config['models']:
            print()
            model_pruned = prune_fc(model=model, data=data, config=config, seed=seed, plot=True, display=False)

            with torch.no_grad():
                valid_pred_ = data.valid.copy(deepcopy=True)
                valid_pred_.scores = model_pruned(data.X_valid)[:, 0].reshape(-1, 1).detach().numpy()
                valid_pred_.labels = np.array(valid_pred_.scores > 0.5)

                test_pred_ = data.test.copy(deepcopy=True)
                test_pred_.scores = model_pruned(data.X_test)[:, 0].reshape(-1, 1).detach().numpy()
                test_pred_.labels = np.array(test_pred_.scores > 0.5)

                logger.info('Finding best threshold for pruned model to minimize objective function')
                threshs = np.linspace(0, 1, 101)
                performances = []
                for thresh in threshs:
                    if config['acc_metric'] == 'balanced_accuracy':
                        perf = balanced_accuracy_score(data.y_valid, valid_pred_.scores > thresh)
                    elif config['acc_metric'] == 'accuracy':
                        perf = accuracy_score(data.y_valid, valid_pred_.scores > thresh)
                    elif config['acc_metric'] == 'f1_score':
                        perf = f1_score(data.y_valid, valid_pred_.scores > thresh)
                    else:
                        print('Accuracy metric not defined')
                    performances.append(perf)
                best_thresh = threshs[np.argmax(performances)]

            logger.info('Evaluating pruned model with best threshold.')
            results_valid['pruning'] = get_valid_objective(valid_pred_.scores > best_thresh, data, config)
            logger.info(f'Results validation: {results_valid["pruning"]}')
            results_test['pruning'] = get_test_objective(test_pred_.scores > best_thresh, data, config)
            logger.info(f'Results test: {results_test["pruning"]}')

        # Evaluate bias gradient descent/ascent
        if 'biasGrad' in config['models']:
            print()
            asc = results_valid['default']['bias'] < 0
            model_ = bias_gradient_decent(model=model, data=data, config=config, seed=seed, plot=True, display=False,
                                          asc=asc)

            with torch.no_grad():
                valid_pred_ = data.valid.copy(deepcopy=True)
                valid_pred_.scores = model_(data.X_valid)[:, 0].reshape(-1, 1).detach().numpy()
                valid_pred_.labels = np.array(valid_pred_.scores > 0.5)

                test_pred_ = data.test.copy(deepcopy=True)
                test_pred_.scores = model_(data.X_test)[:, 0].reshape(-1, 1).detach().numpy()
                test_pred_.labels = np.array(test_pred_.scores > 0.5)

                logger.info('Finding best threshold for pruned model to minimize objective function')
                threshs = np.linspace(0, 1, 101)
                performances = []
                for thresh in threshs:
                    if config['acc_metric'] == 'balanced_accuracy':
                        perf = balanced_accuracy_score(data.y_valid, valid_pred_.scores > thresh)
                    elif config['acc_metric'] == 'accuracy':
                        perf = accuracy_score(data.y_valid, valid_pred_.scores > thresh)
                    elif config['acc_metric'] == 'f1_score':
                        perf = f1_score(data.y_valid, valid_pred_.scores > thresh)
                    else:
                        print('Accuracy metric not defined')
                    performances.append(perf)
                best_thresh = threshs[np.argmax(performances)]

            logger.info('Evaluating debiased model with best threshold.')
            results_valid['biasGrad'] = get_valid_objective(valid_pred_.scores > best_thresh, data, config)
            logger.info(f'Results validation: {results_valid["biasGrad"]}')
            results_test['biasGrad'] = get_test_objective(test_pred_.scores > best_thresh, data, config)
            logger.info(f'Results test: {results_test["biasGrad"]}')

        # Evaluate ROC post-processing
        if 'ROC' in config['models']:
            metric_map = {
                'spd': 'Statistical parity difference',
                'aod': 'Average odds difference',
                'eod': 'Equal opportunity difference'
            }
            ROC = RejectOptionClassification(unprivileged_groups=data.unpriv,
                                             privileged_groups=data.priv,
                                             low_class_thresh=0.01, high_class_thresh=0.99,
                                             num_class_thresh=100, num_ROC_margin=50,
                                             metric_name=metric_map[config['metric']],
                                             metric_ub=config['objective']['epsilon'],
                                             metric_lb=-config['objective']['epsilon'])

            logger.info('Training ROC model with validation dataset.')
            ROC = ROC.fit(data.valid, valid_pred)

            logger.info('Evaluating ROC model.')
            y_pred = ROC.predict(valid_pred).labels.reshape(-1)
            results_valid['ROC'] = get_valid_objective(y_pred, data, config)
            logger.info(f'Results: {results_valid["ROC"]}')

            y_pred = ROC.predict(test_pred).labels.reshape(-1)
            results_test['ROC'] = get_test_objective(y_pred, data, config)
            ROC = None

        # Evaluate equality of odds post-processing
        if 'EqOdds' in config['models']:
            eqodds = EqOddsPostprocessing(privileged_groups=data.priv,
                                          unprivileged_groups=data.unpriv)

            logger.info('Training Equality of Odds model with validation dataset.')
            eqodds = eqodds.fit(data.valid, valid_pred)

            logger.info('Evaluating Equality of Odds model.')
            y_pred = eqodds.predict(valid_pred).labels.reshape(-1)
            results_valid['EqOdds'] = get_valid_objective(y_pred, data, config)
            logger.info(f'Results: {results_valid["EqOdds"]}')

            y_pred = eqodds.predict(test_pred).labels.reshape(-1)
            results_test['EqOdds'] = get_test_objective(y_pred, data, config)
            eqodds = None

        # Evaluate random perturbation intra-processing
        if 'random' in config['models']:
             from algorithms.random import random_debiasing
             results_valid['random'], results_test['random'] = random_debiasing(model_state_dict, data,
                                                                                config, device)

        # Evaluate adversarial intra-processing
        if 'adversarial' in config['models']:
            from algorithms.adversarial import adversarial_debiasing
            results_valid['adversarial'], results_test['adversarial'] = adversarial_debiasing(model_state_dict, data,
                                                                                              config, device)

        # Evaluate adversarial in-processing
        if 'mitigating' in config['models']:
            from algorithms.mitigating import mitigating_debiasing
            results_valid['mitigating'], results_test['mitigating'] = mitigating_debiasing(model_state_dict, data,
                                                                                           config, device)

        # Save the results
        results_valid['config'] = config
        logger.info(f'Validation Results: {results_valid}')
        logger.info(f'Saving validation results to {config["experiment_name"]}_valid_output_{seed}.json')
        with open(Path('results') / 'logs' / f'{config["experiment_name"]}_valid_output_{seed}.json', 'w') as fh:
            json.dump(results_valid, fh)

        results_test['config'] = config
        logger.info(f'Test Results: {results_test}')
        logger.info(f'Saving validation results to {config["experiment_name"]}_test_output_{seed}.json')
        with open(Path('results') / 'logs' / f'{config["experiment_name"]}_test_output_{seed}.json', 'w') as fh:
            json.dump(results_test, fh)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Path to configuration yaml file.')
    args = parser.parse_args()
    with open(args.config, 'r') as fh:
        config = yaml.load(fh, Loader=yaml.FullLoader)
    main(config)
