"""
Pruning intra-processing algorithm
"""
import os.path

import numpy as np

import copy

import torch
from torch import nn

from sklearn.metrics import balanced_accuracy_score

import utils.data_utils
from models.networks_ChestXRay import (ChestXRayVGG16Masked, ChestXRayResNet18Masked)

from utils.evaluation import (get_objective, get_test_objective_)

from utils.plotting import plot_pruning_results

import progressbar

from collections import OrderedDict

from typing import Dict, Callable


def choose_best_thresh_bal_acc(data: utils.data_utils.TabularData, valid_pred_scores: np.ndarray, n_thresh=101):
    """Optimises classification threshold w.r.t. balanced accuracy"""
    threshs = np.linspace(0, 1, n_thresh)
    performances = []
    for thresh in threshs:
        perf = balanced_accuracy_score(data.y_valid, valid_pred_scores > thresh)
        performances.append(perf)
    best_thresh = threshs[np.argmax(performances)]

    return best_thresh


def choose_best_thresh_bal_acc_(y_valid: np.ndarray, valid_pred_scores: np.ndarray, n_thresh=101):
    """Optimises classification threshold w.r.t. balanced accuracy"""
    # NOTE: this function is applied directly to numpy arrays, rather than a TabularData object
    threshs = np.linspace(0, 1, n_thresh)
    performances = []
    for thresh in threshs:
        perf = balanced_accuracy_score(y_valid, valid_pred_scores > thresh)
        performances.append(perf)
    best_thresh = threshs[np.argmax(performances)]

    return best_thresh


def save_pruning_trajectory(results: dict, seed: int, config: dict):
    """Saves traces of the bias, performance, and constrained objective during fine-tuning in a .csv file"""
    arr = np.stack((results['objective'], results['bias'], results['perf']), axis=1)
    np.savetxt(fname=os.path.join('results/logs/') + str(config['experiment_name'] + '_' + str(seed) +
                                                         '_trajectory' + '.csv'), X=arr)


def install_hooks_fc(model: nn.Module):
    """Installs forward hooks on fully connected layers of the model"""
    # NOTE: assumes that FC layers can accessed as model.fcs
    activation = {}
    n_units = {}

    # Create hooks to get layer activations from the model
    def get_activation(name):
        def hook(model, input, output):
            output.retain_grad()
            activation[name] = output

        return hook

    # Assume that all/most fully connected layers are stored in one parameter list
    for (i, fc) in enumerate(model.fcs):
        fc.register_forward_hook(get_activation('fc' + str(i + 1)))
        n_units['fc' + str(i + 1)] = fc.out_features

    model.fc0.register_forward_hook(get_activation('fc0'))
    n_units['fc0'] = model.fc0.out_features

    return activation, n_units


def install_hooks(layers):
    """Installs forward hooks on the given list of layers"""
    activation = {}
    handles = {}

    # Create hooks to get layer activations from the model
    def get_activation(name):
        def hook(model, input, output):
            output.retain_grad()
            activation[name] = output

        return hook

    # Go through the list of layers given
    for (i, l) in enumerate(layers):
        handles['l' + str(i)] = l.register_forward_hook(get_activation('l' + str(i)))

    return activation, handles


def remove_all_forward_hooks(model: torch.nn.Module) -> None:
    """Removes all forward hooks from the given model"""
    for name, child in model._modules.items():
        if child is not None:
            if hasattr(child, "_forward_hooks"):
                child._forward_hooks: Dict[int, Callable] = OrderedDict()
            remove_all_forward_hooks(child)


def eval_saliency(model: nn.Module, data: utils.data_utils.TabularData, idx: np.ndarray, activation: dict, config: dict,
                  total_n_units: int, val_only=False):
    """Evaluates the gradient-based bias influence of units on tabular data"""
    model.eval()
    model.zero_grad()

    # Forward through the model to get activations
    if val_only:
        preds = model(data.X_valid[idx])[:, 0]
    else:
        preds = model(data.X_train[idx])[:, 0]

    # Saliency of each unit, to be used as a pruning criterion
    coeffs = np.zeros((total_n_units,))

    # Compute the corresponding differentiable surrogate for the bias metric
    if config['metric'] == 'spd':
        if val_only:
            bias_measure = torch.mean(preds[data.p_valid[idx] == 0]) - torch.mean(preds[data.p_valid[idx] == 1])
        else:
            bias_measure = torch.mean(preds[data.p_train[idx] == 0]) - torch.mean(preds[data.p_train[idx] == 1])
    elif config['metric'] == 'eod':
        if val_only:
            bias_measure = torch.mean(preds[np.logical_and(data.p_valid[idx] == 0, data.y_valid[idx] == 1)]) - \
                           torch.mean(preds[np.logical_and(data.p_valid[idx] == 1, data.y_valid[idx] == 1)])
        else:
            bias_measure = torch.mean(preds[np.logical_and(data.p_train[idx] == 0, data.y_train[idx] == 1)]) - \
                           torch.mean(preds[np.logical_and(data.p_train[idx] == 1, data.y_train[idx] == 1)])
    else:
        NotImplementedError('Bias metric not supported!')

    # Backpropagate the bias
    bias_measure.backward()

    # Retrieve the corresponding gradient
    for (i, fc) in enumerate(model.fcs):
        grads_fc = torch.sum(activation['fc' + str(i + 1)].grad, 0).cpu().numpy()
        coeffs[((i + 1) * fc.out_features):((i + 2) * fc.out_features)] = grads_fc
    grads_fc = torch.sum(activation['fc0'].grad, 0).cpu().numpy()
    coeffs[0:model.fc0.out_features] = grads_fc

    return coeffs


def eval_saliency_dataloaders(model, layers, data_loader, activation, device, config, pruned=None):
    """Evaluates the gradient-based bias influence of units given the data loaders"""
    total_n_structs = 0
    n_structs = []
    start_idx = []
    end_idx = []

    model.eval()

    # NOTE: pass dummy input to figure out feature map sizes
    __ = model(torch.zeros((16, 3, 224, 224)).to(device).to(torch.float))

    for i_l, l in enumerate(layers):
        start_idx.append(total_n_structs)
        # Count prunable structures
        if isinstance(l, nn.Linear):
            total_n_structs += l.out_features
            n_structs.append(l.out_features)
        elif isinstance(l, nn.Conv2d):
            total_n_structs += activation['l' + str(i_l)].shape[1] * activation['l' + str(i_l)].shape[2] * \
                               activation['l' + str(i_l)].shape[3]
            n_structs.append(activation['l' + str(i_l)].shape[1] * activation['l' + str(i_l)].shape[2] *
                             activation['l' + str(i_l)].shape[3])
        else:
            NotImplementedError('Layer type not supported!')
        end_idx.append(total_n_structs)

    # Vector of saliencies
    coeffs = np.zeros((total_n_structs,))

    # Forward through the model to get activations
    # NOTE: this is a batched version of the algorithm presented in the paper
    tmp = 0
    for inputs, labels, attrs in data_loader:
        model.zero_grad()

        X = inputs.to(device)
        y = labels.to(device).to(torch.float)
        p = attrs.to(device)

        if pruned is None:
            outputs = model(X)
        else:
            outputs = model(X, pruned=np.array(pruned))

        preds = outputs[:, 0]

        # Compute the differentiable surrogate for the bias metric
        bias_measure = None
        if config['metric'] == 'spd':
            bias_measure = torch.mean(preds[p == 0]) - torch.mean(preds[p == 1])
        elif config['metric'] == 'eod':
            bias_measure = torch.mean(preds[torch.logical_and(p == 0, y == 1)]) - \
                            torch.mean(preds[torch.logical_and(p == 1, y == 1)])
        else:
            NotImplementedError('Bias metric not supported!')

        tmp += 1

        if not torch.isnan(bias_measure):
            # Backpropagate the bias
            bias_measure.backward()

            # Compute saliency of each structure
            for (i, l) in enumerate(layers):
                layer_key = 'l' + str(i)

                if isinstance(l, nn.Linear):
                    coeffs[start_idx[i]:end_idx[i]] += torch.mean(activation[layer_key].grad, dim=0).cpu().numpy()
                elif isinstance(l, nn.Conv2d):
                    coeffs[start_idx[i]:end_idx[i]] += torch.mean(activation[layer_key].grad, 0).cpu().numpy().flatten()
                else:
                    NotImplementedError('Layer type not supported!')

    return coeffs, n_structs, start_idx, end_idx


def prune_fc_units(model: nn.Module, to_prune: np.ndarray, n_units: dict, prune_first=False):
    """Prunes specified units in the fully connected layers by adjusting weight matrices"""
    if len(to_prune) == 0:
        return model
    cnt = 0
    if prune_first:
        n_units_i = n_units['fc0']
        # Find units to prune in this layer
        pruned_units_i = to_prune[np.logical_and(cnt <= to_prune, to_prune < cnt + n_units_i)] - cnt
        # Set incoming weights to 0
        model.fc0.weight[pruned_units_i, :] = 0
        model.fc0.bias[pruned_units_i] = 0
        cnt += n_units_i
    for (i, fc) in enumerate(model.fcs):
        n_units_i = n_units['fc' + str(i + 1)]
        # Find units to prune in this layer
        pruned_units_i = to_prune[np.logical_and(cnt <= to_prune, to_prune < cnt + n_units_i)] - cnt
        # Set incoming weights to 0
        fc.weight[pruned_units_i, :] = 0
        fc.bias[pruned_units_i] = 0
        cnt += n_units_i
    return model


def prune_fc(model, data, config, seed, plot=False, display=False, verbose=1):
    """Intra-processing debiasing procedure for pruning fully connected neural networks"""
    # Suppress warnings
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    if verbose:
        print('Pruning the network...')
        print()

    model_pruned = copy.deepcopy(model)

    # Determine the order in which structures are pruned, similar to bias GD/A
    # Predict on validation data
    with torch.enable_grad():
        # Predict on validation set
        valid_pred_scores = model_pruned(data.X_valid)[:, 0].reshape(-1, 1).detach().numpy()
    # Choose the best threshold w.r.t. the balanced accuracy on the held-out data
    best_thresh = choose_best_thresh_bal_acc(data=data, valid_pred_scores=valid_pred_scores)
    # Evaluate all metrics using the best threshold
    obj_dict = get_objective((valid_pred_scores > best_thresh) * 1., data.y_valid.numpy(), data.p_valid,
                             config['metric'], config['objective']['sharpness'],
                             config['objective']['epsilon'])
    asc = obj_dict['bias'] < 0

    # Create hooks to get layer activations from the model
    activation, n_units = install_hooks_fc(model=model_pruned)

    total_n_units = sum(list(n_units.values()))

    if not config['pruning']['val_only']:
        idx = np.arange(0, data.X_train.size(0))
    else:
        idx = np.arange(0, data.X_valid.size(0))

    # Evaluate unit gradient-based bias influence
    coeffs = eval_saliency(model=model_pruned, data=data, idx=idx, activation=activation, config=config,
                           total_n_units=total_n_units, val_only=config['pruning']['val_only'])

    # Sort the units according to their influence and the sign of the initial bias
    if asc:
        unit_inds = np.argsort(coeffs)
    else:
        unit_inds = np.argsort(-coeffs)

    model_pruned.eval()

    with torch.no_grad():
        if verbose:
            bar = progressbar.ProgressBar(maxval=int((len(unit_inds) + 1) / config['pruning']['step_size']))
            bar.start()
            bar_cnt = 0

        # Prune units step-by-step measuring performance changes for every sparsity level
        objective = []
        bias_metric = []
        pred_performance = []
        n_pruned = []
        pruned_inds = []
        pruned = []
        model_pruned_ = copy.deepcopy(model_pruned)
        start_ind = 0

        j_best = -1
        best_bias = 1

        for j in range(0, len(unit_inds) + 1, config['pruning']['step_size']):
            # Recompute influence dynamically for pruned networks
            if config['pruning']['dynamic'] and j > 1:
                with torch.enable_grad():
                    coeffs = eval_saliency(model=model_pruned_, data=data, idx=idx, activation=activation,
                                           config=config, total_n_units=total_n_units,
                                           val_only=config['pruning']['val_only'])
                    if asc:
                        unit_inds = np.argsort(coeffs)
                    else:
                        unit_inds = np.argsort(-coeffs)

            # NOTE: evaluate unpruned network as well (in case it is not biased)
            if j > 0:
                # Prune top salient units
                to_prune = unit_inds[start_ind:(start_ind + config['pruning']['step_size'])]
            else:
                to_prune = []

            if not config['pruning']['dynamic']:
                start_ind += config['pruning']['step_size']

            for _ in to_prune:
                pruned.append(_)
            pruned_inds.append(copy.deepcopy(pruned))
            # Prune the network
            model_pruned_ = prune_fc_units(model=model_pruned_, to_prune=to_prune, n_units=n_units, prune_first=True)

            # Predict on the validation set
            with torch.enable_grad():
                valid_pred_scores = model_pruned_(data.X_valid)[:, 0].reshape(-1, 1).detach().numpy()

            # Choose the best threshold w.r.t. the balanced accuracy on the held-out data
            best_thresh = choose_best_thresh_bal_acc(data=data, valid_pred_scores=valid_pred_scores)

            # Evaluate all metrics using the best threshold
            obj_dict = get_objective((valid_pred_scores > best_thresh) * 1., data.y_valid.numpy(), data.p_valid,
                                     config['metric'], config['objective']['sharpness'],
                                     config['objective']['epsilon'])

            objective.append(obj_dict['objective'])
            bias_metric.append(obj_dict['bias'])
            pred_performance.append(obj_dict['performance'])
            n_pruned.append(j)

            # Save the least biased model that satisfies the specified constraint on the performance
            if np.abs(obj_dict['bias']) < best_bias and obj_dict['performance'] >= config['pruning']['obj_lb']:
                best_bias = np.abs(obj_dict['bias'])
                j_best = len(objective) - 1

            if verbose:
                bar.update(bar_cnt)
                bar_cnt += 1

            # Stop pruning if accuracy drops below 52%
            if config['pruning']['stop_early'] and obj_dict['performance'] < 0.52:
                bar.finish()
                if config['acc_metric'] == 'f1_score':
                    print('\n' * 2)
                    print('WARNING: Early stopping does not support F1-score!')
                break

        if j_best == -1:
            print()
            print()
            print('WARNING: No debiased model satisfies the constraints!')
            j_best = 0

        # Plot performance traces
        if plot:
            plot_pruning_results(n_pruned=n_pruned, total_n_units=total_n_units, objective=objective,
                                 bias_metric=bias_metric, pred_performance=pred_performance, j_best=j_best,
                                 seed=seed, config=config, display=display)

        # Save performance traces
        save_pruning_trajectory(
            results={'objective': pred_performance * (np.abs(bias_metric) < config['objective']['epsilon']),
                     'bias': bias_metric,
                     'perf': pred_performance},
            seed=seed, config=config)

        # List of units pruned in the optimal model
        to_prune = np.array(pruned_inds[j_best])

        # Construct the best model
        with torch.no_grad():
            model_pruned = copy.deepcopy(model)
            model_pruned.eval()

            # Prune the final model
            model_pruned = prune_fc_units(model=model_pruned, to_prune=to_prune, n_units=n_units, prune_first=True)

    return model_pruned


def prune(model, layer_map, data_loader_train, data_loader_val, dataset_size_val, config, seed, device, arch='vgg',
          plot=False, display=False, verbose=1):
    """Intra-processing debiasing procedure for pruning neural networks with the given data loaders"""
    # NOTE: a function returning list of layers to be pruned needs to be passed as an argument
    # Suppress warnings
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    # Get a list of layers from the model to be pruned
    layers = layer_map(model)

    if verbose:
        print('Pruning the network...')
        print()

    # Create hooks to get layer activations from the model
    activation, handles = install_hooks(layers=layers)

    # Log the objectives throughout pruning
    objective = []
    bias_metric = []
    pred_performance = []
    n_pruned = []
    pruned_inds = []
    pruned = []

    j_best = -1
    best_bias = 1
    to_prune_best = None

    model.eval()

    # Evaluate the original model, in case it is unbiased
    with torch.no_grad():
        valid_pred_scores = np.zeros((dataset_size_val,))
        y_valid = np.zeros((dataset_size_val,))
        p_valid = np.zeros((dataset_size_val,))

        cnt = 0
        for X_, y_, p_ in data_loader_val:
            X_ = X_.to(device)
            y_ = y_.to(device).to(torch.float)
            p_ = p_.to(device)

            with torch.enable_grad():
                outputs = model(X_)

            valid_pred_scores[cnt * config['pruning']['batch_size']:(cnt + 1) * config['pruning']['batch_size']] = \
                outputs[:, 0].cpu().numpy()
            y_valid[cnt * config['pruning']['batch_size']:(cnt + 1) * config['pruning']['batch_size']] = \
                y_.cpu().numpy()
            p_valid[cnt * config['pruning']['batch_size']:(cnt + 1) * config['pruning']['batch_size']] = \
                p_.cpu().numpy()

            cnt += 1

        # Choose the best threshold w.r.t. the balanced accuracy on the held-out data
        best_thresh = choose_best_thresh_bal_acc_(y_valid=y_valid, valid_pred_scores=valid_pred_scores)

        obj_dict = get_test_objective_(y_pred=(valid_pred_scores > best_thresh) * 1., y_test=y_valid, p_test=p_valid,
                                       config=config)
        objective.append(obj_dict['objective'])
        bias_metric.append(obj_dict['bias'])
        pred_performance.append(obj_dict['performance'])
        n_pruned.append([0])
        pruned_inds.append([])

        # NOTE: this determines the order in which units are pruned, similar to the bias GD/A
        asc = obj_dict['bias'] < 0

    model.zero_grad()

    # Compute gradient-based bias influence of the units
    coeffs, n_structs, start_idx, end_idx = eval_saliency_dataloaders(
        model=model, layers=layers, data_loader=data_loader_train, activation=activation, device=device,
        config=config, pruned=None)

    # Sort units according to their influence
    # NOTE: order of pruning depends on the sign of the initial bias
    if asc:
        struct_inds = np.argsort(coeffs)
    else:
        struct_inds = np.argsort(-coeffs)

    # Construct a masked model which allows efficiently dropping out/pruning individual units
    if arch == 'vgg':
        model = ChestXRayVGG16Masked(base_model=model, prunable_layers=layers, start_idx=start_idx, end_idx=end_idx)
    elif arch == 'resnet':
        model = ChestXRayResNet18Masked(base_model=model, prunable_layers=layers, start_idx=start_idx, end_idx=end_idx)
    else:
        ValueError('ERROR: Network architecture not supported by pruning!')

    model.eval()

    # The actual pruning routine
    with torch.no_grad():
        if verbose:
            bar = progressbar.ProgressBar(maxval=int((len(struct_inds) + 1) / config['pruning']['step_size']))
            bar.start()
            bar_cnt = 0

        cnt = 0
        step_num = 0
        # Prune step_size units at a time, measuring performance at each sparsity level
        for j in range(0, len(struct_inds), config['pruning']['step_size']):
            # Recompute influence scores
            if config['pruning']['dynamic'] and j > 0:
                with torch.enable_grad():
                    coeffs, n_structs, start_idx, end_idx = eval_saliency_dataloaders(
                        model=model, layers=layers, data_loader=data_loader_train, activation=activation,
                        device=device, config=config, pruned=pruned)

                # Updated ordering of structures
                if asc:
                    struct_inds = np.argsort(coeffs)
                else:
                    struct_inds = np.argsort(-coeffs)

            # Choose top influential units
            to_prune = struct_inds[cnt:(cnt + config['pruning']['step_size'])]
            if not config['pruning']['dynamic']:
                cnt += config['pruning']['step_size']
            for _ in to_prune:
                pruned.append(_)
            pruned_inds.append(copy.deepcopy(pruned))

            # Evaluate the pruned model on validation data
            valid_pred_scores = np.zeros((dataset_size_val,))
            y_valid = np.zeros((dataset_size_val,))
            p_valid = np.zeros((dataset_size_val,))
            cnt_ = 0
            for X_, y_, p_ in data_loader_val:
                model.zero_grad()
                X_ = X_.to(device)
                y_ = y_.to(device).to(torch.float)
                p_ = p_.to(device)

                with torch.enable_grad():
                    outputs = model(X_, pruned=np.array(pruned))

                valid_pred_scores[cnt_ * config['pruning']['batch_size']:(cnt_ + 1) * config['pruning'][
                    'batch_size']] = outputs[:, 0].cpu().numpy()
                y_valid[cnt_ * config['pruning']['batch_size']:(cnt_ + 1) * config['pruning']['batch_size']] = \
                    y_.cpu().numpy()
                p_valid[cnt_ * config['pruning']['batch_size']:(cnt_ + 1) * config['pruning']['batch_size']] = \
                    p_.cpu().numpy()

                cnt_ += 1

            # Choose the best threshold w.r.t. the balanced accuracy on the held-out data
            best_thresh = choose_best_thresh_bal_acc_(y_valid=y_valid, valid_pred_scores=valid_pred_scores)

            obj_dict = get_test_objective_(y_pred=(valid_pred_scores > best_thresh) * 1., y_test=y_valid,
                                           p_test=p_valid, config=config)

            # Save the least biased model that satisfies the specified constraint on the performance
            if np.abs(obj_dict['bias']) < best_bias and obj_dict['performance'] >= config['pruning']['obj_lb']:
                best_bias = np.abs(obj_dict['bias'])
                j_best = len(objective) - 1
                to_prune_best = copy.deepcopy(pruned)

            # Stop pruning if the predictive performance drops below a certain level
            if config['pruning']['stop_early'] and obj_dict['performance'] <= 0.55:
                bar.finish()
                if config['acc_metric'] == 'f1_score':
                    print('\n' * 2)
                    print('WARNING: Early stopping does not support F1-score!')
                break

            # Stop pruning if a maximum number of pruning steps has been reached
            if config['pruning']['stop_early'] and step_num >= config['pruning']['max_steps']:
                bar.finish()
                break

            step_num += 1

            objective.append(obj_dict['objective'])
            bias_metric.append(obj_dict['bias'])
            pred_performance.append(obj_dict['performance'])
            n_pruned.append(j)

            if verbose:
                bar.update(bar_cnt)
                bar_cnt += 1

        if verbose:
            print('\n' * 2)

        # Plot performance traces
        if plot:
            plot_pruning_results(n_pruned=n_pruned, total_n_units=len(coeffs), objective=objective,
                                 bias_metric=bias_metric, pred_performance=pred_performance, j_best=j_best,
                                 seed=seed, config=config, display=display)

        # Save performance traces
        save_pruning_trajectory(
            results={'objective': pred_performance * (np.abs(bias_metric) < config['objective']['epsilon']),
                     'bias': bias_metric,
                     'perf': pred_performance},
            seed=seed, config=config)

        # List of units to be pruned
        to_prune = np.array(to_prune_best)

        model.eval()

        # Remove hooks from the model
        remove_all_forward_hooks(model)

        for k_h, h in handles.items():
            handles[k_h].remove()

        # Return the masked model and the list of units to be pruned
        return model, to_prune
