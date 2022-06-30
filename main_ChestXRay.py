"""
Main file to run MIMIC-CXR experiments.
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np

import torch

import yaml
import copy

import progressbar

import os

from aif360.algorithms.postprocessing import (EqOddsPostprocessing,
                                              RejectOptionClassification)
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score

from utils.evaluation import (get_valid_objective_, get_test_objective_,
                              eval_model_w_data_loaders, compute_empirical_bias)
from utils.data_utils import to_dataframe

from algorithms.pruning import prune
from algorithms.biasGrad import bias_gda_dataloaders
from algorithms.adversarial import (val_model_dataloaders, get_best_objective)
from algorithms.mitigating import train_fair_ChestXRay_model

from datasets.chestxray_dataset import get_ChestXRay_mimic_dataloaders

from models.networks_ChestXRay import (train_ChestXRay_model, ChestXRayResNet18, ChestXRayVGG16,
                                       get_layers_to_prune_ResNet18, get_layers_to_prune_VGG16, Critic)

from torch import optim
from torch import nn
from torch.optim import lr_scheduler

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

    # NOTE: replace with relevant directories
    if config['dataset'] == 'chestxray_mimic':
        ROOT_DIR = '...'
    else:
        NotImplementedError('This chest X-ray dataset not supported!')

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
        if config['dataset'] == 'chestxray_mimic':
            dataloaders, dataset_sizes = get_ChestXRay_mimic_dataloaders(device, root_dir=ROOT_DIR,
                                                                         prot_attr=config['protected'],
                                                                         priv_class=config['priv_class'],
                                                                         unpriv_class=config['unpriv_class'],
                                                                         train_prot_ratio=config['prot_ratio'],
                                                                         class_names=[config['disease'], 'No Finding'],
                                                                         batch_size=config['default']['batch_size'],
                                                                         num_workers=config['num_workers'], seed=seed)
        else:
            NotImplementedError('This chest X-ray dataset not supported!')

        # Get a pretrained model
        if config['default']['arch'] == 'resnet':
            model = ChestXRayResNet18(pretrained=config['default']['pretrained'])
        elif config['default']['arch'] == 'vgg':
            model = ChestXRayVGG16(pretrained=config['default']['pretrained'])
        else:
            ValueError('Network architecture not supported!')
        model = model.to(device)

        model_path = os.path.join('models', config['modelpath'] + str('_') + str(seed) + '.pt')
        if Path(model_path).is_file():
            logger.info(f'Loading Model from {model_path}.')
            model.load_state_dict(torch.load(model_path))
        else:
            logger.info(f'Training model from scratch.')

            optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-8)
            scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

            model, train_acc, train_loss, val_acc, val_loss = train_ChestXRay_model(
                dataloaders=dataloaders, dataset_sizes=dataset_sizes, model=model, criterion=nn.BCELoss(),
                optimizer=optimizer, scheduler=scheduler, device=device, class_names=[config['disease'], 'No Finding'],
                bias_metric=config['metric'], batch_size=config['default']['batch_size'],
                num_epochs=config['default']['n_epochs'])

            torch.save(model.state_dict(), model_path)

        # Preliminaries
        logger.info('Setting up preliminaries.')
        model.eval()

        results_valid = {}
        results_test = {}

        # Evaluate default model
        if 'default' in config['models']:
            logger.info('Finding best threshold for default model to minimize objective function')

            val_scores = np.zeros((dataset_sizes['val'],))
            val_labels = np.zeros((dataset_sizes['val'],))
            val_prot = np.zeros((dataset_sizes['val'],))

            test_scores = np.zeros((dataset_sizes['test'],))
            test_labels = np.zeros((dataset_sizes['test'],))
            test_prot = np.zeros((dataset_sizes['test'],))

            with torch.no_grad():
                cnt = 0
                for inputs, labels, attrs in dataloaders['val']:
                    inputs = inputs.to(device)
                    labels = labels.to(device).to(torch.float)
                    attrs = attrs.to(device)

                    outputs = model(inputs)

                    val_scores[
                    cnt * config['default']['batch_size']:(cnt + 1) * config['default']['batch_size']] = \
                        outputs[:, 0].cpu().numpy()
                    val_labels[cnt * config['default']['batch_size']:(cnt + 1) * config['default']['batch_size']] = \
                        labels.cpu().numpy()
                    val_prot[cnt * config['default']['batch_size']:(cnt + 1) * config['default']['batch_size']] = \
                        attrs.cpu().numpy()

                    cnt += 1

                cnt = 0
                for inputs, labels, attrs in dataloaders['test']:
                    # send inputs and labels to device
                    inputs = inputs.to(device)
                    labels = labels.to(device).to(torch.float)
                    attrs = attrs.to(device)

                    outputs = model(inputs)

                    test_scores[
                    cnt * config['default']['batch_size']:(cnt + 1) * config['default']['batch_size']] = \
                        outputs[:, 0].cpu().numpy()
                    test_labels[cnt * config['default']['batch_size']:(cnt + 1) * config['default']['batch_size']] = \
                        labels.cpu().numpy()
                    test_prot[cnt * config['default']['batch_size']:(cnt + 1) * config['default']['batch_size']] = \
                        attrs.cpu().numpy()

                    cnt += 1

            threshs = np.linspace(0, 1, 101)
            performances = []
            for thresh in threshs:
                if config['acc_metric'] == 'balanced_accuracy':
                    perf = balanced_accuracy_score(val_labels, val_scores > thresh)
                elif config['acc_metric'] == 'accuracy':
                    perf = accuracy_score(val_labels, val_scores > thresh)
                elif config['acc_metric'] == 'f1_score':
                    perf = f1_score(val_labels, val_scores > thresh)
                else:
                    print('Accuracy metric not defined')
                performances.append(perf)
            best_thresh = threshs[np.argmax(performances)]

            logger.info('Evaluating default model with best threshold.')
            results_valid['default'] = get_valid_objective_(y_pred=(val_scores > best_thresh), y_val=val_labels,
                                                            p_val=val_prot, config=config)
            results_test['default'] = get_test_objective_(y_pred=(test_scores > best_thresh), y_test=test_labels,
                                                          p_test=test_prot, config=config)
            logger.info(f'Results: {results_test["default"]}')

            # Get rid of data loaders to free up memory
            dataloaders = None
            dataset_sizes = None

        # Evaluate Random Debiasing
        if 'random' in config['models']:
            print('Perturbing the network randomly...')
            print()

            # Get data
            if config['dataset'] == 'chestxray_mimic':
                dataloaders_rand, dataset_sizes_rand = get_ChestXRay_mimic_dataloaders(
                    device, root_dir=ROOT_DIR, prot_attr=config['protected'], priv_class=config['priv_class'],
                    unpriv_class=config['unpriv_class'], train_prot_ratio=config['prot_ratio'],
                    class_names=[config['disease'], 'No Finding'], batch_size=config['default']['batch_size'],
                    num_workers=config['num_workers'], seed=seed)
            else:
                NotImplementedError('This chest X-ray dataset not supported!')

            rand_result = [-np.inf, None, -1]

            bar = progressbar.ProgressBar(maxval=config['random']['num_trials'])
            bar.start()
            bar_cnt = 0

            for iteration in range(config['random']['num_trials']):
                rand_model = copy.deepcopy(model)
                rand_model.to(device)
                # Perturb model's parameters
                for param in rand_model.parameters():
                    param.data = param.data * (torch.randn_like(param) * config['random']['stddev'] + 1)

                rand_model.eval()

                valid_pred_scores, y_valid, p_valid = eval_model_w_data_loaders(
                    model=rand_model, device=device, dataloader=dataloaders_rand['val'],
                    dataset_size=dataset_sizes_rand['val'], batch_size=config['default']['batch_size'])

                threshs = np.linspace(0, 1, 101)
                performances = []
                for thresh in threshs:
                    if config['acc_metric'] == 'balanced_accuracy':
                        perf = balanced_accuracy_score(y_valid, valid_pred_scores > thresh)
                    elif config['acc_metric'] == 'accuracy':
                        perf = accuracy_score(y_valid, valid_pred_scores > thresh)
                    elif config['acc_metric'] == 'f1_score':
                        perf = f1_score(y_valid, valid_pred_scores > thresh)
                    else:
                        print('Accuracy metric not defined')
                    performances.append(perf)
                best_thresh = threshs[np.argmax(performances)]
                best_obj = get_valid_objective_(y_pred=(valid_pred_scores > best_thresh), y_val=y_valid,
                                                p_val=p_valid, config=config)

                if best_obj['objective'] > rand_result[0]:
                    del rand_result[1]
                    rand_result = [best_obj['objective'], copy.deepcopy(rand_model.state_dict()), best_thresh]

                bar.update(bar_cnt)
                bar_cnt += 1

            print('\n' * 2)

            # Evaluate the best random model
            best_model = copy.deepcopy(model)
            best_model.load_state_dict(rand_result[1])
            best_model.to(device)
            best_thresh = rand_result[2]

            with torch.no_grad():
                valid_pred_scores, y_valid, p_valid = eval_model_w_data_loaders(
                    model=best_model, device=device, dataloader=dataloaders_rand['val'],
                    dataset_size=dataset_sizes_rand['val'], batch_size=config['default']['batch_size'])

                test_pred_scores, y_test, p_test = eval_model_w_data_loaders(
                    model=best_model, device=device, dataloader=dataloaders_rand['test'],
                    dataset_size=dataset_sizes_rand['test'], batch_size=config['default']['batch_size'])

            results_valid['random'] = get_valid_objective_(y_pred=(valid_pred_scores > best_thresh), y_val=y_valid,
                                                             p_val=p_valid, config=config)
            logger.info(f'Results validation: {results_valid["random"]}')
            results_test['random'] = get_test_objective_(y_pred=(test_pred_scores > best_thresh), y_test=y_test,
                                                           p_test=p_test, config=config)
            logger.info(f'Results test: {results_test["random"]}')

        # Evaluate Equality of Odds
        if 'ROC' in config['models']:
            metric_map = {
                'spd': 'Statistical parity difference',
                'aod': 'Average odds difference',
                'eod': 'Equal opportunity difference'
            }

            # Get data
            if config['dataset'] == 'chestxray_mimic':
                dataloaders_roc, dataset_sizes_roc = get_ChestXRay_mimic_dataloaders(
                    device, root_dir=ROOT_DIR, prot_attr=config['protected'], priv_class=config['priv_class'],
                    unpriv_class=config['unpriv_class'], train_prot_ratio=config['prot_ratio'],
                    class_names=[config['disease'], 'No Finding'], batch_size=config['default']['batch_size'],
                    num_workers=config['num_workers'], seed=seed)
            else:
                NotImplementedError('This chest X-ray dataset not supported!')

            ROC = RejectOptionClassification(unprivileged_groups=[{config['protected']: 1.}],
                                             privileged_groups=[{config['protected']: 0.}],
                                             low_class_thresh=0.01, high_class_thresh=0.99,
                                             num_class_thresh=100, num_ROC_margin=50,
                                             metric_name=metric_map[config['metric']],
                                             metric_ub=config['objective']['epsilon'],
                                             metric_lb=-config['objective']['epsilon'])

            with torch.no_grad():
                valid_pred_scores, y_valid, p_valid = eval_model_w_data_loaders(
                    model=model, device=device, dataloader=dataloaders_roc['val'],
                    dataset_size=dataset_sizes_roc['val'], batch_size=config['default']['batch_size'])

                test_pred_scores, y_test, p_test = eval_model_w_data_loaders(
                    model=model, device=device, dataloader=dataloaders_roc['test'],
                    dataset_size=dataset_sizes_roc['test'], batch_size=config['default']['batch_size'])

                threshs = np.linspace(0, 1, 101)
                performances = []
                for thresh in threshs:
                    if config['acc_metric'] == 'balanced_accuracy':
                        perf = balanced_accuracy_score(y_valid, valid_pred_scores > thresh)
                    elif config['acc_metric'] == 'accuracy':
                        perf = accuracy_score(y_valid, valid_pred_scores > thresh)
                    elif config['acc_metric'] == 'f1_score':
                        perf = f1_score(y_valid, valid_pred_scores > thresh)
                    else:
                        print('Accuracy metric not defined')
                    performances.append(perf)
                best_thresh = threshs[np.argmax(performances)]

                val_dataset = to_dataframe(y_true=y_valid, y_pred=valid_pred_scores, y_prot=p_valid,
                                           prot_name=config['protected'])
                val_dataset_pred = to_dataframe(y_true=(valid_pred_scores > best_thresh) * 1., y_pred=valid_pred_scores,
                                                y_prot=p_valid, prot_name=config['protected'])
                test_dataset = to_dataframe(y_true=y_test, y_pred=test_pred_scores, y_prot=p_test,
                                            prot_name=config['protected'])
                test_dataset_pred = to_dataframe(y_true=(test_pred_scores > best_thresh) * 1., y_pred=test_pred_scores,
                                                 y_prot=p_test, prot_name=config['protected'])

                logger.info("Training ROC model with validation dataset.")
                ROC = ROC.fit(val_dataset, val_dataset)

                logger.info("Evaluating ROC model.")

                logger.info('ROC val results')
                val_y_pred = ROC.predict(val_dataset_pred).labels.reshape(-1)
                results_valid['ROC'] = get_valid_objective_(y_pred=val_y_pred, y_val=y_valid, p_val=p_valid,
                                                               config=config)
                logger.info(f'Results validation: {results_valid["ROC"]}')

                logger.info('ROC test results')
                test_y_pred = ROC.predict(test_dataset_pred).labels.reshape(-1)
                results_test['ROC'] = get_test_objective_(y_pred=test_y_pred, y_test=y_test, p_test=p_test,
                                                             config=config)
                logger.info(f'Results test: {results_test["ROC"]}')

                # Get rid of data loaders to free up memory
                dataloaders_roc = None
                dataset_sizes_roc = None
                ROC = None

        # Evaluate Equality of Odds
        if 'EqOdds' in config['models']:
            # Get data
            if config['dataset'] == 'chestxray_mimic':
                dataloaders_eo, dataset_sizes_eo = get_ChestXRay_mimic_dataloaders(
                    device, root_dir=ROOT_DIR, prot_attr=config['protected'], priv_class=config['priv_class'],
                    unpriv_class=config['unpriv_class'], train_prot_ratio=config['prot_ratio'],
                    class_names=[config['disease'], 'No Finding'], batch_size=config['default']['batch_size'],
                    num_workers=config['num_workers'], seed=seed)
            else:
                NotImplementedError('This chest X-ray dataset not supported!')

            eo = EqOddsPostprocessing(privileged_groups=[{config['protected']: 1.}],
                                      unprivileged_groups=[{config['protected']: 0.}])

            with torch.no_grad():
                valid_pred_scores, y_valid, p_valid = eval_model_w_data_loaders(
                    model=model, device=device, dataloader=dataloaders_eo['val'],
                    dataset_size=dataset_sizes_eo['val'], batch_size=config['default']['batch_size'])

                test_pred_scores, y_test, p_test = eval_model_w_data_loaders(
                    model=model, device=device, dataloader=dataloaders_eo['test'],
                    dataset_size=dataset_sizes_eo['test'], batch_size=config['default']['batch_size'])

            threshs = np.linspace(0, 1, 101)
            performances = []
            for thresh in threshs:
                if config['acc_metric'] == 'balanced_accuracy':
                    perf = balanced_accuracy_score(y_valid, valid_pred_scores > thresh)
                elif config['acc_metric'] == 'accuracy':
                    perf = accuracy_score(y_valid, valid_pred_scores > thresh)
                elif config['acc_metric'] == 'f1_score':
                    perf = f1_score(y_valid, valid_pred_scores > thresh)
                else:
                    print('Accuracy metric not defined')
                performances.append(perf)
            best_thresh = threshs[np.argmax(performances)]

            val_dataset = to_dataframe(y_true=y_valid, y_pred=valid_pred_scores, y_prot=p_valid,
                                       prot_name=config['protected'])
            val_dataset_pred = to_dataframe(y_true=(valid_pred_scores > best_thresh) * 1., y_pred=valid_pred_scores,
                                            y_prot=p_valid, prot_name=config['protected'])
            test_dataset = to_dataframe(y_true=y_test, y_pred=test_pred_scores, y_prot=p_test,
                                        prot_name=config['protected'])
            test_dataset_pred = to_dataframe(y_true=(test_pred_scores > best_thresh) * 1., y_pred=test_pred_scores,
                                             y_prot=p_test, prot_name=config['protected'])

            logger.info("Training Equality of Odds model with validation dataset.")
            eo = eo.fit(val_dataset, val_dataset_pred)

            logger.info("Evaluating Equality of Odds model.")

            logger.info('Equality of Odds val results')
            val_y_pred = eo.predict(val_dataset_pred).labels.reshape(-1)
            results_valid['EqOdds'] = get_valid_objective_(y_pred=val_y_pred, y_val=y_valid, p_val=p_valid,
                                                           config=config)
            logger.info(f'Results validation: {results_valid["EqOdds"]}')

            logger.info('Equality of Odds test results')
            test_y_pred = eo.predict(test_dataset_pred).labels.reshape(-1)
            results_test['EqOdds'] = get_test_objective_(y_pred=test_y_pred, y_test=y_test, p_test=p_test,
                                                         config=config)
            logger.info(f'Results test: {results_test["EqOdds"]}')

            # Get rid of data loaders to free up memory
            dataloaders_eo = None
            dataset_sizes_eo = None
            eo = None

        if 'adversarial' in config['models']:
            # Get data
            if config['dataset'] == 'chestxray_mimic':
                dataloaders_adv, dataset_sizes_adv = get_ChestXRay_mimic_dataloaders(
                    device, root_dir=ROOT_DIR, prot_attr=config['protected'], priv_class=config['priv_class'],
                    unpriv_class=config['unpriv_class'], train_prot_ratio=config['prot_ratio'],
                    class_names=[config['disease'], 'No Finding'], batch_size=config['adversarial']['batch_size'],
                    num_workers=config['num_workers'], seed=seed)
            else:
                NotImplementedError('This chest X-ray dataset not supported!')

            base_model = copy.deepcopy(model.vgg16)
            base_model.classifier[-1] = nn.Linear(base_model.classifier[-1].in_features,
                                                  base_model.classifier[-1].in_features)

            actor = nn.Sequential(base_model, nn.Linear(base_model.classifier[-1].in_features, 2))
            actor.to(device)
            actor_optimizer = optim.Adam(actor.parameters(), lr=config['adversarial']['lr'])
            actor_loss_fn = nn.BCEWithLogitsLoss()
            actor_loss = 0.
            actor_steps = config['adversarial']['actor_steps']

            critic = Critic(config['adversarial']['batch_size'] * base_model.classifier[-1].in_features)
            critic.to(device)
            critic_optimizer = optim.Adam(critic.parameters(), lr=1e-4)
            critic_loss_fn = nn.MSELoss()
            critic_loss = 0.
            critic_steps = config['adversarial']['critic_steps']

            for epoch in range(config['adversarial']['epochs']):
                for param in critic.parameters():
                    param.requires_grad = True
                for param in actor.parameters():
                    param.requires_grad = False
                actor.eval()
                critic.train()
                for step, (X, y, p) in enumerate(dataloaders_adv['val']):
                    if step > critic_steps:
                        break
                    X, y, p = X.to(device), y.to(device), p.to(device)
                    if X.size(0) != config['adversarial']['batch_size']:
                        continue

                    critic_optimizer.zero_grad()

                    with torch.no_grad():
                        y_pred = actor(X)

                    y_true = y.float().to(device)
                    y_prot = p.float().to(device)

                    bias = compute_empirical_bias(y_pred, y_true, y_prot, config['metric'])
                    res = critic(base_model(X))
                    loss = critic_loss_fn(bias.unsqueeze(0), res[0])
                    loss.backward()
                    critic_loss += loss.item()
                    critic_optimizer.step()
                    if step % 100 == 0:
                        print_loss = critic_loss if (epoch * critic_steps + step) == 0 else critic_loss / (
                                    epoch * critic_steps + step)
                        logger.info(f'=======> Epoch: {(epoch, step)} Critic loss: {print_loss:.3f}')

                for param in critic.parameters():
                    param.requires_grad = False
                for param in actor.parameters():
                    param.requires_grad = True
                actor.train()
                critic.eval()
                for step, (X, y, p) in enumerate(dataloaders_adv['val']):
                    if step > actor_steps:
                        break
                    X, y, p = X.to(device), y.to(device), p.to(device)
                    if X.size(0) != config['adversarial']['batch_size']:
                        continue
                    actor_optimizer.zero_grad()

                    y_true = y.float().to(device)
                    y_prot = p.float().to(device)

                    est_bias = critic(base_model(X))
                    loss = actor_loss_fn(actor(X)[:, 0], y_true)

                    loss = max(1, config['adversarial']['lambda'] * (
                                abs(est_bias) - config['objective']['epsilon'] + config['adversarial'][
                            'margin']) + 1) * loss

                    loss.backward()
                    actor_loss += loss.item()
                    actor_optimizer.step()
                    if step % 100 == 0:
                        print_loss = critic_loss if (epoch * actor_steps + step) == 0 else critic_loss / (
                                    epoch * actor_steps + step)
                        logger.info(f'=======> Epoch: {(epoch, step)} Actor loss: {print_loss:.3f}')

            _, best_thresh = val_model_dataloaders(actor, dataloaders_adv['val'], get_best_objective, device, config)
            best_thresh = best_thresh.cpu().numpy()

            with torch.no_grad():
                valid_pred_scores, y_valid, p_valid = eval_model_w_data_loaders(
                    model=actor, device=device, dataloader=dataloaders_adv['val'],
                    dataset_size=dataset_sizes_adv['val'], batch_size=config['adversarial']['batch_size'])

                test_pred_scores, y_test, p_test = eval_model_w_data_loaders(
                    model=actor, device=device, dataloader=dataloaders_adv['test'],
                    dataset_size=dataset_sizes_adv['test'], batch_size=config['adversarial']['batch_size'])

            results_valid['adversarial'] = get_valid_objective_(y_pred=(valid_pred_scores > best_thresh), y_val=y_valid,
                                                             p_val=p_valid, config=config)
            logger.info(f'Results validation: {results_valid["adversarial"]}')
            results_test['adversarial'] = get_test_objective_(y_pred=(test_pred_scores > best_thresh), y_test=y_test,
                                                           p_test=p_test, config=config)
            logger.info(f'Results test: {results_test["adversarial"]}')

            # Get rid of data loaders to free up memory
            dataloaders_adv = None
            dataset_sizes_adv = None

        if 'mitigating' in config['models']:
            # Get data
            if config['dataset'] == 'chestxray_mimic':
                dataloaders_mit, dataset_sizes_mit = get_ChestXRay_mimic_dataloaders(
                    device, root_dir=ROOT_DIR, prot_attr=config['protected'],
                    priv_class=config['priv_class'], unpriv_class=config['unpriv_class'],
                    train_prot_ratio=config['prot_ratio'], class_names=[config['disease'], 'No Finding'],
                    batch_size=config['mitigating']['batch_size'], num_workers=config['num_workers'], seed=seed)
            else:
                NotImplementedError('This chest X-ray dataset not supported!')

            model_, _, __, ___, ____ = train_fair_ChestXRay_model(
                dataloaders=dataloaders_mit, dataset_sizes=dataset_sizes_mit, device=device, config=config,
                bias_metric='eod', batch_size=config['mitigating']['batch_size'],
                num_epochs=config['mitigating']['n_epochs'])

            with torch.no_grad():
                valid_pred_scores, y_valid, p_valid = eval_model_w_data_loaders(
                    model=model_, device=device, dataloader=dataloaders_mit['val'],
                    dataset_size=dataset_sizes_mit['val'], batch_size=config['mitigating']['batch_size'])

                test_pred_scores, y_test, p_test = eval_model_w_data_loaders(
                    model=model_, device=device, dataloader=dataloaders_mit['test'],
                    dataset_size=dataset_sizes_mit['test'], batch_size=config['mitigating']['batch_size'])

                logger.info('Finding best threshold for debiased model to minimize objective function')
                threshs = np.linspace(0, 1, 101)
                performances = []
                for thresh in threshs:
                    if config['acc_metric'] == 'balanced_accuracy':
                        perf = balanced_accuracy_score(y_valid, valid_pred_scores > thresh)
                    elif config['acc_metric'] == 'accuracy':
                        perf = accuracy_score(y_valid, valid_pred_scores > thresh)
                    elif config['acc_metric'] == 'f1_score':
                        perf = f1_score(y_valid, valid_pred_scores > thresh)
                    else:
                        print('Accuracy metric not defined')
                    performances.append(perf)
                best_thresh = threshs[np.argmax(performances)]

                logger.info('Evaluating debiased model with best threshold.')
                results_valid['mitigating'] = get_valid_objective_(y_pred=(valid_pred_scores > best_thresh),
                                                                   y_val=y_valid, p_val=p_valid, config=config)
                logger.info(f'Results validation: {results_valid["mitigating"]}')

                results_test['mitigating'] = get_test_objective_(y_pred=(test_pred_scores > best_thresh), y_test=y_test,
                                                                 p_test=p_test, config=config)
                logger.info(f'Results test: {results_test["mitigating"]}')

                # Get rid of data loaders to free up memory
                dataloaders_mit = None
                dataset_sizes_mit = None

        if 'biasGrad' in config['models']:

            # Get data for bias GD/A
            if config['dataset'] == 'chestxray_mimic':
                dataloaders_bgda, dataset_sizes_bgda = get_ChestXRay_mimic_dataloaders(
                    device, root_dir=ROOT_DIR, prot_attr=config['protected'],
                    priv_class=config['priv_class'], unpriv_class=config['unpriv_class'],
                    train_prot_ratio=config['prot_ratio'], class_names=[config['disease'], 'No Finding'],
                    batch_size=config['biasGrad']['batch_size'], num_workers=config['num_workers'], seed=seed)
            else:
                NotImplementedError('This chest X-ray dataset not supported!')

            opt_alg = optim.Adam
            model_ = bias_gda_dataloaders(model=model, data_loader_train=dataloaders_bgda['val'],
                                          data_loader_val=dataloaders_bgda['val'],
                                          dataset_size_val=dataset_sizes_bgda['val'],
                                          opt_alg=opt_alg, device=device, config=config, seed=seed,
                                          plot=True, display=False, verbose=1)

            with torch.no_grad():
                valid_pred_scores, y_valid, p_valid = eval_model_w_data_loaders(
                    model=model_, device=device, dataloader=dataloaders_bgda['val'],
                    dataset_size=dataset_sizes_bgda['val'], batch_size=config['biasGrad']['batch_size'])

                test_pred_scores, y_test, p_test = eval_model_w_data_loaders(
                    model=model_, device=device, dataloader=dataloaders_bgda['test'],
                    dataset_size=dataset_sizes_bgda['test'], batch_size=config['biasGrad']['batch_size'])

                logger.info('Finding best threshold for pruned model to minimize objective function')
                threshs = np.linspace(0, 1, 101)
                performances = []
                for thresh in threshs:
                    if config['acc_metric'] == 'balanced_accuracy':
                        perf = balanced_accuracy_score(y_valid, valid_pred_scores > thresh)
                    elif config['acc_metric'] == 'accuracy':
                        perf = accuracy_score(y_valid, valid_pred_scores > thresh)
                    elif config['acc_metric'] == 'f1_score':
                        perf = f1_score(y_valid, valid_pred_scores > thresh)
                    else:
                        print('Accuracy metric not defined')
                    performances.append(perf)
                best_thresh = threshs[np.argmax(performances)]

            logger.info('Evaluating debiased model with best threshold.')
            results_valid['biasGrad'] = get_valid_objective_(y_pred=(valid_pred_scores > best_thresh), y_val=y_valid,
                                                             p_val=p_valid, config=config)
            logger.info(f'Results validation: {results_valid["biasGrad"]}')
            results_test['biasGrad'] = get_test_objective_(y_pred=(test_pred_scores > best_thresh), y_test=y_test,
                                                           p_test=p_test, config=config)
            logger.info(f'Results test: {results_test["biasGrad"]}')

            # Get rid of data loaders to free up memory
            dataloaders_bgda = None
            dataset_sizes_bgda = None

        # NOTE: needs to be run the last, makes adjustments to the model object directly (to use less memory)
        # Evaluate pruned model
        if 'pruning' in config['models']:
            # Get data for pruning
            if config['dataset'] == 'chestxray_mimic':
                dataloaders_pruning, dataset_sizes_pruning = get_ChestXRay_mimic_dataloaders(
                    device, root_dir=ROOT_DIR, prot_attr=config['protected'],
                    priv_class=config['priv_class'], unpriv_class=config['unpriv_class'],
                    train_prot_ratio=config['prot_ratio'], class_names=[config['disease'], 'No Finding'],
                    batch_size=config['pruning']['batch_size'], num_workers=config['num_workers'], seed=seed)
            else:
                NotImplementedError('This chest X-ray dataset not supported!')

            layer_map = None
            if config['default']['arch'] == 'resnet':
                layer_map = get_layers_to_prune_ResNet18
            elif config['default']['arch'] == 'vgg':
                layer_map = get_layers_to_prune_VGG16
            else:
                ValueError('Network architecture not supported!')

            model_pruned, pruned = prune(model, layer_map, dataloaders_pruning['val'], dataloaders_pruning['val'],
                                         dataset_sizes_pruning['val'], config, seed, device, plot=False, display=False,
                                         verbose=1)

            with torch.no_grad():
                valid_pred_scores, y_valid, p_valid = eval_model_w_data_loaders(
                    model=model_pruned, device=device, dataloader=dataloaders_pruning['val'],
                    dataset_size=dataset_sizes_pruning['val'], batch_size=config['pruning']['batch_size'],
                    forward_args=[pruned])

                test_pred_scores, y_test, p_test = eval_model_w_data_loaders(
                    model=model_pruned, device=device, dataloader=dataloaders_pruning['test'],
                    dataset_size=dataset_sizes_pruning['test'], batch_size=config['pruning']['batch_size'],
                    forward_args=[pruned])

                logger.info('Finding best threshold for pruned model to minimize objective function')
                threshs = np.linspace(0, 1, 101)
                performances = []
                for thresh in threshs:
                    if config['acc_metric'] == 'balanced_accuracy':
                        perf = balanced_accuracy_score(y_valid, valid_pred_scores > thresh)
                    elif config['acc_metric'] == 'accuracy':
                        perf = accuracy_score(y_valid, valid_pred_scores > thresh)
                    elif config['acc_metric'] == 'f1_score':
                        perf = f1_score(y_valid, valid_pred_scores > thresh)
                    else:
                        print('Accuracy metric not defined')
                    performances.append(perf)
                best_thresh = threshs[np.argmax(performances)]

            logger.info('Evaluating debiased model with best threshold.')
            results_valid['pruning'] = get_valid_objective_(y_pred=(valid_pred_scores > best_thresh), y_val=y_valid,
                                                            p_val=p_valid, config=config)
            logger.info(f'Results validation: {results_valid["pruning"]}')
            results_test['pruning'] = get_test_objective_(y_pred=(test_pred_scores > best_thresh), y_test=y_test,
                                                          p_test=p_test, config=config)
            logger.info(f'Results test: {results_test["pruning"]}')

            # Get rid of data loaders to free up memory
            dataloaders_pruning = None
            dataset_sizes_pruning = None

        # Save Results
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
