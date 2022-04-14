"""
Adversarial fine-tuning algorithm.
"""
import logging

import math

import numpy as np
import torch
import torch.optim as optim

from models.networks_tabular import load_model, Critic
from utils.evaluation import get_best_thresh, get_test_objective, get_valid_objective, compute_bias

logger = logging.getLogger("Debiasing")


def val_model_dataloaders(model, loader, criterion, device, config):
    """Validate model on loader with criterion function"""
    y_true, y_pred, y_prot = [], [], []
    model.eval()
    with torch.no_grad():
        for X, y, p in loader:
            X, y, p = X.to(device), y.float().to(device), p.float().to(device)
            y_true.append(y)
            y_prot.append(p)
            y_pred.append(torch.sigmoid(model(X)[:, 0]))
    y_true, y_pred, y_prot = torch.cat(y_true), torch.cat(y_pred), torch.cat(y_prot)
    return criterion(y_true, y_pred, y_prot, config)


def get_best_objective(y_true, y_pred, y_prot, config):
    """Find the threshold for the best objective"""
    num_samples = 5
    threshs = torch.linspace(0, 1, 101)
    best_obj, best_thresh = -math.inf, 0.
    for thresh in threshs:
        indices = np.random.choice(np.arange(y_pred.size()[0]), num_samples*y_pred.size()[0], replace=True).reshape(num_samples, y_pred.size()[0])
        objs = []
        for index in indices:
            y_pred_tmp = y_pred[index]
            y_true_tmp = y_true[index]
            y_prot_tmp = y_prot[index]
            perf = (torch.mean((y_pred_tmp > thresh)[y_true_tmp.type(torch.bool)].type(torch.float32)) + torch.mean((y_pred_tmp <= thresh)[~y_true_tmp.type(torch.bool)].type(torch.float32))) / 2
            bias = compute_bias((y_pred_tmp > thresh).float().cpu(), y_true_tmp.float().cpu(), y_prot_tmp.float().cpu(), config['metric'])
            objs.append(compute_objective(perf, bias))
        obj = float(torch.tensor(objs).mean())
        if obj > best_obj:
            best_obj, best_thresh = obj, thresh

    return best_obj, best_thresh


def compute_objective(performance, bias, epsilon=0.05, margin=0.01):
    if abs(bias) <= (epsilon-margin):
        return performance
    else:
        return 0.0


def adversarial_debiasing(model_state_dict, data, config, device):
    logger.info('Training Adversarial model.')
    actor = load_model(data.num_features, config.get('hyperparameters', {}))
    actor.load_state_dict(model_state_dict)
    actor.to(device)
    hid = config['hyperparameters']['hid'] if 'hyperparameters' in config else 32
    critic = Critic(hid * config['adversarial']['batch_size'], num_deep=config['adversarial']['num_deep'], hid=hid)
    critic.to(device)
    critic_optimizer = optim.Adam(critic.parameters())
    critic_loss_fn = torch.nn.MSELoss()

    actor_optimizer = optim.Adam(actor.parameters(), lr=config['adversarial']['lr'])
    actor_loss_fn = torch.nn.BCELoss()

    for epoch in range(config['adversarial']['epochs']):
        for param in critic.parameters():
            param.requires_grad = True
        for param in actor.parameters():
            param.requires_grad = False
        actor.eval()
        critic.train()
        for step in range(config['adversarial']['critic_steps']):
            critic_optimizer.zero_grad()
            indices = torch.randint(0, data.X_valid.size(0), (config['adversarial']['batch_size'],))
            cX_valid = data.X_valid_gpu[indices]
            cy_valid = data.y_valid[indices]
            cp_valid = data.p_valid[indices]
            with torch.no_grad():
                scores = actor(cX_valid)[:, 0].reshape(-1).cpu().numpy()

            bias = compute_bias(scores, cy_valid.numpy(), cp_valid, config['metric'])

            res = critic(actor.trunc_forward(cX_valid))
            loss = critic_loss_fn(torch.tensor([bias], device=device).float(), res[0])
            loss.backward()
            train_loss = loss.item()
            critic_optimizer.step()
            if (epoch % 10 == 0) and (step % 100 == 0):
                logger.info(f'=======> Critic Epoch: {(epoch, step)} loss: {train_loss}')

        for param in critic.parameters():
            param.requires_grad = False
        for param in actor.parameters():
            param.requires_grad = True
        actor.train()
        critic.eval()
        for step in range(config['adversarial']['actor_steps']):
            actor_optimizer.zero_grad()
            indices = torch.randint(0, data.X_valid.size(0), (config['adversarial']['batch_size'],))
            cy_valid = data.y_valid_gpu[indices]
            cX_valid = data.X_valid_gpu[indices]

            pred_bias = critic(actor.trunc_forward(cX_valid))
            bceloss = actor_loss_fn(actor(cX_valid)[:, 0], cy_valid)

            # loss = lam*abs(pred_bias) + (1-lam)*loss
            objloss = max(1, config['adversarial']['lambda']*(abs(pred_bias[0][0])-config['objective']['epsilon']+config['adversarial']['margin'])+1) * bceloss

            objloss.backward()
            train_loss = objloss.item()
            actor_optimizer.step()
            if (epoch % 10 == 0) and (step % 100 == 0):
                logger.info(f'=======> Actor Epoch: {(epoch, step)} loss: {train_loss}')

        if epoch % 1 == 0:
            with torch.no_grad():
                scores = actor(data.X_valid_gpu)[:, 0].reshape(-1, 1).cpu().numpy()
                _, best_adv_obj = get_best_thresh(scores, np.linspace(0, 1, 101), data, config, valid=False,
                                                  margin=config['adversarial']['margin'])
                logger.info(f'Objective: {best_adv_obj}')

    logger.info('Finding optimal threshold for Adversarial model.')
    with torch.no_grad():
        scores = actor(data.X_valid_gpu)[:, 0].reshape(-1, 1).cpu().numpy()

    best_adv_thresh, _ = get_best_thresh(scores, np.linspace(0, 1, 101), data, config, valid=False, margin=config['adversarial']['margin'])

    logger.info('Evaluating Adversarial model on best threshold.')
    with torch.no_grad():
        labels = (actor(data.X_valid_gpu)[:, 0] > best_adv_thresh).reshape(-1, 1).cpu().numpy()
    results_valid = get_valid_objective(labels, data, config)
    logger.info(f'Results: {results_valid}')

    with torch.no_grad():
        labels = (actor(data.X_test_gpu)[:, 0] > best_adv_thresh).reshape(-1, 1).cpu().numpy()
    results_test = get_test_objective(labels, data, config)

    return results_valid, results_test
