"""
Network architectures and training scheme for MIMIC-CXR.
"""
import time

import copy

import torch

import torch.nn.functional as F

import numpy as np

from utils.evaluation import compute_bias, compute_accuracy_metrics

from torch import nn

from torchvision import models


class ChestXRayResNet18(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()
        self.resnet18 = models.resnet18(pretrained=pretrained)
        self.out = nn.Linear(1000, 1)

    def forward(self, t):
        t = torch.relu(self.resnet18(t))
        return torch.sigmoid(self.out(t))

    def trunc_forward(self, t):
        t = torch.relu(self.resnet18(t))
        return self.out(t)


class ChestXRayVGG16(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()
        self.vgg16 = models.vgg16(pretrained=pretrained)
        self.out = nn.Linear(1000, 1)

    def forward(self, t):
        t = torch.relu(self.vgg16(t))
        return torch.sigmoid(self.out(t))

    def trunc_forward(self, t):
        t = torch.relu(self.vgg16(t))
        return self.out(t)


class ChestXRayVGG16Masked(nn.Module):

    def __init__(self, base_model: ChestXRayVGG16, prunable_layers, start_idx, end_idx):
        super().__init__()
        self.all_layers = get_children(base_model)
        self.prunable_layers = prunable_layers
        self.start_idx = start_idx
        self.end_idx = end_idx

    def forward(self, t, pruned=None):
        cnt = 0
        cnt_ = 0
        for (i, l) in enumerate(self.all_layers):
            # Flatten for FC layers
            if len(t.shape) > 2 and isinstance(l, nn.Linear):
                t = torch.flatten(t, 1)

            t = l(t)

            # If the layer is in the list, prune given units
            if l in self.prunable_layers:
                if pruned is not None:
                    pruned_structs = pruned[np.logical_and(self.start_idx[cnt] <= pruned,
                                                           pruned < self.end_idx[cnt])] - cnt_

                    if isinstance(l, nn.Linear):
                        t[:, pruned_structs] = 0

                        cnt_ += t.shape[1]
                        cnt += 1
                    elif isinstance(l, nn.Conv2d):
                        i_hat = (pruned_structs / (t.shape[2] * t.shape[3])).astype(int)
                        j_hat = ((pruned_structs - t.shape[2] * t.shape[3] * i_hat) / t.shape[3]).astype(int)
                        k_hat = pruned_structs - t.shape[2] * t.shape[3] * i_hat - j_hat * t.shape[3]

                        t[:, i_hat, j_hat, k_hat] = 0

                        cnt_ += t.shape[1] * t.shape[2] * t.shape[3]
                        cnt += 1
                    else:
                        pass

            # Apply ReLU activation to the VGG features
            if i == len(self.all_layers) - 2:
                t = torch.relu(t)

        return torch.sigmoid(t)

    def trunc_forward(self, t, pruned=None):
        cnt = 0
        cnt_ = 0

        for (i, l) in enumerate(self.all_layers):
            # Flatten for FC layers
            if len(t.shape) > 2 and isinstance(l, nn.Linear):
                t = torch.flatten(t, 1)

            t = l(t)

            # If the layer is in the list, prune given units
            if l in self.prunable_layers:
                if pruned is not None:
                    pruned_structs = pruned[np.logical_and(self.start_idx[cnt] <= pruned,
                                                           pruned < self.end_idx[cnt])] - cnt_

                    if isinstance(l, nn.Linear):
                        t[:, pruned_structs] = 0

                        cnt_ += t.shape[1]
                        cnt += 1
                    elif isinstance(l, nn.Conv2d):
                        i_hat = (pruned_structs / (t.shape[2] * t.shape[3])).astype(int)
                        j_hat = ((pruned_structs - t.shape[2] * t.shape[3] * i_hat) / t.shape[3]).astype(int)
                        k_hat = pruned_structs - t.shape[2] * t.shape[3] * i_hat - j_hat * t.shape[3]

                        t[:, i_hat, j_hat, k_hat] = 0

                        cnt_ += t.shape[1] * t.shape[2] * t.shape[3]
                        cnt += 1
                    else:
                        pass

            # Apply ReLU activation to the VGG features
            if i == len(self.all_layers) - 2:
                t = torch.relu(t)

        return t


def get_children(model: torch.nn.Module):
    # get children form model!
    children = list(model.children())
    flatt_children = []
    if children == []:
        # if model has no children; model is last child! :O
        return model
    else:
        # look for children from children... to the last child!
        for child in children:
            try:
                flatt_children.extend(get_children(child))
            except TypeError:
                flatt_children.append(get_children(child))
    return flatt_children


def get_layers_to_prune_ResNet18(model: ChestXRayResNet18):
    NotImplementedError('ResNet18 pruning not implemented yet!')


def get_layers_to_prune_VGG16(model: ChestXRayVGG16):
    # Return all conv layers
    layers = [model.vgg16.features[i] for i in range(len(model.vgg16.features)) if isinstance(model.vgg16.features[i],
                                                                                              nn.Conv2d)]
    return layers


def train_ChestXRay_model(dataloaders, dataset_sizes, model, criterion, optimizer, scheduler, device,
                          class_names=['Pneumothorax', 'No Finding'], bias_metric='spd', batch_size=256, num_epochs=25):
    since = time.time()

    # Best model parameters
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.Inf
    best_acc = 0
    best_bias = 0
    train_acc = []
    val_acc = []
    train_loss = []
    val_loss = []
    train_bias = []
    val_bias = []

    # Iterate over epochs
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            # Set the model to the training mode
            if phase == 'train':
                model.train()

            # Set model to the evaluation mode
            else:
                model.eval()

            # Running parameters
            running_loss = 0.0
            running_corrects = 0
            preds_vec = np.zeros(dataset_sizes[phase])
            labels_vec = np.zeros(dataset_sizes[phase])
            probs_mat = np.zeros((dataset_sizes[phase]))
            priv = np.zeros(dataset_sizes[phase])
            cnt = 0

            # Iterate over data
            for inputs, labels, attrs in dataloaders[phase]:
                # Send inputs and labels to the device
                inputs = inputs.to(device)
                labels = labels.to(device).to(torch.float)
                attrs = attrs.to(device)

                # Zero the gradients
                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    preds = outputs[:, 0] > 0.5
                    loss = criterion(outputs[:, 0], labels)
                    probs_mat[cnt * batch_size:(cnt + 1) * batch_size] = outputs[:, 0].cpu().detach().numpy()
                    preds_vec[cnt * batch_size:(cnt + 1) * batch_size] = preds.cpu().detach().numpy()
                    labels_vec[cnt * batch_size:(cnt + 1) * batch_size] = labels.cpu().detach().numpy()
                    priv[cnt * batch_size:(cnt + 1) * batch_size] = attrs.cpu().detach().numpy()

                    # Backward + optimize only if in the training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # Increment
                cnt = cnt + 1

            if phase == 'train':
                scheduler.step()

            # Get the predictive performance metrics
            auroc, avg_precision, balanced_acc, f1_acc = compute_accuracy_metrics(probs_mat, preds_vec, labels_vec,
                                                                                  class_names)
            if phase == 'val':
                bias = compute_bias(preds_vec, labels_vec, priv, bias_metric)
                print('Bias:', bias)
            acc = running_corrects.double() / dataset_sizes[phase]

            # Save
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = balanced_acc

            # Print
            print('{} Loss: {:.4f} Acc: {:.4f} AUROC: {:.4f} Avg. Precision: {:.4f} Balanced Accuracy: {:.4f} '
                  'f1 Score: {:.4f}'.format(phase, epoch_loss, acc, auroc, avg_precision, balanced_acc, f1_acc))

            # Save the accuracy and loss
            if phase == 'train':
                train_acc.append(epoch_acc)
                train_loss.append(epoch_loss)
            elif phase == 'val':
                val_acc.append(epoch_acc)
                val_loss.append(epoch_loss)

            # Deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_acc = epoch_acc
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

    # Time
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('Best val Bias: {:4f}'.format(best_bias))

    # Load the best model weights
    model.load_state_dict(best_model_wts)
    return model, train_acc, train_loss, val_acc, val_loss


class Critic(nn.Module):
    """Critic class for adversarial debiasing method"""

    def __init__(self, sizein, num_deep=3, hid=32):
        super().__init__()
        self.fc0 = nn.Linear(sizein, hid)
        self.fcs = nn.ModuleList([nn.Linear(hid, hid) for _ in range(num_deep)])
        self.dropout = nn.Dropout(0.2)
        self.out = nn.Linear(hid, 1)

    def forward(self, t):
        t = t.reshape(1, -1)
        t = self.fc0(t)
        for fully_connected in self.fcs:
            t = F.relu(fully_connected(t))
            t = self.dropout(t)
        return self.out(t)
