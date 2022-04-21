"""
Utility functions for plotting.
"""
import os


import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rc

CB_COLOR_CYCLE = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']


def plotting_setup(font_size=12):
    # plot settings
    plt.style.use("seaborn-colorblind")
    plt.rcParams['font.size'] = font_size
    rc('text', usetex=False)
    plt.rcParams["font.family"] = "Times New Roman"
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})


def plot_pruning_trajectories_ave(log_dir: str, exp_name: str, seeds: list, n_steps: int, bias_name: str,
                                  perf_name: str, dir: str = None, legend: bool = False, font_size: int = 16):
    n_runs = len(seeds)
    objectives = np.zeros((n_runs, n_steps))
    biases = np.zeros((n_runs, n_steps))
    perfs = np.zeros((n_runs, n_steps))
    for i in range(n_runs):
        traj_i = np.loadtxt(fname=os.path.join(log_dir, exp_name + '_' + str(seeds[i]) + '_trajectory.csv'))
        objectives[i, 0:min(n_steps, traj_i.shape[0])] = traj_i[0:min(n_steps, traj_i.shape[0]), 0]
        biases[i, 0:min(n_steps, traj_i.shape[0])] = traj_i[0:min(n_steps, traj_i.shape[0]), 1]
        perfs[i, 0:min(n_steps, traj_i.shape[0])] = traj_i[0:min(n_steps, traj_i.shape[0]), 2]

    plot_curves_with_ci(xs=[np.arange(1, n_steps + 1), np.arange(1, n_steps + 1)],
                        avgs=[np.median(biases, axis=0), np.median(perfs, axis=0)],
                        lower=[np.quantile(biases, q=0.25, axis=0), np.quantile(perfs, q=0.25, axis=0)],
                        upper=[np.quantile(biases, q=0.75, axis=0), np.quantile(perfs, q=0.75, axis=0)],
                        labels=[bias_name, perf_name], xlab='# Units Pruned', ylab='',
                        font_size=font_size,
                        baseline=0, baseline_lab=None, baseline_cl='grey', dir=dir,
                        legend=legend, legend_outside=True, cls=CB_COLOR_CYCLE[1:3])


def plot_pruning_trajectories_multi(log_dir: str, exp_name: str, seeds: list, n_steps: int, bias_name: str,
                                    perf_name: str, dir: str = None, legend: bool = False, legend_outside: bool = True,
                                    font_size: int = 16, xlab: str = None):
    n_runs = len(seeds)
    objectives = np.ones((n_runs, n_steps)) * 0.5
    biases = np.zeros((n_runs, n_steps))
    perfs = np.ones((n_runs, n_steps)) * 0.5
    for i in range(n_runs):
        traj_i = np.loadtxt(fname=os.path.join(log_dir, exp_name + '_' + str(seeds[i]) + '_trajectory.csv'))
        objectives[i, 0:min(n_steps, traj_i.shape[0])] = traj_i[0:min(n_steps, traj_i.shape[0]), 0]
        biases[i, 0:min(n_steps, traj_i.shape[0])] = traj_i[0:min(n_steps, traj_i.shape[0]), 1]
        perfs[i, 0:min(n_steps, traj_i.shape[0])] = traj_i[0:min(n_steps, traj_i.shape[0]), 2]

    plotting_setup(font_size)

    fig = plt.figure(figsize=(8, 6))

    cls = CB_COLOR_CYCLE[:3]

    plt.axhline(0, label=None, c='gray', linestyle='--')

    for i in range(n_runs):
        plt.plot(np.arange(1, n_steps+1), biases[i, :], color=cls[1], label=None, marker=None, alpha=0.25)
    plt.plot(np.arange(1, n_steps+1), np.median(biases, axis=0), color=cls[1], label=bias_name, linewidth=5,
             linestyle='-.')

    for i in range(n_runs):
        plt.plot(np.arange(1, n_steps+1), perfs[i, :], color=cls[2], label=None, marker=None, alpha=0.25)
    plt.plot(np.arange(1, n_steps+1), np.median(perfs, axis=0), color=cls[2], label=perf_name,
             linewidth=5, linestyle=':')

    if xlab is None:
        plt.xlabel('# Units Pruned')
    else:
        plt.xlabel(xlab)
    plt.ylabel('')

    plt.xticks(np.linspace(0, n_steps, 5))
    plt.yticks(np.round(np.linspace(-0.4, 0.6, 6), 1))

    if legend:
        if legend_outside:
            leg = plt.legend(loc='upper right', bbox_to_anchor=(-0.15, 1))
        else:
            leg = plt.legend(loc='lower left')

        # Change the marker size manually for both lines
        for i in range(len(leg.legendHandles)):
            leg.legendHandles[i]._legmarker.set_markersize(16)
            leg.legendHandles[i].set_linewidth(5.0)

    if dir is not None:
        plt.savefig(fname=dir, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_curves_with_ci(xs, avgs, lower, upper, labels, xlab, ylab, font_size=16, baseline=None, baseline_lab=None,
                        baseline_cl=None, dir=None, legend=True, legend_outside=True, cls=None):
    plotting_setup(font_size)

    fig = plt.figure()

    ms = ['D', 'o', '^', 'v', 's', 'X', '*']

    if cls is None:
        cls = CB_COLOR_CYCLE[:len(xs)]

    if baseline_cl is None:
        baseline_cl = 'red'

    if baseline is not None:
        plt.axhline(baseline, label=baseline_lab, c=baseline_cl, linestyle='--')

    for i in range(len(xs)):
        upper_i = upper[i]
        lower_i = lower[i]

        plt.plot(xs[i], avgs[i], color=cls[i], label=labels[i], marker=ms[i])
        plt.fill_between(xs[i], lower_i, upper_i, color=cls[i], alpha=0.1)

    plt.xlabel(xlab)
    plt.ylabel(ylab)

    if legend:
        if legend_outside:
            leg = plt.legend(loc='upper right', bbox_to_anchor=(-0.15, 1))
        else:
            leg = plt.legend(loc='upper right')

        # Change the marker size manually for both lines
        for i in range(len(leg.legendHandles)):
            leg.legendHandles[i]._legmarker.set_markersize(16)
            leg.legendHandles[i].set_linewidth(5.0)

    if dir is not None:
        plt.savefig(fname=dir, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_pruning_results(n_pruned: list, total_n_units: int, objective: list, bias_metric: list, pred_performance: list,
                         j_best: int, seed: int, config: dict, suffix='', display=False):
    fig = plt.figure()
    plt.plot(np.array(n_pruned) / total_n_units * 100., objective,
             label='Constrained Objective')
    plt.plot(np.array(n_pruned) / total_n_units * 100., bias_metric,
             label='Bias: ' + str(config['metric']))
    plt.plot(np.array(n_pruned) / total_n_units * 100., pred_performance,
             label='Balanced Accuracy')
    plt.vlines(x=(n_pruned[j_best]) / total_n_units * 100., ymin=min(bias_metric), ymax=1.0, colors='red')
    plt.xlabel('% units pruned')
    plt.legend()
    plt.savefig(fname=os.path.join('results/figures/') + str(config['experiment_name'] + '_' + str(seed) + \
                                                             suffix + '.png'), dpi=300, bbox_inches="tight")
    if display:
        plt.show()
