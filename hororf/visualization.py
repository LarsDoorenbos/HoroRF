
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np


def visualize_decision_boundary(data, labels, feature_1, feature_2, grid_preds, name):

    plt.figure()

    grid_preds = grid_preds.reshape(feature_1.shape).astype(float)
    mask = feature_1 ** 2 + feature_2 ** 2 >= 1
    grid_preds[mask] = np.nan
    plt.contourf(feature_1, feature_2, grid_preds, alpha=0.3, cmap='bwr')

    plt.scatter(data[:, 0], data[:, 1], c=labels, s=3, cmap='bwr')
    circle = plt.Circle((0, 0), 1, color='black', fill=False)
    plt.gca().add_patch(circle)
    plt.axis('off')
    plt.ylim(-1.1, 1.1)
    plt.xlim(-1.1, 1.1)
    plt.savefig(name, bbox_inches='tight', pad_inches=0.0)
    plt.close('all')


def plot_tree(tree, data, labels, name):
    plt.figure()

    plt.scatter(data[:, 0], data[:, 1], c=labels, s=2, cmap='tab10', vmax=10) 
    circle = plt.Circle((0, 0), 1, color='black', fill=False)
    plt.gca().add_patch(circle)

    tree.plot_tree()
    handles, labels = plt.gca().get_legend_handles_labels()
    
    plt.ylim(-1.1, 1.1)
    plt.xlim(-1.1, 1.1)
    plt.axis('off')
    plt.axis('square')

    plt.savefig(name, bbox_inches='tight', pad_inches=0.0)
    plt.close('all')


def visualize_splits(tree, train_x, train_y, name):
    depth = 5
    fig, ax = plt.subplots(nrows=1, ncols=depth)
    vmin = np.min(train_y)
    vmax = np.max(train_y)

    tree.plot_splits(depth, ax, vmin, vmax)

    for level in range(depth):
        ax[level].set_aspect('equal', adjustable='box')
        ax[level].set_xlim(-1.1, 1.1)
        ax[level].set_ylim(-1.1, 1.1)
        ax[level].axis('off')

    plt.savefig(name, bbox_inches='tight', pad_inches=0.0)

