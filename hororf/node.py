
import matplotlib.pyplot as plt
from matplotlib import colors

import numpy as np
from scipy.stats import entropy

import torch

from .geom.horo import busemann, busemann_to_horocycle
from .horosvm import solve_horosvm

cmap = colors.ListedColormap(['0.3','0.3','y','k','r'])
styles = ['-', '--', '-.', ':', (5, (10, 3)), (5, (10, 3))]
np.seterr(all='raise')


def gini_impurity(counts):
    return np.sum(counts * (1 - counts))


class Node:
    def __init__(self, train_x: torch.Tensor, train_y: np.array, depth:int, params: dict):
        self.train_x = train_x
        self.train_y = train_y

        self.params = params
        
        self.depth = depth

        self.max_depth = params["max_depth"]
        self.criterion = params["criterion"]
        self.min_samples_leaf = params["min_samples_leaf"]
        self.min_impurity_decrease = params["min_impurity_decrease"]

        self.radius_search = params["radius_search"]
        self.subsample_size = params["subsample_size"]

        self.number_of_backup_points = params["number_of_backup_points"]
        self.ideal_points = self.get_random_ideal_points(train_x.shape[-1], self.number_of_backup_points)

        self.left = None
        self.right = None

        values, counts = np.unique(self.train_y, return_counts=True)
        max_indices = np.argwhere(counts == np.amax(counts)).flatten()
        max_index = np.random.choice(max_indices)

        self.predicted = values[max_index]
        self.probability = counts[max_index] / counts.sum()
        
        self.best_ideal_point = None
        self.best_radius = None

        self.gini_impurity = -1000

    def get_random_ideal_points(self, dim: int, number_of_backup_points: int):
        ideal_points = torch.normal(0, 1, size=(number_of_backup_points, dim))
        ideal_points = ideal_points / ideal_points.norm(dim=-1, p=2, keepdim=True)

        return ideal_points
    
    def compute_information_gain(self, dists, radius, root_entropy):
        _, counts1 = np.unique(self.train_y[dists >= radius], return_counts=True)
        counts1 = counts1 / len(self.train_y[dists >= radius])
        
        _, counts2 = np.unique(self.train_y[dists < radius], return_counts=True)
        counts2 = counts2 / len(self.train_y[dists < radius])
        
        entropy_1 = gini_impurity(counts1) if self.criterion == "gini" else entropy(counts1, base=2)
        entropy_1 = entropy_1 * (len(self.train_y[dists >= radius]) / len(self.train_y))
        entropy_2 = gini_impurity(counts2) if self.criterion == "gini" else entropy(counts2, base=2)
        entropy_2 = entropy_2 * (len(self.train_y[dists < radius]) / len(self.train_y))

        split_entropy = entropy_1 + entropy_2
        information_gain = root_entropy - split_entropy

        return information_gain
        
    def get_radii(self, dists, method):
        if method == 'exhaustive':
            sort_indices = torch.argsort(dists)

            sorted_dists = dists[sort_indices]
            sorted_y = self.train_y[sort_indices]

            changes = np.diff(sorted_y.flatten()) != 0

            diffs = torch.diff(sorted_dists)
            radii = sorted_dists[:-1] + diffs / 2
            radii = radii[changes]
        elif method == 'random':
            random_dists = np.random.choice(dists, 2, replace=False)
            radii = [np.mean(random_dists)]
        elif method == 'subsample': 
            random_inds = np.random.choice(dists.shape[0], min(len(dists), self.subsample_size), replace=False)

            sorted_dists = torch.sort(dists[random_inds])[0]
            diffs = torch.diff(sorted_dists)

            radii = sorted_dists[:-1] + diffs / 2
        
        return radii
    
    def find_best_point_without_optimization(self, points_to_consider, root_entropy, max_ig):
        for ideal_point in points_to_consider:
            
            dists = busemann(self.train_x, ideal_point)
            dists = dists[:, 0]

            radii = self.get_radii(dists, self.radius_search)
            for radius in radii:
                if len(self.train_y[dists >= radius]) >= self.min_samples_leaf and len(self.train_y[dists < radius]) >= self.min_samples_leaf:
                    information_gain = self.compute_information_gain(dists, radius, root_entropy)

                    if information_gain > self.min_impurity_decrease:
                        if information_gain > max_ig:
                            max_ig = information_gain

                            best_ideal_point = [ideal_point]
                            best_radius = [radius]
                        elif information_gain == max_ig:
                            best_ideal_point.append(ideal_point)
                            best_radius.append(radius)

        if max_ig == -1000:
            return None, max_ig, None, None

        best_random_split = np.random.choice(len(best_ideal_point))
        
        self.best_ideal_point = best_ideal_point[best_random_split]
        self.best_radius = best_radius[best_random_split]
        best_dists = busemann(self.train_x, self.best_ideal_point)[:, 0]

        return best_dists, max_ig, best_ideal_point, best_radius
    
    def find_best_point_with_horosvm(self, root_entropy, max_ig):
        ideal_points, radii = solve_horosvm(self.train_x, self.train_y, self.params)
        
        for ideal_point, radius in zip(ideal_points, radii):
            dists = busemann(self.train_x, ideal_point)
            dists = dists[:, 0]

            if len(self.train_y[dists >= radius]) >= self.min_samples_leaf and len(self.train_y[dists < radius]) >= self.min_samples_leaf:
                information_gain = self.compute_information_gain(dists, radius, root_entropy)

                if information_gain > self.min_impurity_decrease:
                    if information_gain > max_ig:
                        max_ig = information_gain

                        best_ideal_point = [ideal_point]
                        best_radius = [radius]
                    elif information_gain == max_ig:
                        best_ideal_point.append(ideal_point)
                        best_radius.append(radius)

        if max_ig == -1000:
            return None, max_ig

        best_random_split = np.random.choice(len(best_ideal_point))
        
        self.best_ideal_point = best_ideal_point[best_random_split]
        self.best_radius = best_radius[best_random_split]
        best_dists = busemann(self.train_x, self.best_ideal_point)[:, 0]

        return best_dists, max_ig

    def grow(self):
        values, counts = np.unique(self.train_y, return_counts=True)
        
        if len(values) == 1 or self.depth == self.max_depth:
            return
        
        root_entropy = gini_impurity(counts / counts.sum()) if self.criterion == "gini" else entropy(counts / counts.sum(), base=2)
        self.gini_impurity = root_entropy

        best_dists, max_ig_horo = self.find_best_point_with_horosvm(root_entropy, -1000)

        if max_ig_horo == -1000:
            best_dists, max_ig_horo, _, _ = self.find_best_point_without_optimization(self.ideal_points, root_entropy, -1000)  

        if max_ig_horo == -1000:
            return

        # Split the node
        left_node = Node(self.train_x[best_dists >= self.best_radius], self.train_y[best_dists >= self.best_radius], self.depth + 1, self.params)
        right_node = Node(self.train_x[best_dists < self.best_radius], self.train_y[best_dists < self.best_radius], self.depth + 1, self.params)

        left_node.grow()
        right_node.grow()

        self.left = left_node
        self.right = right_node

    def plot_tree(self):
        if self.best_ideal_point != None and self.best_radius != None:
            c, r = busemann_to_horocycle(self.best_ideal_point, torch.as_tensor(self.best_radius))
            circle = plt.Circle(c, r, color=cmap(self.depth), fill=False, label='Split ' + str(self.depth + 1), lw=2.5, linestyle=styles[self.depth])
            plt.gca().add_patch(circle)
        
        if self.left is not None: 
            self.left.plot_tree()
        
        if self.right is not None:
            self.right.plot_tree()

    def predict(self, test_x):
        if self.left is None and self.right is None:
            return self.predicted, self.probability
        
        if len(test_x.shape) == 1:
            test_x = test_x[None]

        dists = busemann(test_x, self.best_ideal_point)
        if dists >= self.best_radius:
            return self.left.predict(test_x)
        else:
            return self.right.predict(test_x)
        
    def get_depth(self):
        if self.left is None and self.right is None:
            return self.depth
        
        return max(self.left.get_depth(), self.right.get_depth())
    
    def plot_splits(self, max_depth, axes, vmin, vmax):
        if self.depth >= max_depth:
            return 
        
        if self.best_ideal_point != None and self.best_radius != None:
            c, r = busemann_to_horocycle(self.best_ideal_point, torch.as_tensor(self.best_radius))
            circle = plt.Circle(c, r, color='0.3', fill=False, label='Split ' + str(self.depth + 1), lw=2)

            axes[self.depth].add_patch(circle)
            axes[self.depth].scatter(self.train_x[:, 0], self.train_x[:, 1], c=self.train_y, s=1, marker='.', cmap='tab10', vmin=0, vmax=7)

            circle = plt.Circle((0, 0), 1, color='black', fill=False)
            axes[self.depth].add_patch(circle)

        axes[self.depth].set_aspect('equal', adjustable='box')
        axes[self.depth].set_xlim(-1.01, 1.01)
        axes[self.depth].set_ylim(-1.01, 1.01)
        axes[self.depth].axis('off')

        if self.left is None and self.right is None:
            return
        elif self.left is not None and self.right is None:
            self.left.plot_splits(max_depth, axes, vmin, vmax)
        elif self.left is None and self.right is not None:
            self.right.plot_splits(max_depth, axes, vmin, vmax)
        elif self.left.gini_impurity > self.right.gini_impurity: 
            self.left.plot_splits(max_depth, axes, vmin, vmax)
        elif self.right is not None:
            self.right.plot_splits(max_depth, axes, vmin, vmax)
        