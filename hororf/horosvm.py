
import logging
import itertools

import torch

import numpy as np

import pymanopt
from pymanopt import Problem
from pymanopt.manifolds import Sphere, Product, SymmetricPositiveDefinite
from pymanopt.optimizers.conjugate_gradient import ConjugateGradient

from .geom.klein import poincare_mean
from .geom.lca import hyp_lca

LOGGER = logging.getLogger(__name__)

MIN_NORM = 1e-15


def get_class_weights(train_y, beta):
    _, counts = np.unique(train_y, return_counts=True)

    try:
        class_weights = (1 - beta) / (1 - beta ** counts)
    except FloatingPointError:
        class_weights = []
        for count in counts:
            try:
                class_weights.append((1 - beta) / (1 - beta ** count))
            except FloatingPointError:
                class_weights.append(1 - beta)
        class_weights = np.array(class_weights)

    weights_per_sample = class_weights[np.clip(np.array(train_y, dtype=np.int32), a_min=0, a_max=None)]
    weights_per_sample = torch.tensor(weights_per_sample)

    return weights_per_sample


def inverse_busemann(p, x):
    xnorm = x.norm(dim=-1, p=2, keepdim=True)
    pnorm = p.norm(dim=-1, p=2, keepdim=True)
    p = p / pnorm.clamp_min(MIN_NORM)

    num = (1 - xnorm ** 2).clamp_min(MIN_NORM)
    den = torch.norm(p - x, dim=-1, keepdim=True) ** 2

    ans = torch.log((num / den).clamp_min(MIN_NORM))

    return ans


def train_svm(train_x, train_y, C, beta):
    manifold = Product([Sphere(train_x.shape[-1]), SymmetricPositiveDefinite(1), SymmetricPositiveDefinite(1)])
    weights_per_sample = get_class_weights(train_y, beta)

    @pymanopt.function.pytorch(manifold)
    def cost(ideal_point, mu, b):
        mu_reg = 0.5 * mu[0][0]**2

        hinge = torch.clamp(1 - torch.as_tensor(train_y) * (mu[0][0] * inverse_busemann(ideal_point, train_x) - b[0][0]), min=0)
        weighted_hinge = hinge * weights_per_sample
        
        return mu_reg + C * weighted_hinge.sum()

    problem = Problem(manifold, cost)
    optimizer = ConjugateGradient(verbosity=0, beta_rule='FletcherReeves', max_iterations=500, min_step_size=1e-10)

    Xopt = optimizer.run(problem).point
    radius = -torch.tensor(Xopt[2] / Xopt[1])[0][0]
    ideal_point = torch.tensor(Xopt[0])

    return ideal_point, radius


def solve_horosvm(train_x, train_y, params):
    Cexp = np.random.uniform(params["Cexp_min"], params["Cexp_max"])
    C = 2 ** Cexp

    classes = np.unique(train_y).astype(float)

    ideal_points = []
    radii = []
    
    for one_class in classes:
        ova_train_y = np.zeros_like(train_y, dtype=np.float32)

        ova_train_y[train_y == one_class] = 1
        ova_train_y[train_y != one_class] = -1

        ideal_point, radius = train_svm(train_x, ova_train_y, C, params["beta"])

        ideal_points.append(ideal_point)
        radii.append(radius)

    if params["hyperclasses"]:
        new_train_y = np.copy(train_y)

        # get class means
        class_means = {}
        for one_class in classes:
            samples = np.where(new_train_y == one_class)[0]
            class_mean = poincare_mean(train_x[samples], dim=0)
            class_means[one_class] = class_mean[None]
        
        # compute LCAs
        combinations = np.array(list(itertools.combinations(classes, 2)))

        first_means = torch.cat([class_means[x[0]] for x in combinations])
        second_means = torch.cat([class_means[x[1]] for x in combinations])

        _, dists = hyp_lca(first_means, second_means)

        while len(classes) > 2:           
            # Merge 2 classes with highest dist from origin to LCA
            to_merge = combinations[np.argmax(dists)]

            new_train_y[new_train_y == to_merge[1]] = to_merge[0]

            ova_train_y = np.zeros_like(new_train_y, dtype=np.float32)

            ova_train_y[new_train_y == to_merge[0]] = 1
            ova_train_y[new_train_y != to_merge[0]] = -1

            ideal_point, radius = train_svm(train_x, ova_train_y, C, params["beta"])
            ideal_points.append(ideal_point)
            radii.append(radius)

            classes = np.unique(new_train_y).astype(float)

            # Update merged class mean
            samples = np.where(new_train_y == to_merge[0])[0]
            class_mean = poincare_mean(train_x[samples], dim=0)
            class_means[to_merge[0]] = class_mean[None]

            del class_means[to_merge[1]]

            # Update LCAs
            to_keep = np.array([i for i, c in enumerate(combinations) if to_merge[1] not in c])
            combinations = combinations[to_keep]
            dists = dists[to_keep]

            to_redo = np.array([i for i, c in enumerate(combinations) if to_merge[0] in c])

            first_means = torch.cat([class_means[x[0]] for x in combinations[to_redo]])
            second_means = torch.cat([class_means[x[1]] for x in combinations[to_redo]])

            _, new_dists = hyp_lca(first_means, second_means)
            dists[to_redo] = new_dists
    
    return ideal_points, radii