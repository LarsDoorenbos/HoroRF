
import torch
import numpy as np
from sklearn.model_selection import train_test_split


def get_training_data(class_label, seed):
    data = np.load('datasets/files/multi_vecs' + str(class_label) + '.npy')
    labels = np.load('datasets/files/multi_labels' + str(class_label) + '.npy')

    train, test, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=seed, stratify=labels)
    train, val, train_labels, val_labels = train_test_split(train, train_labels, test_size=0.25, random_state=seed, stratify=train_labels)

    return torch.as_tensor(train), train_labels[:, None]


def get_testing_data(class_label, seed):
    data = np.load('datasets/files/multi_vecs' + str(class_label) + '.npy')
    labels = np.load('datasets/files/multi_labels' + str(class_label) + '.npy')

    train, test, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=seed, stratify=labels)
    train, val, train_labels, val_labels = train_test_split(train, train_labels, test_size=0.25, random_state=seed, stratify=train_labels)

    return torch.as_tensor(val), val_labels[:, None]


def get_space():
    return 'hyperbolic'