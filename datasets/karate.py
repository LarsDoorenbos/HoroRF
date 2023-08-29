
import torch
import scipy.io


def get_training_data(class_label, seed):
    mat = scipy.io.loadmat('datasets/files/karate_data_' + str(class_label) + '.mat')
    data = mat["B"]
    labels = mat["label"]

    return torch.as_tensor(data), labels


def get_testing_data(class_label, seed):
    mat = scipy.io.loadmat('datasets/files/karate_data_' + str(class_label) + '.mat')
    data = mat["B"]
    labels = mat["label"]

    return torch.as_tensor(data), labels


def get_space():
    return 'hyperbolic'