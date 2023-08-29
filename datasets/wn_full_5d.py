
import torch
import numpy as np
from sklearn.model_selection import train_test_split

label_to_name = {
    1: 'animal',
    2: 'group',
    3: 'worker',
    4: 'mammal',
    5: 'tree',
    6: 'solid',
    7: 'occupation',
    8: 'rodent'
}

def get_training_data(class_label, seed):
    print('Running', label_to_name[class_label], 'experiment')
    data = np.load('datasets/files/5d_vecs.npy')
    labels = np.load('datasets/files/' + label_to_name[class_label] + '_labels.npy')
    
    train, test, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=seed, stratify=labels)
    
    return torch.as_tensor(train), train_labels[:, None]


def get_testing_data(class_label, seed):
    data = np.load('datasets/files/5d_vecs.npy')
    labels = np.load('datasets/files/' + label_to_name[class_label] + '_labels.npy')

    train, test, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=seed, stratify=labels)

    return torch.as_tensor(test), test_labels[:, None]


def get_space():
    return 'hyperbolic'