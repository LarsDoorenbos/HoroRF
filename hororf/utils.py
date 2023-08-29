
import logging
from typing import Union
import os
import shutil
import random
import importlib

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

from torch import nn
import torch

LOGGER = logging.getLogger(__name__)
ParallelType = Union[nn.DataParallel, nn.parallel.DistributedDataParallel]


def _build_datasets(params):
    dataset_file: str = params['dataset_file']
    dataset_module = importlib.import_module(dataset_file)
    train_x, train_y = dataset_module.get_training_data(params["class_label"], params["seed"])  # type: ignore
    test_x, test_y = dataset_module.get_testing_data(params["class_label"], params["seed"])  # type: ignore

    space =  dataset_module.get_space()  # type: ignore
    
    LOGGER.info("%d datapoints in dataset '%s'", len(train_x), params['dataset_file'])
    LOGGER.info("%d datapoints in test dataset '%s'", len(test_x), params['dataset_file'])

    return train_x, train_y, test_x, test_y, space


def kfold_split(train_x, train_y, fold, params):
    if params["stratified"]:
        kf = StratifiedKFold(n_splits=params["folds"], shuffle=True, random_state=params["seed"])
    else:
        kf = KFold(n_splits=params["folds"], shuffle=True, random_state=params["seed"])
    
    for i, (train_index, test_index) in enumerate(kf.split(train_x, train_y)):
        if i == fold:
            return train_x[train_index], train_y[train_index], train_x[test_index], train_y[test_index]
        

class WithStateDict(nn.Module):
    """Wrapper to provide a `state_dict` method to a single tensor."""
    def __init__(self, **tensors):
        super().__init__()
        for name, value in tensors.items():
            self.register_buffer(name, value)


def expanduservars(path: str) -> str:
    return os.path.expanduser(os.path.expandvars(path))


def archive_code(path: str, params_file: str) -> None:
    shutil.copy(params_file, path)
    # Copy the current code to the output folder.
    os.system(f"git ls-files -z | xargs -0 tar -czf {os.path.join(path, 'code.tar.gz')}")


def set_seeds(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_results(output_path, params, f1_micro, f1_macro, aupr):
    f1_micro = f1_micro / params["folds"]
    f1_macro = f1_macro / params["folds"]
    aupr = aupr / params["folds"]

    f = open(output_path + "/results" + '_micro' + ".txt", "a")
    f.write(str(f1_micro) + ' ')
    f.close()

    f = open(output_path + "/results" + '_macro' + ".txt", "a")
    f.write(str(f1_macro) + ' ')
    f.close()

    f = open(output_path + "/results" + '_aupr' + ".txt", "a")
    f.write(str(aupr) + ' ')
    f.close()