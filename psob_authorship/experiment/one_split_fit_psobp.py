import datetime
import logging
import os
import time

import numpy as np
import torch
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from torch import optim, nn

from psob_authorship.experiment.utils import make_experiment_reproducible
from psob_authorship.features.PsobDataset import PsobDataset
from psob_authorship.features.utils import configure_logger_by_default
from psob_authorship.model.Model import Model
from psob_authorship.pso.PSO import PSO
from psob_authorship.train.train_bp import train_bp
from psob_authorship.train.train_pso import train_pso

CONFIG = {
    'experiment_name': os.path.basename(__file__).split('.')[0],
    'experiment_notes': "look at loss of bp",
    'number_of_authors': 40,
    'labels_features_common_name': "../calculated_features/extracted_for_each_file",
    'epochs': 10000,
    'batch_size': 32,
    'early_stopping_rounds': 500,
    'lr': 0.001,
    'n_splits': 10,
    'random_state': 4562,
    'criterion': nn.CrossEntropyLoss(),
    'optimizer': optim.Adam,
    'shuffle': True,
    'trainers_to_use': ['bp'],
    'pso_options': {'c1': 1.49, 'c2': 1.49, 'w': (0.4, 0.9),
                    'unchanged_iterations_stop': 100, 'use_pyswarms': False,
                    'particle_clamp': (-1, 1), 'use_only_early_stopping': False
                    },
    'pso_velocity_clamp': (-1, 1),
    'n_particles': 100,
    'pso_iters': 1000,
    'pso_optimizer': PSO
}
CONFIG['cv'] = StratifiedKFold(n_splits=CONFIG['n_splits'], shuffle=True, random_state=CONFIG['random_state'])
INPUT_FEATURES = torch.load(CONFIG['labels_features_common_name'] + "_features.tr").numpy()
INPUT_LABELS = torch.load(CONFIG['labels_features_common_name'] + "_labels.tr").numpy()
make_experiment_reproducible(CONFIG['random_state'])


def fit_model(file_to_print):
    logger = logging.getLogger('one_split_fit')
    configure_logger_by_default(logger)
    logger.info("START fit_model")

    def print_info(info):
        logger.info(info)
        print(info)
        file_to_print.write(info + "\n")

    CONFIG['pso_options']['print_info'] = print_info
    train_index, test_index = next(CONFIG['cv'].split(INPUT_FEATURES, INPUT_LABELS))
    train_features, train_labels = INPUT_FEATURES[train_index], INPUT_LABELS[train_index]
    test_features, test_labels = INPUT_FEATURES[test_index], INPUT_LABELS[test_index]
    scaler = preprocessing.StandardScaler().fit(train_features)
    train_features = scaler.transform(train_features)
    test_features = scaler.transform(test_features)

    train_features = torch.from_numpy(train_features)
    train_labels = torch.from_numpy(train_labels)
    test_features = torch.from_numpy(test_features)
    test_labels = torch.from_numpy(test_labels)

    model = Model()
    for trainer_type in CONFIG['trainers_to_use']:
        if trainer_type == 'bp':
            train_bp(model, train_features, train_labels, test_features, test_labels, CONFIG)
        elif trainer_type == 'pso':
            train_pso(model, train_features, train_labels, test_features, test_labels, CONFIG)
    logger.info("END fit_model")


def conduct_one_split_fit_experiment():
    with open("../experiment_result/" + CONFIG['experiment_name'] + "_" + str(datetime.datetime.now()), 'w') as f:
        f.write("Config: " + str(CONFIG) + "\n")
        start = time.time()
        fit_model(f)
        end = time.time()
        execution_time = end - start
        f.write("Execution time: " + str(datetime.timedelta(seconds=execution_time)) + "\n")


if __name__ == '__main__':
    conduct_one_split_fit_experiment()
