import datetime
import logging
import os
import time

import numpy as np
import torch
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from torch import nn

from psob_authorship.experiment.utils import make_experiment_reproducible
from psob_authorship.features.PsobDataset import PsobDataset
from psob_authorship.features.utils import configure_logger_by_default
from psob_authorship.model.Model import Model
from psob_authorship.pso.PSO import PSO

CONFIG = {
    'experiment_name': os.path.basename(__file__).split('.')[0],
    'experiment_notes': "c1, c2 -> 1.49, max velocity range",
    'number_of_authors': 40,
    'labels_features_common_name': "../calculated_features/extracted_for_each_file",
    'metrics': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],  # 9 is macro
    'early_stopping_rounds': 700,
    'n_splits': 10,
    'random_state': 4562,
    'criterion': nn.CrossEntropyLoss,
    'pso_options': {'c1': 1.49, 'c2': 1.49, 'w': 0.4},
    'pso_velocity_range': (0, 1),
    'n_particles': 100,
    'pso_iters': 400,
    'optimizer': PSO,
    'shuffle': True
}
CONFIG['cv'] = StratifiedKFold(n_splits=CONFIG['n_splits'], random_state=CONFIG['random_state'], shuffle=True)
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

    train_index, test_index = next(CONFIG['cv'].split(INPUT_FEATURES, INPUT_LABELS))
    model = Model(len(CONFIG['metrics']))
    criterion = CONFIG['criterion']()
    optimizer = CONFIG['optimizer'](model, criterion, CONFIG['pso_options'], CONFIG['n_particles'])

    train_features, train_labels = INPUT_FEATURES[train_index], INPUT_LABELS[train_index]
    test_features, test_labels = INPUT_FEATURES[test_index], INPUT_LABELS[test_index]
    scaler = preprocessing.StandardScaler().fit(train_features)
    train_features = scaler.transform(train_features)
    test_features = scaler.transform(test_features)

    train_features = torch.from_numpy(train_features)
    train_labels = torch.from_numpy(train_labels)
    test_features = torch.from_numpy(test_features)
    test_labels = torch.from_numpy(test_labels)
    best_accuracy = -1.0
    pso_bounds = (np.full((model.dimensions,), CONFIG['pso_velocity_range'][0]),
                  np.full((model.dimensions,), CONFIG['pso_velocity_range'][1]))
    loss, _ = optimizer.optimize(train_features, train_labels, CONFIG['pso_iters'], pso_bounds)
    print_info("Loss after PSO optimizing = " + str(loss))
    correct = 0
    total = 0
    labels_dist = torch.zeros(CONFIG['number_of_authors'])
    labels_correct = torch.zeros(CONFIG['number_of_authors'])
    outputs = model(test_features)
    _, predicted = torch.max(outputs.data, 1)
    total += test_labels.size(0)
    correct += (predicted == test_labels).sum().item()
    for i, label in enumerate(test_labels):
        labels_dist[label] += 1
        labels_correct[label] += predicted[i] == test_labels[i]
    print_info('Best accuracy: ' + str(max(best_accuracy, correct / total)))
    print_info('Final accuracy of the network: %d / %d = %d %%' % (correct, total, 100 * correct / total))
    print_info("Correct labels / labels for each author:\n" + str(torch.stack((labels_correct, labels_dist), dim=1)))
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
