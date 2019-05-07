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
from psob_authorship.features.utils import configure_logger_by_default
from psob_authorship.model.Model import Model
from psob_authorship.train.train_bp import train_bp

CONFIG = {
    'experiment_name': os.path.basename(__file__).split('.')[0],
    'experiment_notes': "train on part of train set and check train and test accuracy",
    'train_splits': [0.2, 0.4, 0.6, 0.8, 1],
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
    'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    'pso_options': {}
}
CONFIG['cv'] = StratifiedKFold(n_splits=CONFIG['n_splits'],
                               shuffle=CONFIG['shuffle'], random_state=CONFIG['random_state'])
INPUT_FEATURES = torch.load(CONFIG['labels_features_common_name'] + "_features.tr").numpy()
INPUT_LABELS = torch.load(CONFIG['labels_features_common_name'] + "_labels.tr").numpy()
make_experiment_reproducible(CONFIG['random_state'])


def get_accuracies(file_to_print):
    logger = logging.getLogger(CONFIG['experiment_name'])
    configure_logger_by_default(logger)
    logger.info("START " + CONFIG['experiment_name'])

    def print_info(info):
        logger.info(info)
        print(info)
        file_to_print.write(info + "\n")

    CONFIG['pso_options']['print_info'] = print_info
    full_train_index, test_index = next(CONFIG['cv'].split(INPUT_FEATURES, INPUT_LABELS))
    np.random.shuffle(full_train_index)
    train_accuracies = []
    test_accuracies = []
    for train_percentage in CONFIG['train_splits']:
        train_size = int(train_percentage * full_train_index.shape[0])
        train_index = full_train_index[0:train_size]
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
        test_acc, train_acc = train_bp(model, train_features, train_labels, test_features, test_labels, CONFIG)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
    print_info("======RESULTING ACCURACIES======")
    print_info("TRAIN: " + str(train_accuracies))
    print_info("TEST: " + str(test_accuracies))
    logger.info("END " + CONFIG['experiment_name'])
    return train_accuracies, test_accuracies


def draw_accuracies_graph(train_accuracies, test_accuracies):
    import matplotlib.pyplot as plt
    plt.plot(CONFIG['train_splits'], train_accuracies, '-o')
    plt.plot(CONFIG['train_splits'], test_accuracies, '-o')
    plt.legend(('train', 'test'))
    plt.ylabel("Accuracy")
    plt.xlabel("Percentage of train")
    img_path_to_save_plot = "../experiment_result/" + CONFIG['experiment_name'] + "_"\
                            + str(datetime.datetime.now()) + ".png"
    plt.savefig(img_path_to_save_plot)


def conduct_partial_train_test_accuracy_experiment():
    with open("../experiment_result/" + CONFIG['experiment_name'] + "_" + str(datetime.datetime.now()), 'w') as f:
        f.write("Config: " + str(CONFIG) + "\n")
        start = time.time()
        train_accuracies, test_accuracies = get_accuracies(f)
        draw_accuracies_graph(train_accuracies, test_accuracies)
        end = time.time()
        execution_time = end - start
        f.write("Execution time: " + str(datetime.timedelta(seconds=execution_time)) + "\n")


if __name__ == '__main__':
    conduct_partial_train_test_accuracy_experiment()
