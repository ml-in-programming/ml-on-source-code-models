import datetime
import logging
import os
import time
from typing import Tuple, List

import torch
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from torch import nn, optim
from tqdm import tqdm

from psob_authorship.experiment.utils import make_experiment_reproducible
from psob_authorship.features.PsobDataset import PsobDataset
from psob_authorship.features.utils import configure_logger_by_default
from psob_authorship.model.Model import Model

CONFIG = {
    'experiment_name': os.path.basename(__file__).split('.')[0],
    'experiment_notes': "change: a / 0 = 1 with 700 early stopping rounds",
    'number_of_authors': 40,
    'labels_features_common_name': "../calculated_features/extracted_for_each_file",
    'metrics': [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    'epochs': 10000,
    'batch_size': 32,
    'lr': 0.001,
    'random_state': 4562,
    'n_splits': 10,
    'scoring': "accuracy",
    'criterion': nn.CrossEntropyLoss,
    'optimizer': optim.Adam,
    'momentum': 0.9,
    'shuffle': True
}
CONFIG['cv'] = StratifiedKFold(n_splits=CONFIG['n_splits'], random_state=CONFIG['random_state'], shuffle=True)
INPUT_FEATURES = torch.load(CONFIG['labels_features_common_name'] + "_features.tr").numpy()
INPUT_LABELS = torch.load(CONFIG['labels_features_common_name'] + "_labels.tr").numpy()
make_experiment_reproducible(CONFIG['random_state'])


def get_test_accuracy_by_epoch() -> Tuple[List[int], List[float], List[int]]:
    logger = logging.getLogger('early_stopping')
    configure_logger_by_default(logger)
    logger.info("START get_test_accuracy_by_epoch")
    train_index, test_index = next(CONFIG['cv'].split(INPUT_FEATURES, INPUT_LABELS))

    model = Model(INPUT_FEATURES.shape[1])
    criterion = CONFIG['criterion']()
    optimizer = CONFIG['optimizer'](model.parameters(), lr=CONFIG['lr'])
    train_features, train_labels = INPUT_FEATURES[train_index], INPUT_LABELS[train_index]
    test_features, test_labels = INPUT_FEATURES[test_index], INPUT_LABELS[test_index]
    scaler = preprocessing.StandardScaler().fit(train_features)
    train_features = scaler.transform(train_features)
    test_features = scaler.transform(test_features)
    trainloader = torch.utils.data.DataLoader(
        PsobDataset(train_features, train_labels),
        batch_size=CONFIG['batch_size'], shuffle=CONFIG['shuffle'], num_workers=2
    )
    testloader = torch.utils.data.DataLoader(
        PsobDataset(test_features, test_labels),
        batch_size=CONFIG['batch_size'], shuffle=CONFIG['shuffle'], num_workers=2
    )
    accuracies = []
    best_accuracy = -1
    durations = []
    current_duration = 0
    for epoch in tqdm(range(CONFIG['epochs'])):
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                features, labels = data
                outputs = model(features)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        if best_accuracy >= accuracy:
            current_duration += 1
        else:
            if current_duration != 0:
                durations.append(current_duration)
            current_duration = 0
        best_accuracy = max(best_accuracy, accuracy)
        accuracies.append(accuracy)
        logger.info(str(epoch) + ": " + str(accuracy))
    if current_duration != 0:
        durations.append(current_duration)
    logger.info("END get_test_accuracy_by_epoch")
    return [i for i in range(CONFIG['epochs'])], accuracies, durations


def draw_test_accuracy_by_epoch(epochs, accuracies):
    import matplotlib.pyplot as plt
    plt.plot(epochs, accuracies, '-o')
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.savefig("../experiment_result/early_stopping_plots")


def conduct_early_stopping_experiment():
    start = time.time()
    epochs, accuracies, durations = get_test_accuracy_by_epoch()
    end = time.time()
    execution_time = end - start
    draw_test_accuracy_by_epoch(epochs, accuracies)
    coordinates = list(zip(epochs, accuracies))
    filepath = "../experiment_result/early_stopping_plots_data"
    with open(filepath, 'w') as f:
        f.write("Execution time: " + str(datetime.timedelta(seconds=execution_time)) + "\n")
        f.write("Config: " + str(CONFIG) + "\n")
        f.write("Durations of not growing accuracy: " + str(durations) + "\n")
        f.write("Sorted durations of not growing accuracy: " + str(list(reversed(sorted(durations)))) + "\n")
        for item in coordinates:
            f.write("%s\n" % str(item))


if __name__ == '__main__':
    conduct_early_stopping_experiment()
