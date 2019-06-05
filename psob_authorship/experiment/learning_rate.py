import datetime
import logging
import time
from collections import defaultdict
from typing import Dict

import torch
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from torch import nn
from tqdm import tqdm

from psob_authorship.experiment.utils import make_experiment_reproducible
from psob_authorship.features.PsobDataset import PsobDataset
from psob_authorship.features.utils import configure_logger_by_default
from psob_authorship.model import Model
from psob_authorship.model.Model import Model

CONFIG = {
    'experiment_name': "learning_rate",
    'labels_features_common_name': "../calculated_features/without_5",
    'epochs': 5000,
    'batch_size': 32,
    'early_stopping_rounds': 350,
    'n_splits': 10,
    'scoring': 'accuracy',
    'criterion': torch.nn.CrossEntropyLoss,
    'optimizer': optim.SGD,
    'momentum': 0.9,
    'random_state': 4562,
    'shuffle': True,
    'params': {
        'lr': [0.001, 0.005, 0.01, 0.015, 0.02, 0.025]
    }
}
CONFIG['cv'] = StratifiedKFold(n_splits=CONFIG['cv'], shuffle=True, random_state=CONFIG['random_state'])
INPUT_FEATURES = torch.load(CONFIG['labels_features_common_name'] + "_features.tr").numpy()
INPUT_LABELS = torch.load(CONFIG['labels_features_common_name'] + "_labels.tr").numpy()
make_experiment_reproducible(CONFIG['random_state'])


def get_accuracies_for_lr() -> Dict[float, float]:
    logger = logging.getLogger('learning_rate')
    configure_logger_by_default(logger)
    logger.info("START get_accuracies_for_lr")
    accuracies_by_lr = defaultdict(lambda: -1.0)
    for lr in CONFIG['params']['lr']:
        logger.info("lr = " + str(lr))
        skf = CONFIG['cv']
        train_index, test_index = next(skf.split(INPUT_FEATURES, INPUT_LABELS))

        model = Model(INPUT_FEATURES.shape[1])
        criterion = CONFIG['criterion']()
        optimizer = CONFIG['optimizer'](model.parameters(), lr=lr, momentum=CONFIG['momentum'])
        train_features, train_labels = INPUT_FEATURES[train_index], INPUT_LABELS[train_index]
        test_features, test_labels = INPUT_FEATURES[test_index], INPUT_LABELS[test_index]
        trainloader = torch.utils.data.DataLoader(
            PsobDataset(train_features, train_labels),
            batch_size=CONFIG['batch_size'], shuffle=CONFIG['shuffle'], num_workers=2
        )
        testloader = torch.utils.data.DataLoader(
            PsobDataset(test_features, test_labels),
            batch_size=CONFIG['batch_size'], shuffle=CONFIG['shuffle'], num_workers=2
        )
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
            if accuracies_by_lr[lr] >= accuracy:
                current_duration += 1
            else:
                current_duration = 0
            accuracies_by_lr[lr] = max(accuracies_by_lr[lr], accuracy)
            if current_duration > CONFIG['early_stopping_rounds']:
                break
            if epoch % 10 == 0:
                logger.info("CHECKPOINT EACH 10th EPOCH" + str(epoch) + ": " + str(accuracy))
            if epoch % 100 == 0:
                logger.info("CHECKPOINT EACH 100th EPOCH" + str(epoch) + ": " + str(accuracy))
            logger.info(str(epoch) + ": " + str(accuracy))
    logger.info("END get_accuracies_for_lr")
    return accuracies_by_lr


def conduct_learning_rate_experiment():
    start = time.time()
    accuracies_by_lr = get_accuracies_for_lr()
    end = time.time()
    execution_time = end - start
    with open("../experiment_result/" + CONFIG['experiment_name'], 'w') as f:
        f.write("Execution time: " + str(datetime.timedelta(seconds=execution_time)) + "\n")
        f.write("Config: " + str(CONFIG) + "\n")
        f.write("Accuracies for learning rates: \n" + str(dict(accuracies_by_lr)) + "\n")
        f.write("Best: " + str(max(accuracies_by_lr, key=accuracies_by_lr.get)))


if __name__ == '__main__':
    conduct_learning_rate_experiment()
