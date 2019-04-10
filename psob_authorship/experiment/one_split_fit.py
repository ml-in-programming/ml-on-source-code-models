import datetime
import logging
import os
import time

import torch
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from torch import optim, nn

from psob_authorship.features.PsobDataset import PsobDataset
from psob_authorship.features.utils import configure_logger_by_default
from psob_authorship.model.Model import Model

CONFIG = {
    'experiment_name': os.path.basename(__file__).split('.')[0],
    'experiment_notes': "change: SGD -> Adam",
    'number_of_authors': 40,
    'labels_features_common_name': "../calculated_features/extracted_for_each_file",
    'metrics': [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18],  # 9 is macro
    'epochs': 5000,
    'batch_size': 32,
    'early_stopping_rounds': 700,
    'lr': 0.02,
    'cv': StratifiedKFold(n_splits=10, random_state=1, shuffle=True),
    'scoring': "accuracy",
    'criterion': nn.CrossEntropyLoss,
    'optimizer': optim.Adam,
    'momentum': 0.9,
    'shuffle': False
}
INPUT_FEATURES = torch.load(CONFIG['labels_features_common_name'] + "_features.tr").numpy()
INPUT_LABELS = torch.load(CONFIG['labels_features_common_name'] + "_labels.tr").numpy()


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
    optimizer = CONFIG['optimizer'](model.parameters(), lr=CONFIG['lr'])
    train_features, train_labels = INPUT_FEATURES[train_index], INPUT_LABELS[train_index]
    test_features, test_labels = INPUT_FEATURES[test_index], INPUT_LABELS[test_index]
    scaler = preprocessing.StandardScaler().fit(train_features)
    train_features = scaler.transform(train_features)
    test_features = scaler.transform(test_features)
    trainloader = torch.utils.data.DataLoader(
        PsobDataset(train_features, train_labels, CONFIG['metrics']),
        batch_size=CONFIG['batch_size'], shuffle=CONFIG['shuffle'], num_workers=2
    )
    testloader = torch.utils.data.DataLoader(
        PsobDataset(test_features, test_labels, CONFIG['metrics']),
        batch_size=CONFIG['batch_size'], shuffle=CONFIG['shuffle'], num_workers=2
    )
    best_accuracy = -1.0
    current_duration = 0
    for epoch in range(CONFIG['epochs']):
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
            current_duration = 0
        best_accuracy = max(best_accuracy, accuracy)
        if current_duration > CONFIG['early_stopping_rounds']:
            print_info("On epoch " + str(epoch) + " training was early stopped")
            break
        if epoch % 10 == 0:
            logger.info("CHECKPOINT EACH 10th EPOCH" + str(epoch) + ": " + str(accuracy))
        if epoch % 100 == 0:
            print_info("CHECKPOINT EACH 100th EPOCH " + str(epoch) + ": current accuracy " + str(accuracy) + " , best "
                       + str(best_accuracy))
        logger.info(str(epoch) + ": " + str(accuracy))

    logger.info('Finished Training')

    correct = 0
    total = 0
    labels_dist = torch.zeros(CONFIG['number_of_authors'])
    labels_correct = torch.zeros(CONFIG['number_of_authors'])
    with torch.no_grad():
        for data in testloader:
            features, labels = data
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for i, label in enumerate(labels):
                labels_dist[label] += 1
                labels_correct[label] += predicted[i] == labels[i]
    print_info('Best accuracy: ' + str(max(best_accuracy, correct / total)))
    print_info('Final accuracy of the network: %d / %d = %d %%' % (correct, total, 100 * correct / total))
    print_info("Correct labels / labels for each author:\n" + str(torch.stack((labels_correct, labels_dist), dim=1)))
    logger.info("END run_cross_validation")


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
