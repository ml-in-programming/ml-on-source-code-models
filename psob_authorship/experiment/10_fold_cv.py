import datetime
import logging
import time

import torch
from sklearn.model_selection import RepeatedStratifiedKFold
from torch import optim, nn

from psob_authorship.features.PsobDataset import PsobDataset
from psob_authorship.features.utils import configure_logger_by_default
from psob_authorship.model.Model import Model

CONFIG = {
    'experiment_name': "10_fold_cross_validation",
    'number_of_authors': 40,
    'labels_features_common_name': "../calculated_features/without_5",
    'epochs': 10,
    'batch_size': 32,
    'early_stopping_rounds': 350,
    'lr': 0.02,
    'n_splits': 10,
    'n_repeats': 2,
    'cv': RepeatedStratifiedKFold(n_splits=10, n_repeats=2, random_state=0),
    'scoring': "accuracy",
    'criterion': nn.CrossEntropyLoss,
    'optimizer': optim.SGD,
    'momentum': 0.9,
    'shuffle': True
}
INPUT_FEATURES = torch.load(CONFIG['labels_features_common_name'] + "_features.tr").numpy()
INPUT_LABELS = torch.load(CONFIG['labels_features_common_name'] + "_labels.tr").numpy()


def run_cross_validation() -> torch.Tensor:
    logger = logging.getLogger('10_fold_cv')
    configure_logger_by_default(logger)
    logger.info("START run_cross_validation")
    accuracies = torch.zeros((CONFIG['n_splits'], CONFIG['n_repeats']))
    loop = 0
    for train_index, test_index in CONFIG['cv'].split(INPUT_FEATURES, INPUT_LABELS):
        logger.info("New " + str([loop % 10]) + " fold. loop = " + str(loop))
        model = Model(INPUT_FEATURES.shape[1])
        criterion = CONFIG['criterion']()
        optimizer = CONFIG['optimizer'](model.parameters(), lr=CONFIG['lr'], momentum=CONFIG['momentum'])
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
                break
            if epoch % 10 == 0:
                logger.info("CHECKPOINT EACH 10th EPOCH" + str(epoch) + ": " + str(accuracy))
            if epoch % 100 == 0:
                print("CHECKPOINT EACH 100th EPOCH" + str(epoch) + ": " + str(accuracy))
                logger.info("CHECKPOINT EACH 100th EPOCH" + str(epoch) + ": " + str(accuracy))
            logger.info(str(epoch) + ": " + str(accuracy))

        logger.info('Finished Training')

        correct = 0
        total = 0
        labels_correct = torch.zeros(CONFIG['number_of_authors'])
        with torch.no_grad():
            for data in testloader:
                features, labels = data
                outputs = model(features)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                for i, label in enumerate(labels):
                    labels_correct[label] += predicted[i] == labels[i]
        accuracy_info = str([loop % 10]) + ' fold. Accuracy of the network: %d / %d = %d %%' %\
            (correct, total, 100 * correct / total)
        logger.info(accuracy_info)
        print(accuracy_info)
        labels_info = "Correct answers for each author: " + str(labels_correct)
        logger.info(labels_info)
        print(labels_info)
        accuracies[loop % 10][int(loop / 10)] = correct / total
        loop += 1
    logger.info("END run_cross_validation")
    return accuracies


def conduct_10_fold_cv_experiment():
    start = time.time()
    accuracies = run_cross_validation()
    end = time.time()
    execution_time = end - start
    with open("../experiment_result/" + CONFIG['experiment_name'], 'w') as f:
        f.write("Execution time: " + str(datetime.timedelta(seconds=execution_time)) + "\n")
        f.write("Config: " + str(CONFIG) + "\n")
        f.write("Accuracies: \n" + str(accuracies) + "\n")
        f.write("Means: " + str(torch.mean(accuracies, 1)) + "\n")
        f.write("Stds: " + str(torch.std(accuracies, 1)))


if __name__ == '__main__':
    conduct_10_fold_cv_experiment()
