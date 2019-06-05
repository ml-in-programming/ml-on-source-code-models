import logging
import os
import random
from itertools import chain, combinations
from typing import Tuple, List, Set

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, GridSearchCV
from skorch import NeuralNetClassifier
from skorch.dataset import CVSplit
from tqdm import tqdm

from psob_authorship.features.PsobDataset import PsobDataset
from psob_authorship.features.java.MetricsCalculator import MetricsCalculator
from psob_authorship.features.utils import chunks, configure_logger_by_default
from psob_authorship.model.Model import Model

EPOCHS = 100
BATCH_SIZE = 128
NUM_OF_AUTHORS = 40


def powerset(iterable):
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def print_author_files_distribution(dataset_path="../dataset") -> None:
    print("ID,author_name,number_of_author_files")
    for author_id, author in enumerate(os.listdir(dataset_path)):
        num_of_files = 0
        for _, _, files in os.walk(os.path.join(dataset_path, author)):
            num_of_files += len(files)
        print('{},{},{}'.format(author_id, author, num_of_files))


def get_labeled_data(split=None, dataset_path=os.path.abspath("../dataset"),
                     ast_path=os.path.abspath("../asts")) -> Tuple[torch.Tensor, torch.Tensor]:
    metrics_calculator = MetricsCalculator(dataset_path, ast_path)
    features = []
    labels = []
    for author_id, author in enumerate(tqdm(os.listdir(dataset_path))):
        for root, dirs, files in os.walk(os.path.join(dataset_path, author)):
            if len(files) == 0:
                continue
            random.shuffle(files)
            splitted_files = chunks(files, len(files)) if split is None else chunks(files, split)
            for chunk in splitted_files:
                filepaths = {os.path.abspath(os.path.join(root, file)) for file in chunk}
                files_features = metrics_calculator.get_metrics(filepaths)
                features.append(files_features)
                labels.append(author_id)
    return torch.stack(tuple(features)), torch.tensor(labels)


def run_cross_validation():
    k_fold = 10
    loaded_features, loaded_labels = \
        torch.load("../calculated_features/split_each_file_features.tr"),\
        torch.load("../calculated_features/split_each_file_labels.tr")
    skf = RepeatedStratifiedKFold(n_splits=k_fold, n_repeats=10)
    # metrics = [i for i in range(19)]
    metrics = [0, 1, 2, 3, 4, 6, 7, 9]
    accuraces = torch.zeros((10, 10))
    loop = 0
    for train_index, test_index in skf.split(loaded_features, loaded_labels):
        model = Model(len(metrics))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        train_features, train_labels = loaded_features[train_index], loaded_labels[train_index]
        test_features, test_labels = loaded_features[test_index], loaded_labels[test_index]
        trainloader = torch.utils.data.DataLoader(
            PsobDataset(train_features, train_labels, metrics),
            batch_size=BATCH_SIZE, shuffle=True, num_workers=2
        )
        testloader = torch.utils.data.DataLoader(
            PsobDataset(test_features, test_labels, metrics),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=2
        )
        for epoch in range(EPOCHS):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 10 == 9:
                    # print('[%d, %5d] loss: %.3f' %
                    #      (epoch + 1, i + 1, running_loss / 10))
                    running_loss = 0.0

        print('Finished Training')

        correct = 0
        total = 0
        labels_correct = torch.zeros(NUM_OF_AUTHORS)
        with torch.no_grad():
            for data in testloader:
                features, labels = data
                outputs = model(features)
                _, predicted = torch.max(outputs.data, 1)
                print(predicted)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                for i, label in enumerate(labels):
                    labels_correct[label] += predicted[i] == labels[i]
        print('Accuracy of the network: %d / %d = %d %%' % (
            correct, total, 100 * correct / total))
        print(labels_correct)
        accuraces[loop % 10][int(loop / 10)] = correct / total
        loop += 1
        return
    print(torch.mean(accuraces, 1))
    print(torch.std(accuraces, 1))
    print(accuraces)


def run_train():
    features, labels = torch.load("../calculated_features/features.tr"),\
                       torch.load("../calculated_features/labels.tr")
    permutation = torch.randperm(labels.shape[0])
    features, labels = features[permutation], labels[permutation]
    train_test_split_id = int(labels.shape[0] * 0.75)
    train_features, train_labels = features[:train_test_split_id], labels[:train_test_split_id]
    test_features, test_labels = features[train_test_split_id:], labels[train_test_split_id:]
    trainloader = torch.utils.data.DataLoader(
        PsobDataset(train_features, train_labels),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    testloader = torch.utils.data.DataLoader(
        PsobDataset(test_features, test_labels),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    model = Model(features.shape[1])
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    for epoch in range(EPOCHS):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0

    print('Finished Training')

    correct = 0
    total = 0
    labels_correct = torch.zeros(NUM_OF_AUTHORS)
    with torch.no_grad():
        for data in testloader:
            features, labels = data
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for i, label in enumerate(labels):
                labels_correct[label] += predicted[i] == labels[i]
    print('Accuracy of the network: %d / %d = %d %%' % (
        correct, total, 100 * correct / total))
    print(labels_correct)


def run_train_skorch():
    features, labels = torch.load("../calculated_features/without_5_features.tr"), \
                       torch.load("../calculated_features/without_5_labels.tr")
    net = NeuralNetClassifier(Model, optimizer=optim.SGD, optimizer__momentum=0.9, max_epochs=EPOCHS, lr=0.1, criterion=torch.nn.CrossEntropyLoss, train_split=CVSplit(10, stratified=True), batch_size=BATCH_SIZE)
    net.fit(features, labels)
    print(features[0])
    print(net.predict(features[:10]))
    print(labels[:10])
    print(net.predict_proba(features[:10]))


def get_best_metrics_and_accuracy_from_metrics_set(metrics_sets) -> Tuple[List[int], float]:
    logger = logging.getLogger('finding_best_metrics_and_accuracy')
    configure_logger_by_default(logger)
    logger.info("STARTED FINDING BEST METRICS SET")
    loaded_features, loaded_labels = \
        torch.load("../calculated_features/split_each_file_features.tr"), torch.load("../calculated_features/split_each_file_labels.tr")
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    train_index, test_index = next(skf.split(loaded_features, loaded_labels))

    best_metrics = None
    best_accuracy = -1
    for metrics in tqdm(metrics_sets):
        metrics = list(metrics)
        if len(metrics) == 0:
            continue

        model = Model(len(metrics))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        train_features, train_labels = loaded_features[train_index], loaded_labels[train_index]
        test_features, test_labels = loaded_features[test_index], loaded_labels[test_index]
        trainloader = torch.utils.data.DataLoader(
            PsobDataset(train_features, train_labels, metrics),
            batch_size=BATCH_SIZE, shuffle=True, num_workers=2
        )
        testloader = torch.utils.data.DataLoader(
            PsobDataset(test_features, test_labels, metrics),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=2
        )
        for _ in range(EPOCHS):
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
        log_info = str(metrics) + ": " + str(accuracy)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_metrics = metrics
            log_info += " NEW BEST"
        logger.info(log_info)
    logger.info("END FINDING BEST METRICS SET")
    return best_metrics, best_accuracy


def get_best_metrics_and_accuracy() -> Tuple[List[int], float]:
    return get_best_metrics_and_accuracy_from_metrics_set(list(powerset(range(19))))


def print_best_metrics(metrics_sets=None):
    if metrics_sets is None:
        best_metrics, best_accuracy = get_best_metrics_and_accuracy()
    else:
        best_metrics, best_accuracy = get_best_metrics_and_accuracy_from_metrics_set(metrics_sets)
    print("Best metrics: " + str(best_metrics))
    print("Accuracy on them: " + str(best_accuracy))


def get_best_metrics_leave_one_out(metrics: Set[int] = None):
    # {2, 3, 13, 14, 15, 17, 18} best
    if metrics is None:
        metrics = {i for i in range(19)}
    removed = False
    for metric_id in metrics:
        best_metrics, best_accuracy = get_best_metrics_and_accuracy_from_metrics_set(
            [metrics, metrics - {metric_id}]
        )
        if metrics != {metric_id for metric_id in best_metrics}:
            print("Removed " + str(metric_id))
            print("Accuracy " + str(best_accuracy))
            metrics.remove(metric_id)
            removed = True
            break
    if not removed:
        return metrics
    return get_best_metrics_leave_one_out(metrics)


def print_features_ordered_by_relationship():
    X, y = pd.DataFrame(torch.load("../calculated_features/split_each_file_features.tr").numpy()), pd.DataFrame(torch.load("../calculated_features/split_each_file_labels.tr").numpy())
    # apply SelectKBest class to extract top 10 best features
    bestfeatures = SelectKBest(score_func=mutual_info_classif, k=19)
    fit = bestfeatures.fit(X, y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    # concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    print(featureScores.nlargest(19, 'Score'))  # print 10 best features


def grid_search_hyperparameters():
    X, y = torch.load("../calculated_features/without_5_features.tr").numpy(),\
           torch.load("../calculated_features/without_5_labels.tr").numpy()
    net = NeuralNetClassifier(Model, optimizer=optim.SGD, optimizer__momentum=0.9, max_epochs=EPOCHS, lr=0.1,
                              criterion=torch.nn.CrossEntropyLoss, batch_size=BATCH_SIZE, verbose=1)
    params = {
        'lr': [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3],
        'max_epochs': [2, 10, 100],
        'batch_size': [1, 10, 50, 100, 128, 196, 256]
    }
    gs = GridSearchCV(net, params, scoring='accuracy', cv=10, n_jobs=-1)
    grid_result = gs.fit(X, y)
    print(grid_result)
    print(grid_result.best_score_)
    print(grid_result.best_params_)


if __name__ == '__main__':
    grid_search_hyperparameters()
