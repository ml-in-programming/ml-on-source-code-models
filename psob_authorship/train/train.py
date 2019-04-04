import os
import random
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from skorch import NeuralNetClassifier
from skorch.dataset import CVSplit
from tqdm import tqdm

from psob_authorship.features.PsobDataset import PsobDataset
from psob_authorship.features.java.MetricsCalculator import MetricsCalculator
from psob_authorship.features.utils import chunks
from psob_authorship.model.Model import Model

EPOCHS = 100
BATCH_SIZE = 128
NUM_OF_AUTHORS = 40


def get_labeled_data(split=None) -> Tuple[torch.Tensor, torch.Tensor]:
    dataset_path = "../dataset"
    ast_path = "../asts"
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
                filepaths = {os.path.abspath(os.path.join(root,  file)) for file in chunk}
                files_features = metrics_calculator.get_metrics(filepaths)
                features.append(torch.tensor(files_features))
                labels.append(author_id)
    return torch.stack(tuple(features)), torch.tensor(labels)


def run_cross_validation():
    k_fold = 10
    loaded_features, loaded_labels = \
        torch.load("./split_each_file_features.tr"), torch.load("./split_each_file_labels.tr")
    skf = RepeatedStratifiedKFold(n_splits=k_fold, n_repeats=10)
    metrics = [0, 1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 18]
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
    features, labels = torch.load("./features.tr"), torch.load("./labels.tr")
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
    features, labels = get_labeled_data()
    net = NeuralNetClassifier(Model, max_epochs=100, lr=0.1, train_split=CVSplit(4))
    net.fit(features, labels)
    print(features[0])
    print(net.predict(features[:10]))
    print(labels[:10])
    print(net.predict_proba(features[:10]))


if __name__ == '__main__':
    run_cross_validation()
