import logging
from typing import Tuple, List

import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn, optim
from tqdm import tqdm

from psob_authorship.features.PsobDataset import PsobDataset
from psob_authorship.features.utils import configure_logger_by_default
from psob_authorship.model.Model import Model

LABELS_FEARURES_COMMON_NAME = "../calculated_features/without_5"
INPUT_FEATURES = torch.load(LABELS_FEARURES_COMMON_NAME + "_features.tr")
INPUT_LABELS = torch.load(LABELS_FEARURES_COMMON_NAME + "_labels.tr")
EPOCHS = 2000
BATCH_SIZE = 32


def get_test_accuracy_by_epoch() -> Tuple[List[int], List[float], int]:
    logger = logging.getLogger('early_stopping')
    configure_logger_by_default(logger)
    logger.info("START draw_test_accuracy_graph")
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    train_index, test_index = next(skf.split(INPUT_FEATURES, INPUT_LABELS))

    model = Model(INPUT_FEATURES.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    train_features, train_labels = INPUT_FEATURES[train_index], INPUT_LABELS[train_index]
    test_features, test_labels = INPUT_FEATURES[test_index], INPUT_LABELS[test_index]
    trainloader = torch.utils.data.DataLoader(
        PsobDataset(train_features, train_labels),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    testloader = torch.utils.data.DataLoader(
        PsobDataset(test_features, test_labels),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )
    accuracies = []
    best_accuracy = -1
    longest = -1
    current_duration = 0
    for epoch in tqdm(range(EPOCHS)):
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
            longest = max(longest, current_duration)
            current_duration = 0
        best_accuracy = max(best_accuracy, accuracy)
        accuracies.append(accuracy)
        logger.info(str(epoch) + ": " + str(accuracy))
    longest = max(longest, current_duration)
    logger.info("END draw_test_accuracy_graph")
    return [i for i in range(EPOCHS)], accuracies, longest


def draw_test_accuracy_by_epoch(epochs, accuracies):
    import matplotlib.pyplot as plt
    plt.plot(epochs, accuracies, '-o')
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.savefig("../experiment_result/early_stopping_plots")


def conduct_early_stopping_experiment():
    epochs, accuracies, longest = get_test_accuracy_by_epoch()
    draw_test_accuracy_by_epoch(epochs, accuracies)
    coordinates = list(zip(epochs, accuracies))
    with open("../experiment_result/early_stopping_plots_data", 'w') as f:
        f.write(str(longest) + " epochs - longest duration of not improving accuracy\n")
        for item in coordinates:
            f.write("%s\n" % str(item))


if __name__ == '__main__':
    conduct_early_stopping_experiment()
