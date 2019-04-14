import datetime
import logging
import os
import time

import numpy as np
import torch
from sklearn import preprocessing
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from torch import optim, nn
from torch.utils.data import Dataset


class PsobDataset(Dataset):
    def __init__(self, features, labels, metrics=None) -> None:
        super().__init__()
        self.features = features
        self.labels = labels
        if metrics is None:
            self.metrics = [i for i in range(self.features.shape[1])]
        else:
            self.metrics = metrics

    def __getitem__(self, index):
        return self.features[index][self.metrics], self.labels[index]

    def __len__(self):
        return len(self.labels)


def get_log_filepath(log_filename: str) -> str:
    log_root = "./"
    log_filename = log_filename + ".log"
    return os.path.join(log_root, log_filename)


def configure_logger_by_default(logger: logging.Logger) -> None:
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(get_log_filepath(logger.name))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)


class Model(nn.Module):
    HIDDEN_DIM = 150

    def __init__(self, input_dim=19, output=40, nonlin=torch.tanh):
        super(Model, self).__init__()

        self.nonlin = nonlin
        self.input_hidden = nn.Linear(input_dim, Model.HIDDEN_DIM, bias=True)
        self.hidden_output = nn.Linear(Model.HIDDEN_DIM, output, bias=True)

    def forward(self, x, **kwargs):
        x = self.nonlin(self.input_hidden(x))
        x = self.hidden_output(x)
        return x


def make_experiment_reproducible(random_seed: int):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)


CONFIG = {
    'experiment_name': "10_fold_cross_validation",
    'experiment_notes': "10 fold cv for Adam",
    'number_of_authors': 40,
    'labels_features_common_name': "./extracted_for_each_file",
    'metrics': [i for i in range(19)],
    'epochs': 10000,
    'batch_size': 32,
    'early_stopping_rounds': 500,
    'lr': 0.001,
    'n_splits': 10,
    'n_repeats': 1,
    'random_state': 4562,
    'scoring': "accuracy",
    'criterion': nn.CrossEntropyLoss,
    'optimizer': optim.Adam,
    'shuffle': True,
    'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
}
CONFIG['cv'] = StratifiedKFold(n_splits=CONFIG['n_splits'], shuffle=True, random_state=CONFIG['random_state']) \
    if CONFIG['n_repeats'] == 1 else RepeatedStratifiedKFold(n_splits=CONFIG['n_splits'],
                                                             n_repeats=CONFIG['n_repeats'],
                                                             random_state=CONFIG['random_state'])
INPUT_FEATURES = torch.load(CONFIG['labels_features_common_name'] + "_features.tr").numpy()
INPUT_LABELS = torch.load(CONFIG['labels_features_common_name'] + "_labels.tr").numpy()
make_experiment_reproducible(CONFIG['random_state'])


def run_cross_validation(file_to_print) -> torch.Tensor:
    logger = logging.getLogger('10_fold_cv')
    configure_logger_by_default(logger)
    logger.info("START run_cross_validation")

    def print_info(info):
        logger.info(info)
        print(info)
        file_to_print.write(info + "\n")

    accuracies = torch.zeros((CONFIG['n_splits'], CONFIG['n_repeats']))
    fold_number = -1
    repeat_number = -1
    for train_index, test_index in CONFIG['cv'].split(INPUT_FEATURES, INPUT_LABELS):
        fold_number += 1
        if fold_number % CONFIG['n_splits'] == 0:
            fold_number = 0
            repeat_number += 1

        print_info("New " + str(fold_number) + " fold. repeat = " + str(repeat_number))
        model = Model(len(CONFIG['metrics'])).to(CONFIG['device'])
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
                inputs = inputs.to(CONFIG['device'])
                labels = labels.to(CONFIG['device'])
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
                    features = features.to(CONFIG['device'])
                    labels = labels.to(CONFIG['device'])
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
                print_info(
                    "CHECKPOINT EACH 100th EPOCH " + str(epoch) + ": current accuracy " + str(accuracy) + " , best "
                    + str(best_accuracy))
            logger.info(str(epoch) + ": " + str(accuracy))

        correct = 0
        total = 0
        labels_dist = torch.zeros(CONFIG['number_of_authors'])
        labels_correct = torch.zeros(CONFIG['number_of_authors'])
        with torch.no_grad():
            for data in testloader:
                features, labels = data
                features = features.to(CONFIG['device'])
                labels = labels.to(CONFIG['device'])
                outputs = model(features)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                for i, label in enumerate(labels):
                    labels_dist[label] += 1
                    labels_correct[label] += predicted[i] == labels[i]
        print_info('Finished training for ' + str(fold_number) + ' fold, repeat ' + str(repeat_number))
        print_info('Best accuracy: ' + str(max(best_accuracy, correct / total)))
        print_info('Accuracy of the last validation of the network: %d / %d = %d %%' %
                   (correct, total, 100 * correct / total))
        print_info("Correct labels / labels for each author of last validation:\n" +
                   str(torch.stack((labels_correct, labels_dist), dim=1)))
        accuracies[fold_number][repeat_number] = best_accuracy
    logger.info("END run_cross_validation")
    return accuracies


def conduct_10_fold_cv_experiment():
    with open("./" + CONFIG['experiment_name'] + "_" + str(datetime.datetime.now()), 'w') as f:
        f.write("Config: " + str(CONFIG) + "\n")

        start = time.time()
        accuracies = run_cross_validation(f)
        end = time.time()
        execution_time = end - start

        f.write("Execution time: " + str(datetime.timedelta(seconds=execution_time)) + "\n")
        f.write("Accuracies: \n" + str(accuracies) + "\n")

        f.write("Mean of all accuracies: " + str(torch.mean(accuracies)) + "\n")
        f.write("Std of all accuracies: " + str(torch.std(accuracies, unbiased=False)) + "\n")
        f.write("Std unbiased of all accuracies (Bessel's correction): " + str(torch.std(accuracies)) + "\n")

        means = torch.mean(accuracies, 1)
        f.write("Means: " + str(means) + "\n")
        f.write("Stds: " + str(torch.std(accuracies, 1, unbiased=False)) + "\n")
        f.write("Stds unbiased (Bessel's correction): " + str(torch.std(accuracies, 1)) + "\n")

        f.write("Mean of means: " + str(torch.mean(means)) + "\n")
        f.write("Std of means: " + str(torch.std(means, unbiased=False)) + "\n")
        f.write("Std of means (Bessel's correction): " + str(torch.std(means)) + "\n")


if __name__ == '__main__':
    conduct_10_fold_cv_experiment()
