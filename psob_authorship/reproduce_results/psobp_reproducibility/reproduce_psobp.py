import datetime
import logging
import os
import time

import torch
from sklearn import preprocessing
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from torch import optim, nn
import numpy as np

from functools import reduce


def make_experiment_reproducible(random_seed: int):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)


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

    def __init__(self, input_dim=19, output_dim=40, nonlin=torch.tanh):
        super(Model, self).__init__()

        self.nonlin = nonlin
        self.input_hidden = nn.Linear(input_dim, Model.HIDDEN_DIM, bias=True)
        self.hidden_output = nn.Linear(Model.HIDDEN_DIM, output_dim, bias=True)
        self.dimensions = input_dim * Model.HIDDEN_DIM + Model.HIDDEN_DIM + \
                          Model.HIDDEN_DIM * output_dim + output_dim

    def forward(self, x, **kwargs):
        x = self.nonlin(self.input_hidden(x))
        x = self.hidden_output(x)
        return x


class DecreasingWeightPsoOptimizer:
    def __init__(self, n_particles, dimensions, options, velocity_clamp) -> None:
        super().__init__()
        self.n_particles = n_particles
        self.dimensions = dimensions
        self.options = options
        self.velocity_clamp = velocity_clamp
        self.particles = None
        self.velocities = None

    def optimize(self, f, iters, print_checkpoint):
        self.initialize_particles()
        pbs = np.copy(self.particles)
        pbs_loss = f(pbs)
        pg = pbs[np.argmin(pbs_loss)]
        pg_loss = np.min(pbs_loss)
        w_min = self.options['w'][0]
        w_max = self.options['w'][1]
        stays_unchanged = 0
        i = 0
        while self.options['use_only_early_stopping'] or i < iters:
            w = w_max - (w_max - w_min) / iters * i
            self.velocities = w * self.velocities + \
                              self.options['c1'] * np.random.uniform(0, 1, size=self.particles.shape) * \
                              (pbs - self.particles) + \
                              self.options['c2'] * np.random.uniform(0, 1, size=self.particles.shape) * \
                              (pg - self.particles)
            self.particles = self.particles + self.velocities
            if self.velocity_clamp is not None:
                self.velocities = np.clip(self.velocities, self.velocity_clamp[0], self.velocity_clamp[1])
            if self.options['use_particle_clamp_each_iteration']:
                self.particles = np.clip(self.particles,
                                         self.options['particle_clamp'][0], self.options['particle_clamp'][1])
            new_loss = f(self.particles)
            pbs_with_new_loss = np.vstack((pbs_loss, new_loss))
            pbs_loss = np.min(pbs_with_new_loss, axis=0)
            new_or_not = np.argmin(pbs_with_new_loss, axis=0)
            pbs = np.stack((pbs, self.particles), axis=1)[np.arange(pbs.shape[0]), new_or_not]
            pg = pbs[np.argmin(pbs_loss)]
            pg_loss_prev = pg_loss
            pg_loss = np.min(pbs_loss)
            if i % 100 == 0:
                print_checkpoint(i, pg)
            stays_unchanged = stays_unchanged + 1 if pg_loss_prev == pg_loss else 0
            if stays_unchanged == self.options['unchanged_iterations_stop']:
                self.options['print_info']("Training early stopped on iteration " + str(i))
                return pg_loss, pg
            i += 1
        return pg_loss, pg

    def initialize_particles(self):
        self.particles = np.random.uniform(
            low=self.options['particle_clamp'][0], high=self.options['particle_clamp'][1],
            size=(self.n_particles, self.dimensions)
        )
        if self.velocity_clamp is not None:
            self.velocities = np.random.uniform(low=self.velocity_clamp[0], high=self.velocity_clamp[1],
                                                size=(self.n_particles, self.dimensions))
        else:
            self.velocities = np.random.random_sample(size=(self.n_particles, self.dimensions))


def get_model_accuracy_and_loss(model, criterion, features, labels):
    with torch.no_grad():
        outputs = model(features)
        loss = criterion(outputs, labels).item()
        _, predicted = torch.max(outputs.data, 1)
        correct = 0
        total = 0
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        accuracy = correct / total
        return loss, accuracy


def get_model_accuracy_and_loss_for_train_and_test(model, criterion,
                                                   train_features, train_labels,
                                                   test_features, test_labels):
    train_loss, train_accuracy = get_model_accuracy_and_loss(model, criterion, train_features, train_labels)
    test_loss, test_accuracy = get_model_accuracy_and_loss(model, criterion, test_features, test_labels)
    return train_loss, test_loss, train_accuracy, test_accuracy


def print_100th_checkpoint_evaluation(epoch,
                                      model, criterion,
                                      train_features, train_labels,
                                      test_features, test_labels,
                                      print_info):
    train_loss, test_loss, train_accuracy, test_accuracy = \
        get_model_accuracy_and_loss_for_train_and_test(model, criterion,
                                                       train_features, train_labels,
                                                       test_features, test_labels)
    print_info(
        "CHECKPOINT EACH 100th EPOCH " + str(epoch) + ". Train Loss: " + str(train_loss) +
        " , Test Loss: " + str(test_loss) + ", Train Accuracy: " + str(train_accuracy) +
        ", Test Accuracy: " + str(test_accuracy)
    )


def print_evaluation_before_train(model, criterion,
                                  train_features, train_labels,
                                  test_features, test_labels,
                                  print_info):
    train_loss, test_loss, train_accuracy, test_accuracy = \
        get_model_accuracy_and_loss_for_train_and_test(model, criterion,
                                                       train_features, train_labels,
                                                       test_features, test_labels)
    print_info(
        "INITIAL DATA. Train Loss: " + str(train_loss) +
        " , Test Loss: " + str(test_loss) + ", Train Accuracy: " + str(train_accuracy) +
        ", Test Accuracy: " + str(test_accuracy)
    )


class PSO:
    def __init__(self, model: nn.Module, criterion, options, n_particles) -> None:
        super().__init__()
        self.criterion = criterion
        self.model = model
        self.options = options
        self.n_particles = n_particles

    def print_pso_checkpoint(self,
                             train_features: torch.Tensor, train_labels: torch.Tensor,
                             test_features: torch.Tensor, test_labels: torch.Tensor,
                             print_info):
        def semi_applied_func(iteration, particle):
            self.set_model_weights(particle)
            with torch.no_grad():
                print_100th_checkpoint_evaluation(iteration,
                                                  self.model, self.criterion,
                                                  train_features, train_labels,
                                                  test_features, test_labels,
                                                  print_info)

        return semi_applied_func

    def optimize(self, train_features: torch.Tensor, train_labels: torch.Tensor,
                 test_features: torch.Tensor, test_labels: torch.Tensor,
                 iters: int, velocity_clamp):
        def f_to_optimize(particles):
            with torch.no_grad():
                losses = []
                for particle in particles:
                    self.set_model_weights(particle)
                    outputs = self.model(train_features)
                    loss = self.criterion(outputs, train_labels)
                    losses.append(loss)
                return np.array(losses)

        if self.options['use_pyswarms']:
            import pyswarms as ps
            optimizer = ps.single.GlobalBestPSO(n_particles=self.n_particles, dimensions=self.model.dimensions,
                                                options=self.options, velocity_clamp=velocity_clamp)
        else:
            optimizer = DecreasingWeightPsoOptimizer(n_particles=self.n_particles, dimensions=self.model.dimensions,
                                                     options=self.options, velocity_clamp=velocity_clamp)

        final_loss, best_params = optimizer.optimize(f_to_optimize, iters=iters,
                                                     print_checkpoint=
                                                     self.print_pso_checkpoint(train_features, train_labels,
                                                                               test_features, test_labels,
                                                                               self.options['print_info']))
        self.set_model_weights(best_params)
        return final_loss, best_params

    def set_model_weights(self, particle):
        start = 0
        for param in self.model.parameters():
            gap = reduce(lambda x, y: x * y, param.shape)
            param.data = torch.tensor(particle[start:start + gap].reshape(param.shape)).type(torch.FloatTensor)
            start += gap


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


def train_bp(model, train_features, train_labels, test_features, test_labels, config):
    print_info = config['pso_options']['print_info']

    criterion = config['criterion']
    optimizer = config['optimizer'](model.parameters(), lr=config['lr'])

    trainloader = torch.utils.data.DataLoader(
        PsobDataset(train_features, train_labels),
        batch_size=config['batch_size'], shuffle=config['shuffle'], num_workers=2
    )
    testloader = torch.utils.data.DataLoader(
        PsobDataset(test_features, test_labels),
        batch_size=config['batch_size'], shuffle=config['shuffle'], num_workers=2
    )

    best_accuracy = -1.0
    current_duration = 0
    print_evaluation_before_train(model, criterion,
                                  train_features, train_labels,
                                  test_features, test_labels,
                                  print_info)
    for epoch in range(config['epochs']):
        for inputs, labels in trainloader:
            inputs = inputs.to(config['device'])
            labels = labels.to(config['device'])
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs = inputs.to(config['device'])
                labels = labels.to(config['device'])
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        if best_accuracy >= accuracy:
            current_duration += 1
        else:
            current_duration = 0
        best_accuracy = max(best_accuracy, accuracy)
        if current_duration > config['early_stopping_rounds']:
            print_info("On epoch " + str(epoch) + " training was early stopped")
            break
        if epoch % 100 == 0:
            with torch.no_grad():
                print_100th_checkpoint_evaluation(epoch,
                                                  model, criterion,
                                                  train_features, train_labels,
                                                  test_features, test_labels,
                                                  print_info)
    correct = 0
    total = 0
    labels_dist = torch.zeros(config['number_of_authors'])
    labels_correct = torch.zeros(config['number_of_authors'])
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.to(config['device'])
            labels = labels.to(config['device'])
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for i, label in enumerate(labels):
                labels_dist[label] += 1
                labels_correct[label] += predicted[i] == labels[i]
    print_info('Finished training')
    best_accuracy = max(best_accuracy, correct / total)
    print_info('Best accuracy: ' + str(best_accuracy))
    print_info('Accuracy of the last validation of the network: %d / %d = %d %%' %
               (correct, total, 100 * correct / total))
    print_info("Correct labels / labels for each author of last validation:\n" +
               str(torch.stack((labels_correct, labels_dist), dim=1)))
    return best_accuracy


def train_pso(model, train_features, train_labels, test_features, test_labels, config):
    print_info = config['pso_options']['print_info']

    criterion = config['criterion']
    optimizer = config['pso_optimizer'](model, criterion, config['pso_options'], config['n_particles'])

    print_evaluation_before_train(model, criterion,
                                  train_features, train_labels,
                                  test_features, test_labels,
                                  print_info)
    loss, _ = optimizer.optimize(train_features, train_labels,
                                 test_features, test_labels,
                                 config['pso_iters'], config['pso_velocity_clamp'])
    print_info("Train loss after PSO optimizing = " + str(loss))

    correct = 0
    total = 0
    outputs = model(train_features)
    _, predicted = torch.max(outputs.data, 1)
    total += train_labels.size(0)
    correct += (predicted == train_labels).sum().item()
    print_info('Train accuracy of the network: %d / %d = %d %%' % (correct, total, 100 * correct / total))

    correct = 0
    total = 0
    labels_dist = torch.zeros(config['number_of_authors'])
    labels_correct = torch.zeros(config['number_of_authors'])
    outputs = model(test_features)
    _, predicted = torch.max(outputs.data, 1)
    total += test_labels.size(0)
    correct += (predicted == test_labels).sum().item()
    for i, label in enumerate(test_labels):
        labels_dist[label] += 1
        labels_correct[label] += predicted[i] == test_labels[i]
    final_accuracy = correct / total
    print_info('Final accuracy of the network: %d / %d = %d %%' % (correct, total, 100 * final_accuracy))
    print_info("Correct labels / labels for each author:\n" + str(torch.stack((labels_correct, labels_dist), dim=1)))
    print_info("END OF PSO TRAINING")
    return final_accuracy


# EXPERIMENT CODE
CONFIG = {
    'experiment_name': "10_fold_cv_psobp",
    'experiment_notes': "reproduction of results, all params as in paper, r1 and r2 are random vectors generated each "
                        "iteration",
    'number_of_authors': 40,
    'labels_features_common_name': "../extracted_for_each_file",
    'epochs': 10000,
    'batch_size': 32,
    'early_stopping_rounds': 500,
    'lr': 0.001,
    'n_splits': 10,
    'n_repeats': 1,
    'random_state': 4562,
    'criterion': nn.CrossEntropyLoss(),
    'optimizer': optim.Adam,
    'shuffle': True,
    'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    'trainers_to_use': ['pso', 'bp'],
    'pso_options': {'c1': 1.49, 'c2': 1.49, 'w': (0.4, 0.9),
                    'use_pyswarms': False,
                    'particle_clamp': (-1, 1), 'use_particle_clamp_each_iteration': False,
                    'unchanged_iterations_stop': 20000, 'use_only_early_stopping': False
                    # no early stopping used, that is why 20k
                    },
    'pso_velocity_clamp': (-1, 1),
    'n_particles': 100,
    'pso_iters': 5000,
    'pso_optimizer': PSO,
}
CONFIG['cv'] = StratifiedKFold(n_splits=CONFIG['n_splits'], shuffle=True, random_state=CONFIG['random_state']) \
    if CONFIG['n_repeats'] == 1 else RepeatedStratifiedKFold(n_splits=CONFIG['n_splits'],
                                                             n_repeats=CONFIG['n_repeats'],
                                                             random_state=CONFIG['random_state'])
INPUT_FEATURES = torch.load(CONFIG['labels_features_common_name'] + "_features.tr").numpy()
INPUT_LABELS = torch.load(CONFIG['labels_features_common_name'] + "_labels.tr").numpy()
make_experiment_reproducible(CONFIG['random_state'])


def run_cross_validation_psobp(file_to_print) -> torch.Tensor:
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
    CONFIG['pso_options']['print_info'] = print_info

    for train_index, test_index in CONFIG['cv'].split(INPUT_FEATURES, INPUT_LABELS):
        fold_number += 1
        if fold_number % CONFIG['n_splits'] == 0:
            fold_number = 0
            repeat_number += 1

        def print_info(info):
            logger.info(info)
            print(info)
            file_to_print.write(info + "\n")

        print_info("New " + str(fold_number) + " fold. repeat = " + str(repeat_number))
        train_features, train_labels = INPUT_FEATURES[train_index], INPUT_LABELS[train_index]
        test_features, test_labels = INPUT_FEATURES[test_index], INPUT_LABELS[test_index]

        scaler = preprocessing.StandardScaler().fit(train_features)
        train_features = scaler.transform(train_features)
        test_features = scaler.transform(test_features)

        train_features = torch.from_numpy(train_features)
        train_labels = torch.from_numpy(train_labels)
        test_features = torch.from_numpy(test_features)
        test_labels = torch.from_numpy(test_labels)

        model = Model().to(CONFIG['device'])

        best_accuracy_bp, best_accuracy_pso = None, None
        for trainer_type in CONFIG['trainers_to_use']:
            if trainer_type == 'bp':
                best_accuracy_bp = train_bp(model, train_features, train_labels, test_features, test_labels, CONFIG)
            elif trainer_type == 'pso':
                best_accuracy_pso = train_pso(model, train_features, train_labels, test_features, test_labels, CONFIG)
        print_info('Best accuracy pso: ' + str(best_accuracy_pso) + ', bp: ' + str(best_accuracy_bp))
        best_accuracy = max(best_accuracy_bp, best_accuracy_pso)
        accuracies[fold_number][repeat_number] = best_accuracy
        print_info('Best final accuracy: ' + str(best_accuracy))
        print_info('END OF EVALUATION OF ' + str(fold_number) + ' FOLD, REPEAT ' + str(repeat_number))
        print_info('=========================================================================')
    logger.info("END run_cross_validation")
    return accuracies


def conduct_10_fold_cv_psobp_experiment():
    with open("./" + CONFIG['experiment_name'] + "_" + str(datetime.datetime.now()), 'w') as f:
        f.write("Config: " + str(CONFIG) + "\n")

        start = time.time()
        accuracies = run_cross_validation_psobp(f)
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
    conduct_10_fold_cv_psobp_experiment()
