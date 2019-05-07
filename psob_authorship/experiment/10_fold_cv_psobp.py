import datetime
import logging
import os
import time

import torch
from sklearn import preprocessing
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from torch import optim, nn

from psob_authorship.experiment.utils import make_experiment_reproducible
from psob_authorship.features.utils import configure_logger_by_default
from psob_authorship.model.Model import Model
from psob_authorship.pso.PSO import PSO
from psob_authorship.train.train_bp import train_bp
from psob_authorship.train.train_pso import train_pso

CONFIG = {
    'experiment_name': os.path.basename(__file__).split('.')[0],
    'experiment_notes': "reproduction of results, all params as in paper, r1 and r2 are random vectors generated each "
                        "iteration",
    'number_of_authors': 40,
    'labels_features_common_name': "../calculated_features/extracted_for_each_file",
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
                    'unchanged_iterations_stop': 20000,  'use_only_early_stopping': False
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
    with open("../experiment_result/" + CONFIG['experiment_name'] + "_" + str(datetime.datetime.now()), 'w') as f:
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
