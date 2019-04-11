import datetime
import time

import skorch
import torch
import torch.optim as optim
from sklearn.model_selection import GridSearchCV
from skorch import NeuralNetClassifier
from torch import nn
from typing import Tuple

from psob_authorship.experiment.utils import make_experiment_reproducible
from psob_authorship.model import Model
from psob_authorship.model.Model import Model

CONFIG = {
    'experiment_name': 'learning_rate_grid_search',
    'labels_features_common_name': "../calculated_features/without_5",
    'epochs': 5000,
    'batch_size': 32,
    'early_stopping_rounds': 350,
    'cv': 10,
    'scoring': 'accuracy',
    'criterion': torch.nn.CrossEntropyLoss,
    'momentum': 0.9,
    'random_state': 4562,
    'params': {
        'lr': [0.025]
        # 'lr': [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3]
    }
}
INPUT_FEATURES = torch.load(CONFIG['labels_features_common_name'] + "_features.tr").numpy()
INPUT_LABELS = torch.load(CONFIG['labels_features_common_name'] + "_labels.tr").numpy()
make_experiment_reproducible(CONFIG['random_state'])


def find_best_learning_rate() -> Tuple[float, float]:
    net = NeuralNetClassifier(Model, optimizer=optim.SGD, optimizer__momentum=0.9, max_epochs=CONFIG['epochs'], lr=0.1,
                              criterion=torch.nn.CrossEntropyLoss, batch_size=CONFIG['batch_size'], verbose=1,
                              callbacks=[skorch.callbacks.EarlyStopping(monitor='valid_acc',
                                                                        patience=CONFIG['early_stopping_rounds'],
                                                                        lower_is_better=False)])
    gs = GridSearchCV(net, CONFIG['params'], scoring=CONFIG['scoring'], cv=CONFIG['cv'])#, n_jobs=-1)
    grid_result = gs.fit(INPUT_FEATURES, INPUT_LABELS)
    return grid_result.best_score_, grid_result.best_params_


def conduct_learning_rate_experiment():
    start = time.time()
    best_score, best_lr = find_best_learning_rate()
    end = time.time()
    execution_time = end - start
    with open("../experiment_result/" + CONFIG['experiment_name'], 'w') as f:
        f.write("Execution time: " + str(datetime.timedelta(seconds=execution_time)) + "\n")
        f.write("Config: " + str(CONFIG) + "\n")
        f.write("Best score: " + str(best_score) + "\n")
        f.write("Best learning rate: " + str(best_lr))


if __name__ == '__main__':
    conduct_learning_rate_experiment()
