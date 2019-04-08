import datetime
import time

import skorch
import torch
import torch.optim as optim
from sklearn.model_selection import GridSearchCV
from skorch import NeuralNetClassifier
from torch import nn

from psob_authorship.model import Model
from psob_authorship.model.Model import Model

CONFIG = {
    'labels_features_common_name': "../calculated_features/without_5",
    'epochs': 5000,
    'batch_size': 32,
    'early_stopping_rounds': 350,
    'lr': 0.02,
    'cv': 3,
    'scoring': "accuracy",
    'params': {
        'optimizer__momentum': [0.85, 0.9, 0.95, 0.99],
        # 'criterion': [torch.nn.CrossEntropyLoss, torch.nn.NLLLoss, torch.nn.MSELoss]
    }
}
INPUT_FEATURES = torch.load(CONFIG['labels_features_common_name'] + "_features.tr").numpy()
INPUT_LABELS = torch.load(CONFIG['labels_features_common_name'] + "_labels.tr").numpy()


def grid_search_hyperparameters() -> GridSearchCV:
    net = NeuralNetClassifier(Model, optimizer=optim.SGD,
                              max_epochs=CONFIG['epochs'], lr=CONFIG['lr'],
                              criterion=torch.nn.CrossEntropyLoss, batch_size=CONFIG['batch_size'], verbose=1,
                              callbacks=[skorch.callbacks.EarlyStopping(monitor='valid_acc',
                                                                        patience=CONFIG['early_stopping_rounds'],
                                                                        lower_is_better=False)])
    gs = GridSearchCV(net, CONFIG['params'], scoring=CONFIG['scoring'], cv=CONFIG['cv'], n_jobs=-1)
    return gs.fit(INPUT_FEATURES, INPUT_LABELS)


def conduct_grid_search_experiment():
    start = time.time()
    grid_result = grid_search_hyperparameters()
    end = time.time()
    execution_time = end - start
    with open("../experiment_result/grid_search_result", 'w') as f:
        f.write("Execution time: " + str(datetime.timedelta(seconds=execution_time)) + "\n")
        f.write("Config: " + str(CONFIG) + "\n")
        f.write("Best score: " + str(grid_result.best_score_) + "\n")
        f.write("Best params: " + str(grid_result.best_params_))


if __name__ == '__main__':
    conduct_grid_search_experiment()
