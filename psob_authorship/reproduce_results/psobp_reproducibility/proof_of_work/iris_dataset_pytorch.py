"""
Example is taken from https://pyswarms.readthedocs.io/en/latest/examples/custom_objective_function.html
Comparison between my implementation of PSO and pyswarms is made on iris dataset.
Optimizing PyTorch model.
Assert is taken on absolute difference in final accuracy with 0.015 threshold.
Also for train loss threshold is 0.03.
Thresholds and results are different because instead of softmax logsoftmax is used.
"""
# Import modules
# Import PySwarms
import torch
from sklearn.datasets import load_iris
from torch import nn

from psob_authorship.experiment.utils import make_experiment_reproducible
from psob_authorship.pso.PSO import PSO
from psob_authorship.train.utils import get_model_accuracy_and_loss

CONFIG = {
    'random_state': 4562,
    'criterion': nn.CrossEntropyLoss(),
    'pso_options': {'c1': 0.5, 'c2': 0.3, 'w': 0.9,
                    'particle_clamp': (0, 1), 'use_particle_clamp_each_iteration': False,
                    'unchanged_iterations_stop': 20000, 'use_only_early_stopping': False
                    # 20k not to use early stopping, so exactly 1000 iterations will be performed
                    },
    'n_particles': 100,
    'pso_iters': 1000,
    'pso_optimizer': PSO,
    'train_loss_threshold': 0.03,
    'accuracy_threshold': 0.015
}
make_experiment_reproducible(CONFIG['random_state'])


def print_info(string):
    print(string)


CONFIG['pso_options']['print_info'] = print_info


class Model(nn.Module):
    # Neural network architecture
    n_inputs = 4
    n_hidden = 20
    n_classes = 3
    dimensions = (n_inputs * n_hidden) + (n_hidden * n_classes) + n_hidden + n_classes

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(Model.n_inputs, Model.n_hidden, bias=True)
        self.nonlin1 = nn.Tanh()
        self.fc2 = nn.Linear(Model.n_hidden, Model.n_classes, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.nonlin1(x)
        x = self.fc2(x)
        return x


def conduct_iris_dataset_pytorch_comparison_experiment():
    # Load the iris dataset
    data = load_iris()

    # Store the features as X and the labels as y
    X = torch.FloatTensor(data.data)
    y = torch.LongTensor(data.target)

    model = Model()
    criterion = CONFIG['criterion']

    def pso_optimize(use_pyswarms):
        CONFIG['pso_options']['use_pyswarms'] = use_pyswarms
        optimizer = CONFIG['pso_optimizer'](model, criterion, CONFIG['pso_options'], CONFIG['n_particles'])
        optimizer.optimize(X, y,
                           X, y,
                           CONFIG['pso_iters'], None)

    # pyswarms
    pso_optimize(use_pyswarms=True)
    pyswarms_cost, pyswarms_accuracy = get_model_accuracy_and_loss(model, criterion, X, y)

    # my implementation
    CONFIG['pso_options']['w'] = (CONFIG['pso_options']['w'], CONFIG['pso_options']['w'])
    pso_optimize(use_pyswarms=False)
    my_cost, my_accuracy = get_model_accuracy_and_loss(model, criterion, X, y)

    accuracy_diff = abs(pyswarms_accuracy - my_accuracy)
    cost_diff = abs(pyswarms_cost - my_cost)
    assert (accuracy_diff <= CONFIG['accuracy_threshold'])
    assert (cost_diff <= CONFIG['train_loss_threshold'])
    print("ASSERTIONS PASSED")
    print("Thresholds for accuracy " +
          str(CONFIG['accuracy_threshold']) + ", for train loss " +
          str(CONFIG['train_loss_threshold']))
    print("My      :" + str(my_accuracy) + "   " + str(my_cost))
    print("Pyswarms:" + str(pyswarms_accuracy) + "   " + str(pyswarms_cost))
    print("Accuracy difference: " + str(accuracy_diff))
    print("Loss difference: " + str(cost_diff))


if __name__ == '__main__':
    conduct_iris_dataset_pytorch_comparison_experiment()
