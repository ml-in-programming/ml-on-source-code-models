"""
Example is taken from https://pyswarms.readthedocs.io/en/latest/examples/custom_objective_function.html
Comparison between my implementation of PSO and pyswarms is made on iris dataset.
Assert is taken on absolute difference in final accuracy with 0.01 threshold.
Also for train loss threshold is 0.01 too.
"""
# Import modules
import numpy as np
# Import PySwarms
import pyswarms as ps
from sklearn.datasets import load_iris

from psob_authorship.experiment.utils import make_experiment_reproducible
from psob_authorship.pso.DecreasingWeightPsoOptimizer import DecreasingWeightPsoOptimizer

CONFIG = {
    'random_state': 4562,
    'pso_options': {'c1': 0.5, 'c2': 0.3, 'w': 0.9,
                    'particle_clamp': (0, 1), 'use_particle_clamp_each_iteration': False,
                    'unchanged_iterations_stop': 20000, 'use_only_early_stopping': False
                    # 20k not to use early stopping, so exactly 1000 iterations will be performed
                    },
    'n_particles': 100,
    'pso_iters': 1000,
    'train_loss_threshold': 0.01,
    'accuracy_threshold': 0.01
}

make_experiment_reproducible(CONFIG['random_state'])


def conduct_iris_dataset_comparison_experiment():
    # Load the iris dataset
    data = load_iris()

    # Store the features as X and the labels as y
    X = data.data
    y = data.target

    # Forward propagation
    def forward_prop(params):
        """Forward propagation as objective function

        This computes for the forward propagation of the neural network, as
        well as the loss. It receives a set of parameters that must be
        rolled-back into the corresponding weights and biases.

        Inputs
        ------
        params: np.ndarray
            The dimensions should include an unrolled version of the
            weights and biases.

        Returns
        -------
        float
            The computed negative log-likelihood loss given the parameters
        """
        # Neural network architecture
        n_inputs = 4
        n_hidden = 20
        n_classes = 3

        # Roll-back the weights and biases
        W1 = params[0:80].reshape((n_inputs, n_hidden))
        b1 = params[80:100].reshape((n_hidden,))
        W2 = params[100:160].reshape((n_hidden, n_classes))
        b2 = params[160:163].reshape((n_classes,))

        # Perform forward propagation
        z1 = X.dot(W1) + b1  # Pre-activation in Layer 1
        a1 = np.tanh(z1)  # Activation in Layer 1
        z2 = a1.dot(W2) + b2  # Pre-activation in Layer 2
        logits = z2  # Logits for Layer 2

        # Compute for the softmax of the logits
        exp_scores = np.exp(logits)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # Compute for the negative log likelihood
        N = 150  # Number of samples
        corect_logprobs = -np.log(probs[range(N), y])
        loss = np.sum(corect_logprobs) / N

        return loss

    def f(x):
        """Higher-level method to do forward_prop in the
        whole swarm.

        Inputs
        ------
        x: numpy.ndarray of shape (n_particles, dimensions)
            The swarm that will perform the search

        Returns
        -------
        numpy.ndarray of shape (n_particles, )
            The computed loss for each particle
        """
        n_particles = x.shape[0]
        j = [forward_prop(x[i]) for i in range(n_particles)]
        return np.array(j)

    dimensions = (4 * 20) + (20 * 3) + 20 + 3

    # pyswarms
    optimizer = ps.single.GlobalBestPSO(n_particles=CONFIG['n_particles'], dimensions=dimensions,
                                        options=CONFIG['pso_options'])
    pyswarms_cost, pyswarms_pos = optimizer.optimize(f, iters=CONFIG['pso_iters'])

    # my implementation
    CONFIG['pso_options']['w'] = (CONFIG['pso_options']['w'], CONFIG['pso_options']['w'])
    my_optimizer = DecreasingWeightPsoOptimizer(CONFIG['n_particles'], dimensions, CONFIG['pso_options'], None)
    my_cost, my_pos = my_optimizer.optimize(f, CONFIG['pso_iters'], None)

    def predict(X, pos):
        """
        Use the trained weights to perform class predictions.

        Inputs
        ------
        X: numpy.ndarray
            Input Iris dataset
        pos: numpy.ndarray
            Position matrix found by the swarm. Will be rolled
            into weights and biases.
        """
        # Neural network architecture
        n_inputs = 4
        n_hidden = 20
        n_classes = 3

        # Roll-back the weights and biases
        W1 = pos[0:80].reshape((n_inputs, n_hidden))
        b1 = pos[80:100].reshape((n_hidden,))
        W2 = pos[100:160].reshape((n_hidden, n_classes))
        b2 = pos[160:163].reshape((n_classes,))

        # Perform forward propagation
        z1 = X.dot(W1) + b1  # Pre-activation in Layer 1
        a1 = np.tanh(z1)  # Activation in Layer 1
        z2 = a1.dot(W2) + b2  # Pre-activation in Layer 2
        logits = z2  # Logits for Layer 2

        y_pred = np.argmax(logits, axis=1)
        return y_pred

    pyswarms_accuracy = (predict(X, pyswarms_pos) == y).mean()
    my_accuracy = (predict(X, my_pos) == y).mean()
    accuracy_diff = abs(pyswarms_accuracy - my_accuracy)
    cost_diff = abs(pyswarms_cost - my_cost)
    assert(accuracy_diff <= CONFIG['accuracy_threshold'])
    assert(cost_diff <= CONFIG['train_loss_threshold'])
    print("ASSERTIONS PASSED")
    print("Thresholds for accuracy " +
          str(CONFIG['accuracy_threshold']) + ", for train loss " +
          str(CONFIG['train_loss_threshold']))
    print("My      :" + str(my_accuracy) + "   " + str(my_cost))
    print("Pyswarms:" + str(pyswarms_accuracy) + "   " + str(pyswarms_cost))
    print("Accuracy difference: " + str(accuracy_diff))
    print("Loss difference: " + str(cost_diff))


if __name__ == '__main__':
    conduct_iris_dataset_comparison_experiment()
