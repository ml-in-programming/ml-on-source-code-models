"""
Ackley function for test:
https://pyswarms.readthedocs.io/en/latest/api/pyswarms.utils.functions.html#module-pyswarms.utils.functions.single_obj
It is just simple function: R^n -> R
Thresholds are 1e-14.
"""
import numpy as np
import pyswarms as ps
from pyswarms.utils.functions.single_obj import ackley

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
    'velocity_clamp': (-1, 1),
    'pso_iters': 1000,
    'function_value_threshold': 1e-14,
    'point_threshold': 1e-14
}
make_experiment_reproducible(CONFIG['random_state'])


def conduct_ackley_comparison_experiment():
    dimensions = 10
    correct_function_value, correct_point = 0, np.zeros(dimensions)

    # pyswarms implementation
    pyswarms_optimizer = ps.single.GlobalBestPSO(n_particles=CONFIG['n_particles'], dimensions=dimensions,
                                                 options=CONFIG['pso_options'], velocity_clamp=CONFIG['velocity_clamp'])
    pyswarms_function_value, pyswarms_point = pyswarms_optimizer.optimize(ackley, iters=CONFIG['pso_iters'])

    # my implementation
    CONFIG['pso_options']['w'] = (CONFIG['pso_options']['w'], CONFIG['pso_options']['w'])
    my_optimizer = DecreasingWeightPsoOptimizer(CONFIG['n_particles'], dimensions,
                                                CONFIG['pso_options'], CONFIG['velocity_clamp'])
    my_function_value, my_point = my_optimizer.optimize(ackley, CONFIG['pso_iters'], None)

    point_diff = np.linalg.norm(correct_point - my_point)
    function_value_diff = abs(correct_function_value - my_function_value)
    assert (point_diff <= CONFIG['point_threshold'])
    assert (function_value_diff <= CONFIG['function_value_threshold'])
    print("ASSERTIONS PASSED")
    print("Thresholds for point " +
          str(CONFIG['point_threshold']) + ", for function value " +
          str(CONFIG['function_value_threshold']))
    print("Correct :" + str(correct_function_value) + "   " + str(correct_point))
    print("My      :" + str(my_function_value) + "   " + str(my_point))
    print("Pyswarms:" + str(pyswarms_function_value) + "   " + str(pyswarms_point))
    print("Point difference: " + str(point_diff))
    print("Function value difference: " + str(function_value_diff))


if __name__ == '__main__':
    conduct_ackley_comparison_experiment()
