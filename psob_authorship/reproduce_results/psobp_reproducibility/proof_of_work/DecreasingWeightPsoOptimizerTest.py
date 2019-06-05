import unittest

import numpy as np

from psob_authorship.pso.DecreasingWeightPsoOptimizer import DecreasingWeightPsoOptimizer


class DecreasingWeightPsoOptimizerTest(unittest.TestCase):
    """
    Tests for DecreasingWeightPsoOptimizer class.
    See iris_dataset.py and ackley.py for tests of optimize method of DecreasingWeightPsoOptimizerTest
    """
    def test_initialize_particles_is_in_bounds(self):
        n_particles = 10
        dimensions = 15

        velocity_clamp = None
        options = {'particle_clamp': (0, 1)}
        optimizer = DecreasingWeightPsoOptimizer(n_particles, dimensions, options, velocity_clamp)
        optimizer.initialize_particles()
        self.assertTrue(np.all(0 <= optimizer.particles) and np.all(optimizer.particles < 1))
        self.assertTrue(np.all(0 <= optimizer.velocities) and np.all(optimizer.velocities < 1))

        velocity_clamp = (10, 20)
        options = {'particle_clamp': (40, 50)}
        optimizer = DecreasingWeightPsoOptimizer(n_particles, dimensions, options, velocity_clamp)
        optimizer.initialize_particles()
        self.assertTrue(np.all(40 <= optimizer.particles) and np.all(optimizer.particles < 50))
        self.assertTrue(np.all(10 <= optimizer.velocities) and np.all(optimizer.velocities < 20))
