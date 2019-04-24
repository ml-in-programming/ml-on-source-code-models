import numpy as np
import torch
from tqdm import tqdm


class DecreasingWeightPsoOptimizer:
    def __init__(self, n_particles, dimensions, options, bounds) -> None:
        super().__init__()
        self.n_particles = n_particles
        self.dimensions = dimensions
        self.options = options
        self.bounds = bounds
        self.particles = None
        self.velocities = None

    def optimize(self, f, iters):
        self.initialize_particles()
        pbs = np.copy(self.particles)
        pbs_loss = f(pbs)
        pg = pbs[np.argmin(pbs_loss)]
        pg_loss = np.min(pbs_loss)
        w_min = self.options['w'][0]
        w_max = self.options['w'][1]
        r1, r2 = np.random.uniform(), np.random.uniform()
        for i in tqdm(range(iters)):
            w = w_max - (w_max - w_min) / iters * i
            self.velocities = w * self.velocities + \
                              self.options['c1'] * np.random.uniform(0, 1, size=self.particles.shape) * (pbs - self.particles) + \
                              self.options['c2'] * np.random.uniform(0, 1, size=self.particles.shape) * (pg - self.particles)
            self.particles = self.particles + self.velocities
            if self.bounds is not None:
                self.velocities = np.clip(self.velocities, self.bounds[0], self.bounds[1])
            new_loss = f(self.particles)
            pbs_with_new_loss = np.vstack((pbs_loss, new_loss))
            pbs_loss = np.min(pbs_with_new_loss, axis=0)
            new_or_not = np.argmin(pbs_with_new_loss, axis=0)
            pbs = np.stack((pbs, self.particles), axis=1)[np.arange(pbs.shape[0]), new_or_not]
            pg = pbs[np.argmin(pbs_loss)]
            pg_loss_prev = pg_loss
            pg_loss = np.min(pbs_loss)
            print("Loss: " + str(pg_loss))
            # if pg_loss_prev == pg_loss:
            #    return pg
        return pg_loss, pg

    def initialize_particles(self):
        self.particles = np.random.uniform(
            low=0.0, high=1.0, size=(self.n_particles, self.dimensions)
        )
        if self.bounds is not None:
            self.velocities = np.random.uniform(low=self.bounds[0], high=self.bounds[1],
                                                size=(self.n_particles, self.dimensions))
        else:
            self.velocities = np.random.random_sample(size=(self.n_particles, self.dimensions))
