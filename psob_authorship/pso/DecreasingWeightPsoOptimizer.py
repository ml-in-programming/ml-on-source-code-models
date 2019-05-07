import numpy as np


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
        r1, r2 = np.random.uniform(), np.random.uniform()
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
