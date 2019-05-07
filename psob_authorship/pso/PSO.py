from functools import reduce

import numpy as np
import torch
from torch import nn

from psob_authorship.pso.DecreasingWeightPsoOptimizer import DecreasingWeightPsoOptimizer
from psob_authorship.train.utils import print_100th_checkpoint_evaluation


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
