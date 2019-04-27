from functools import reduce

import numpy as np
import torch
from torch import nn

from psob_authorship.pso.DecreasingWeightPsoOptimizer import DecreasingWeightPsoOptimizer


class PSO:
    def __init__(self, model: nn.Module, criterion, options, n_particles) -> None:
        super().__init__()
        self.criterion = criterion
        self.model = model
        self.options = options
        self.n_particles = n_particles

    def get_best_test_loss_and_acc(self, test_features: torch.Tensor, test_labels: torch.Tensor):
        def semi_applied_func(particle):
            self.set_model_weights(particle)
            with torch.no_grad():
                outputs = self.model(test_features)
                correct = 0
                total = 0
                _, predicted = torch.max(outputs.data, 1)
                total += test_labels.size(0)
                correct += (predicted == test_labels).sum().item()

                loss = self.criterion(outputs, test_labels).item()
                accuracy = correct / total
            return loss, accuracy
        return semi_applied_func

    def optimize(self, train_features: torch.Tensor, train_labels: torch.Tensor,
                 test_features: torch.Tensor, test_labels: torch.Tensor,
                 iters: int, velocity_clamp):
        def f_to_optimize(particles):
            losses = []
            for particle in particles:
                self.set_model_weights(particle)
                with torch.no_grad():
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

        loss, best_params = optimizer.optimize(f_to_optimize, iters=iters,
                                               test_loss_and_acc=
                                               self.get_best_test_loss_and_acc(test_features, test_labels))
        self.set_model_weights(best_params)
        return loss, best_params

    def set_model_weights(self, particle):
        start = 0
        for param in self.model.parameters():
            gap = reduce(lambda x, y: x * y, param.shape)
            param.data = torch.tensor(particle[start:start + gap].reshape(param.shape)).type(torch.FloatTensor)
            start += gap
