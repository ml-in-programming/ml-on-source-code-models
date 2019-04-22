from functools import reduce

import numpy as np
import torch
from torch import nn
import pyswarms as ps


class PSO:
    def __init__(self, model: nn.Module, criterion, options, n_particles) -> None:
        super().__init__()
        self.criterion = criterion
        self.model = model
        self.options = options
        self.n_particles = n_particles

    def optimize(self, train_features: torch.Tensor, train_labels: torch.Tensor, iters: int, bounds):
        def f_to_optimize(particles):
            losses = []
            for particle in particles:
                self.set_model_weights(particle)
                with torch.no_grad():
                    outputs = self.model(train_features)
                    loss = self.criterion(outputs, train_labels)
                    losses.append(loss)
            return np.array(losses)

        optimizer = ps.single.GlobalBestPSO(n_particles=self.n_particles, dimensions=self.model.dimensions,
                                            options=self.options, bounds=bounds)

        loss, best_params = optimizer.optimize(f_to_optimize, iters=iters)
        self.set_model_weights(best_params)
        return loss, best_params

    def set_model_weights(self, particle):
        start = 0
        for param in self.model.parameters():
            gap = reduce(lambda x, y: x*y, param.shape)
            param.data = torch.tensor(particle[start:start + gap].reshape(param.shape)).type(torch.FloatTensor)
            start += gap
