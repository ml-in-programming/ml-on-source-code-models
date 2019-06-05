import unittest

import numpy as np
import torch
from torch import nn

from psob_authorship.pso.PSO import PSO


class PsoUnitTest(unittest.TestCase):
    """
    Tests for PSO class.
    See iris_dataset_pytorch.py for tests of optimize method
    """
    def test_set_model_weights(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_hidden = nn.Linear(19, 150, bias=True)
                self.hidden_output = nn.Linear(150, 40, bias=True)
                self.dimensions = 19 * 150 + 150 + 150 * 40 + 40

            def forward(self, x):
                pass

        model = Model()
        pso = PSO(model, None, None, None)

        zeros = np.zeros(model.dimensions)
        pso.set_model_weights(zeros)
        expected_params = [torch.zeros((150, 19)), torch.zeros((150,)), torch.zeros((40, 150)), torch.zeros((40,))]
        model_params = list(model.parameters())
        for i in range(len(expected_params)):
            self.assertTrue(torch.all(expected_params[i] == model_params[i]))

        ones = np.ones(model.dimensions)
        pso.set_model_weights(ones)
        expected_params = [torch.ones((150, 19)), torch.ones((150,)), torch.ones((40, 150)), torch.ones((40,))]
        model_params = list(model.parameters())
        for i in range(len(expected_params)):
            self.assertTrue(torch.all(expected_params[i] == model_params[i]))

        range_to_n = np.arange(model.dimensions)
        pso.set_model_weights(range_to_n)
        expected_params = [torch.arange(0, 150 * 19).reshape((150, 19)).float(),
                           torch.arange(150 * 19, 150 * 19 + 150).reshape((150,)).float(),
                           torch.arange(150 * 19 + 150, 150 * 19 + 150 + 40 * 150).reshape((40, 150)).float(),
                           torch.arange(150 * 19 + 150 + 40 * 150, 150 * 19 + 150 + 40 * 150 + 40).reshape((40,)).float()]
        model_params = list(model.parameters())
        for i in range(len(expected_params)):
            self.assertTrue(torch.all(expected_params[i] == model_params[i]))
